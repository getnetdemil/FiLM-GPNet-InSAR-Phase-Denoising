"""
Crawl the Capella STAC catalog for the IEEE GRSS 2026 contest, build a
metadata manifest of all SLC collects, then optionally download a subset.

Usage
-----
# 1. Build the full metadata index (no download):
python scripts/download_subset.py --index_only \
    --out_manifest data/manifests/full_index.parquet

# 2. Download SLC tif + extended JSON for selected AOIs:
python scripts/download_subset.py \
    --manifest data/manifests/full_index.parquet \
    --aoi_filter "Hawaii,SanFrancisco" \
    --mode_filter spotlight \
    --max_collects 80 \
    --out_dir data/raw/ \
    --n_workers 8
"""

import argparse
import concurrent.futures
import json
import logging
from pathlib import Path

import boto3
import pandas as pd
import pystac
from botocore import UNSIGNED
from botocore.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

STAC_ROOT = "https://capella-open-data.s3.us-west-2.amazonaws.com/stac"
CONTEST_COLLECTION = "capella-open-data-ieee-data-contest/collection.json"
S3_BUCKET = "capella-open-data"
S3_REGION = "us-west-2"


def get_s3_client():
    return boto3.client(
        "s3",
        region_name=S3_REGION,
        config=Config(signature_version=UNSIGNED),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capella STAC crawler + downloader.")
    p.add_argument("--stac_root", default=STAC_ROOT)
    p.add_argument("--out_manifest", default="data/manifests/full_index.parquet",
                   help="Output manifest path (.parquet or .csv).")
    p.add_argument("--index_only", action="store_true",
                   help="Only build the manifest, skip downloading.")
    p.add_argument("--manifest", default=None,
                   help="Existing manifest to use for download (skip re-crawl).")
    p.add_argument("--aoi_filter", default=None,
                   help="Comma-separated AOI names (matches 'aoi' column).")
    p.add_argument("--mode_filter", default=None,
                   help="Instrument mode filter, e.g. 'spotlight'.")
    p.add_argument("--orbit_filter", default=None,
                   help="Orbit state filter: 'ascending' or 'descending'.")
    p.add_argument("--max_collects", type=int, default=None,
                   help="Max number of SLC collects to download.")
    p.add_argument("--out_dir", default="data/raw/",
                   help="Root directory for downloaded files.")
    p.add_argument("--n_workers", type=int, default=4,
                   help="Parallel download workers.")
    p.add_argument("--assets", default="slc,metadata",
                   help="Comma-separated asset keys to download: slc,metadata,preview.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# STAC crawl
# ---------------------------------------------------------------------------

def _extract_item_record(item: pystac.Item) -> dict:
    """Pull key metadata fields from a STAC item into a flat dict."""
    p = item.properties
    lon = p.get("proj:centroid", {}).get("lon") or (item.bbox[0] + item.bbox[2]) / 2
    lat = p.get("proj:centroid", {}).get("lat") or (item.bbox[1] + item.bbox[3]) / 2

    assets = item.assets
    slc_href = assets["HH"].href if "HH" in assets else None
    meta_href = assets["metadata"].href if "metadata" in assets else None

    return {
        "id": item.id,
        "collect_id": p.get("capella:collect_id"),
        "datetime": str(item.datetime),
        "platform": p.get("platform"),
        "instrument_mode": p.get("sar:instrument_mode"),
        "orbit_state": p.get("sat:orbit_state"),
        "look_direction": p.get("sar:observation_direction"),
        "orbital_plane": p.get("capella:orbital_plane"),
        "incidence_angle_deg": p.get("view:incidence_angle"),
        "look_angle_deg": p.get("capella:look_angle"),
        "azimuth_deg": p.get("view:azimuth"),
        "squint_angle_deg": p.get("capella:squint_angle", 0),
        "center_freq_ghz": p.get("sar:center_frequency"),
        "px_spacing_rg_m": p.get("sar:pixel_spacing_range"),
        "px_spacing_az_m": p.get("sar:pixel_spacing_azimuth"),
        "lon": lon,
        "lat": lat,
        "bbox_w": item.bbox[0],
        "bbox_s": item.bbox[1],
        "bbox_e": item.bbox[2],
        "bbox_n": item.bbox[3],
        "slc_href": slc_href,
        "meta_href": meta_href,
        "stac_href": item.get_self_href(),
    }


def crawl_contest_collection(stac_root: str) -> pd.DataFrame:
    """Crawl all SLC items from the contest collection. Returns a DataFrame."""
    col_url = f"{stac_root}/{CONTEST_COLLECTION}"
    log.info("Loading contest collection from %s", col_url)
    col = pystac.Collection.from_file(col_url)

    item_links = [l for l in col.links if l.rel == "item"]
    slc_links = [l for l in item_links if "_SLC_" in l.href]
    log.info("Found %d SLC item links", len(slc_links))

    records = []
    for i, link in enumerate(slc_links):
        if i % 50 == 0:
            log.info("  Fetching item %d / %d ...", i, len(slc_links))
        try:
            item = pystac.Item.from_file(link.get_absolute_href())
            records.append(_extract_item_record(item))
        except Exception as e:
            log.warning("Failed to fetch %s: %s", link.href, e)

    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    log.info("Crawled %d SLC collects.", len(df))
    return df


def assign_aoi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cluster collects into named AOIs by geographic proximity.
    Uses a 0.5-degree grid to group spatially overlapping scenes.
    """
    df = df.copy()
    df["grid_lon"] = (df["lon"] / 0.5).round() * 0.5
    df["grid_lat"] = (df["lat"] / 0.5).round() * 0.5
    unique_grids = sorted(
        df[["grid_lon", "grid_lat"]].drop_duplicates().itertuples(index=False, name=None)
    )
    aoi_map = {grid: f"AOI_{idx:03d}" for idx, grid in enumerate(unique_grids)}
    df["aoi"] = df.apply(lambda r: aoi_map[(r["grid_lon"], r["grid_lat"])], axis=1)
    return df


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _s3_key_from_href(href: str) -> str:
    """Convert https://capella-open-data.s3.amazonaws.com/data/... → data/..."""
    marker = ".amazonaws.com/"
    return href[href.index(marker) + len(marker):]


def download_asset(s3, href: str, out_path: Path) -> bool:
    if out_path.exists():
        log.debug("Already exists: %s", out_path)
        return True
    out_path.parent.mkdir(parents=True, exist_ok=True)
    key = _s3_key_from_href(href)
    try:
        s3.download_file(S3_BUCKET, key, str(out_path))
        log.info("Downloaded %s", out_path.name)
        return True
    except Exception as e:
        log.error("Failed %s: %s", key, e)
        return False


def download_collect(row: pd.Series, out_dir: Path, asset_keys: list[str]) -> dict:
    s3 = get_s3_client()
    collect_dir = out_dir / row["aoi"] / row["id"]
    results = {}
    for key in asset_keys:
        href = row.get(f"{key}_href") if key != "slc" else row.get("slc_href")
        if key == "slc":
            href = row["slc_href"]
        elif key == "metadata":
            href = row["meta_href"]
        elif key == "preview":
            href = row.get("preview_href")
        if not href:
            continue
        fname = href.split("/")[-1]
        ok = download_asset(s3, href, collect_dir / fname)
        results[key] = ok
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- Build or load manifest ---
    if args.manifest and Path(args.manifest).exists():
        log.info("Loading existing manifest from %s", args.manifest)
        df = pd.read_parquet(args.manifest) if args.manifest.endswith(".parquet") \
            else pd.read_csv(args.manifest)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    else:
        raw_cache = Path(args.out_manifest).parent / "_raw_crawl_cache.parquet"
        if raw_cache.exists():
            log.info("Loading raw crawl cache from %s (skip re-fetch)", raw_cache)
            df = pd.read_parquet(raw_cache)
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        else:
            df = crawl_contest_collection(args.stac_root)
            # Cache raw crawl so re-running after a bug doesn't re-fetch 791 items
            raw_cache.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(raw_cache, index=False)
            log.info("Raw crawl cached to %s", raw_cache)

        df = assign_aoi(df)

        out = Path(args.out_manifest)
        out.parent.mkdir(parents=True, exist_ok=True)
        if str(out).endswith(".parquet"):
            df.to_parquet(out, index=False)
        else:
            df.to_csv(out, index=False)
        log.info("Manifest saved to %s", out)

        # Print AOI summary
        aoi_summary = df.groupby("aoi").agg(
            n_collects=("id", "count"),
            modes=("instrument_mode", lambda x: ",".join(sorted(set(x)))),
            orbits=("orbit_state", lambda x: ",".join(sorted(set(x)))),
            inc_min=("incidence_angle_deg", "min"),
            inc_max=("incidence_angle_deg", "max"),
            lon=("lon", "mean"),
            lat=("lat", "mean"),
        ).sort_values("n_collects", ascending=False)
        print("\n=== AOI Summary ===")
        print(aoi_summary.to_string())
        print(f"\nTotal SLC collects: {len(df)}")

    if args.index_only:
        log.info("--index_only set. Done.")
        return

    # --- Apply filters ---
    subset = df.copy()
    if args.aoi_filter:
        aois = [a.strip() for a in args.aoi_filter.split(",")]
        subset = subset[subset["aoi"].isin(aois)]
        log.info("AOI filter '%s': %d collects", args.aoi_filter, len(subset))
    if args.mode_filter:
        subset = subset[subset["instrument_mode"] == args.mode_filter]
        log.info("Mode filter '%s': %d collects", args.mode_filter, len(subset))
    if args.orbit_filter:
        subset = subset[subset["orbit_state"] == args.orbit_filter]
        log.info("Orbit filter '%s': %d collects", args.orbit_filter, len(subset))
    if args.max_collects:
        subset = subset.head(args.max_collects)
        log.info("Capped at %d collects", len(subset))

    log.info("Downloading %d collects to %s ...", len(subset), args.out_dir)
    out_dir = Path(args.out_dir)
    asset_keys = [a.strip() for a in args.assets.split(",")]

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_workers) as ex:
        futures = {
            ex.submit(download_collect, row, out_dir, asset_keys): row["id"]
            for _, row in subset.iterrows()
        }
        ok = fail = 0
        for fut in concurrent.futures.as_completed(futures):
            results = fut.result()
            if all(results.values()):
                ok += 1
            else:
                fail += 1

    log.info("Done. %d succeeded, %d failed.", ok, fail)

    # Save download manifest
    dl_manifest = out_dir / "download_manifest.parquet"
    subset.to_parquet(dl_manifest, index=False)
    log.info("Download manifest saved to %s", dl_manifest)


if __name__ == "__main__":
    main()
