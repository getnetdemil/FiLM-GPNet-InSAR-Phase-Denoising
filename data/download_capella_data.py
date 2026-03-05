"""
Download Capella Space SAR Data for DFC 2026
=============================================
Downloads GEO (geocoded) SAR imagery from Capella Open Data via STAC catalog
or direct S3 access. Supports downloading by pair manifest, by region, or
by specific collect IDs.

Usage:
    # Download top stacks by score
    python download_capella_data.py --scores stac_results/stack_scores.csv \
        --n_stacks 5 --output_dir ../data/raw

    # Download specific pairs from manifest
    python download_capella_data.py --manifest stac_results/pair_manifest.json \
        --output_dir ../data/raw --asset_type GEO

    # Direct S3 download (no auth required)
    python download_capella_data.py --use_s3 --collect_ids id1,id2 \
        --output_dir ../data/raw
"""

import argparse
import json
import logging
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Asset types available in Capella Open Data
ASSET_TYPES = {
    "GEO": "Geocoded amplitude image (most common, use this for DL)",
    "GEC": "Geocoded ellipsoid corrected (reduced terrain artifacts)",
    "SLC": "Single look complex (phase information, for InSAR)",
    "CPHD": "Complex phase history (raw, for custom processing)",
    "thumbnail": "Quick-look PNG for visual inspection",
    "metadata": "JSON metadata sidecar",
}

CAPELLA_S3_BUCKET = "capella-open-data"


def download_from_stac(
    pair_manifest: list,
    output_dir: Path,
    asset_type: str = "GEO",
    n_workers: int = 4,
    max_pairs: Optional[int] = None,
) -> list:
    """
    Download SAR images using STAC item URLs.

    Args:
        pair_manifest: List of pair dicts from explore_stac.py
        output_dir: Directory to save downloaded files
        asset_type: Which asset to download (GEO recommended for DL)
        n_workers: Number of parallel download threads
        max_pairs: Limit number of pairs to download (for testing)

    Returns:
        List of successfully downloaded file paths
    """
    try:
        import requests
        from pystac_client import Client
    except ImportError:
        raise ImportError("Install: pip install pystac-client requests")

    pairs_to_download = pair_manifest[:max_pairs] if max_pairs else pair_manifest
    downloaded = []

    def download_item(collect_id: str, stac_url: str, out_dir: Path) -> Optional[Path]:
        """Download a single SAR acquisition."""
        item_dir = out_dir / collect_id
        item_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Fetch STAC item
            response = requests.get(stac_url, timeout=30)
            response.raise_for_status()
            item = response.json()

            # Get asset URL
            assets = item.get("assets", {})
            if asset_type not in assets:
                available = list(assets.keys())
                logger.warning(
                    f"{collect_id}: Asset '{asset_type}' not found. Available: {available}"
                )
                # Try thumbnail for verification
                if "thumbnail" in assets:
                    asset_url = assets["thumbnail"]["href"]
                    fname = item_dir / f"{collect_id}_thumbnail.png"
                else:
                    return None
            else:
                asset_url = assets[asset_type]["href"]
                ext = ".tif" if "GEO" in asset_type or "GEC" in asset_type else ".nc"
                fname = item_dir / f"{collect_id}_{asset_type}{ext}"

            # Skip if already downloaded
            if fname.exists():
                logger.info(f"Already exists: {fname.name}")
                return fname

            # Download
            logger.info(f"Downloading {fname.name} ...")
            r = requests.get(asset_url, stream=True, timeout=120)
            r.raise_for_status()

            total = int(r.headers.get("content-length", 0))
            downloaded_bytes = 0
            with open(fname, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded_bytes += len(chunk)

            size_mb = downloaded_bytes / 1024 / 1024
            logger.info(f"Downloaded {fname.name} ({size_mb:.1f} MB)")

            # Save metadata sidecar
            meta_path = item_dir / f"{collect_id}_metadata.json"
            with open(meta_path, "w") as f:
                json.dump(item, f, indent=2)

            return fname

        except Exception as e:
            logger.error(f"Failed to download {collect_id}: {e}")
            return None

    # Collect all unique collect IDs
    collect_items = {}
    for pair in pairs_to_download:
        for role in ["ref", "sec"]:
            cid = pair[f"collect_id_{role}"]
            url = pair[f"stac_url_{role}"]
            if cid and url and cid not in collect_items:
                collect_items[cid] = url

    logger.info(f"Downloading {len(collect_items)} unique acquisitions ({n_workers} workers)...")

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(download_item, cid, url, output_dir): cid
            for cid, url in collect_items.items()
        }
        for future in as_completed(futures):
            result = future.result()
            if result:
                downloaded.append(result)

    logger.info(f"Successfully downloaded {len(downloaded)} / {len(collect_items)} files")
    return downloaded


def download_from_s3(
    collect_ids: list,
    output_dir: Path,
    asset_type: str = "GEO",
) -> list:
    """
    Download SAR images directly from Capella's public S3 bucket.
    No authentication required.

    Args:
        collect_ids: List of Capella collect IDs
        output_dir: Directory to save files
        asset_type: Asset type to download

    Returns:
        List of downloaded file paths
    """
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError:
        raise ImportError("Install: pip install boto3")

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    downloaded = []

    for collect_id in collect_ids:
        item_dir = output_dir / collect_id
        item_dir.mkdir(parents=True, exist_ok=True)

        # S3 key pattern for Capella Open Data
        prefix = f"items/{collect_id}/"

        try:
            response = s3.list_objects_v2(
                Bucket=CAPELLA_S3_BUCKET, Prefix=prefix
            )
            objects = response.get("Contents", [])

            if not objects:
                logger.warning(f"No objects found for collect {collect_id}")
                continue

            for obj in objects:
                key = obj["Key"]
                filename = Path(key).name

                # Filter by asset type
                if asset_type not in filename.upper() and "metadata" not in filename.lower():
                    continue

                local_path = item_dir / filename
                if local_path.exists():
                    logger.info(f"Already exists: {filename}")
                    downloaded.append(local_path)
                    continue

                logger.info(f"Downloading s3://{CAPELLA_S3_BUCKET}/{key}")
                s3.download_file(CAPELLA_S3_BUCKET, key, str(local_path))
                size_mb = local_path.stat().st_size / 1024 / 1024
                logger.info(f"Downloaded {filename} ({size_mb:.1f} MB)")
                downloaded.append(local_path)

        except Exception as e:
            logger.error(f"S3 download failed for {collect_id}: {e}")

    return downloaded


def verify_downloads(download_dir: Path) -> pd.DataFrame:
    """
    Verify downloaded files and report statistics.

    Returns:
        DataFrame with file verification results
    """
    rows = []
    for collect_dir in sorted(download_dir.iterdir()):
        if not collect_dir.is_dir():
            continue

        tif_files = list(collect_dir.glob("*.tif"))
        meta_files = list(collect_dir.glob("*metadata.json"))

        for tif in tif_files:
            size_mb = tif.stat().st_size / 1024 / 1024
            rows.append(
                {
                    "collect_id": collect_dir.name,
                    "file": tif.name,
                    "size_mb": round(size_mb, 2),
                    "has_metadata": len(meta_files) > 0,
                    "status": "ok" if size_mb > 0.1 else "empty",
                }
            )

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if not df.empty:
        total_gb = df["size_mb"].sum() / 1024
        logger.info(f"Downloaded {len(df)} files, {total_gb:.2f} GB total")
        logger.info(f"  OK: {(df['status']=='ok').sum()}")
        logger.info(f"  Empty/corrupt: {(df['status']!='ok').sum()}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Download Capella SAR data for DFC 2026")
    parser.add_argument("--manifest", type=str, help="Path to pair_manifest.json")
    parser.add_argument("--scores", type=str, help="Path to stack_scores.csv")
    parser.add_argument("--collect_ids", type=str, help="Comma-separated collect IDs")
    parser.add_argument("--output_dir", type=str, default="../data/raw")
    parser.add_argument(
        "--asset_type",
        type=str,
        default="GEO",
        choices=list(ASSET_TYPES.keys()),
        help="Which SAR product to download",
    )
    parser.add_argument("--n_stacks", type=int, default=5, help="Number of top stacks to download")
    parser.add_argument("--max_pairs", type=int, default=None, help="Limit pairs for testing")
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--use_s3", action="store_true", help="Use direct S3 access")
    parser.add_argument("--verify_only", action="store_true", help="Only verify existing downloads")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Asset type: {args.asset_type} — {ASSET_TYPES[args.asset_type]}")

    if args.verify_only:
        df = verify_downloads(output_dir)
        if not df.empty:
            print(df.to_string(index=False))
        return

    if args.collect_ids and args.use_s3:
        collect_ids = [c.strip() for c in args.collect_ids.split(",")]
        download_from_s3(collect_ids, output_dir, args.asset_type)

    elif args.manifest:
        with open(args.manifest) as f:
            pairs = json.load(f)
        download_from_stac(pairs, output_dir, args.asset_type, args.n_workers, args.max_pairs)

    elif args.scores:
        df_scores = pd.read_csv(args.scores)
        top_stacks = df_scores.head(args.n_stacks)
        logger.info(f"Selected {len(top_stacks)} top stacks:")
        for _, row in top_stacks.iterrows():
            logger.info(
                f"  {row['stack_id']}: {row['n_acquisitions']} acquisitions, "
                f"score={row['score']}"
            )
        # Download thumbnails first for quick visual inspection
        logger.info("Tip: Download thumbnails first for visual inspection:")
        logger.info("  python download_capella_data.py --manifest pair_manifest.json --asset_type thumbnail")

    else:
        parser.print_help()
        logger.info("\nAsset types available:")
        for k, v in ASSET_TYPES.items():
            logger.info(f"  {k}: {v}")

    # Verify after download
    df_verify = verify_downloads(output_dir)
    if not df_verify.empty:
        df_verify.to_csv(output_dir / "download_manifest.csv", index=False)


if __name__ == "__main__":
    main()
