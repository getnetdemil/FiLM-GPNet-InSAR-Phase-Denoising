"""
Explore Capella Space STAC Catalog for DFC 2026
================================================
Queries the Capella Open Data STAC catalog to find temporal stacks suitable
for change detection. Filters by temporal depth, geographic diversity, and
imaging geometry consistency.

Usage:
    python explore_stac.py --output_dir ./stac_results
    python explore_stac.py --region urban --min_acquisitions 5
"""

import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CAPELLA_STAC_URL = "https://stacindex.org/catalogs/capella-space-opendata"
CAPELLA_S3_BUCKET = "s3://capella-open-data/items"

# Known Capella Open Data STAC endpoint (from contest documentation)
CAPELLA_API_URL = "https://api.capellaspace.com/catalog"


def get_stac_client():
    """Initialize STAC client for Capella Open Data catalog."""
    try:
        from pystac_client import Client
        # Public endpoint — no authentication required
        client = Client.open(CAPELLA_STAC_URL)
        logger.info(f"Connected to STAC catalog: {client.title}")
        return client
    except ImportError:
        logger.error("pystac-client not installed. Run: pip install pystac-client")
        raise
    except Exception as e:
        logger.error(f"Failed to connect to STAC: {e}")
        logger.info("Try the alternative: python download_capella_data.py --use_s3")
        raise


def query_temporal_stacks(
    client,
    bbox: Optional[list] = None,
    date_range: Optional[tuple] = None,
    max_results: int = 500,
) -> pd.DataFrame:
    """
    Query all available SAR acquisitions and identify temporal stacks.

    Args:
        client: STAC client instance
        bbox: [west, south, east, north] bounding box filter
        date_range: (start_date, end_date) as ISO strings
        max_results: Maximum number of items to retrieve

    Returns:
        DataFrame with one row per acquisition, columns from CSV metadata
    """
    search_params = {"collections": ["capella-open-data"], "max_items": max_results}

    if bbox:
        search_params["bbox"] = bbox
    if date_range:
        search_params["datetime"] = f"{date_range[0]}/{date_range[1]}"

    logger.info("Querying STAC catalog...")
    items = list(client.search(**search_params).items())
    logger.info(f"Found {len(items)} items")

    records = []
    for item in items:
        props = item.properties
        records.append(
            {
                "collect_id": item.id,
                "datetime": props.get("datetime"),
                "mode": props.get("sar:instrument_mode", "unknown"),
                "polarization": props.get("sar:polarizations", []),
                "platform": props.get("platform", "unknown"),
                "incidence_angle": props.get("sar:center_incidence_angle"),
                "look_direction": props.get("sar:look_direction"),
                "flight_direction": props.get("sat:orbit_state"),
                "latitude": item.geometry["coordinates"][0][0][1] if item.geometry else None,
                "longitude": item.geometry["coordinates"][0][0][0] if item.geometry else None,
                "assets": list(item.assets.keys()),
                "stac_url": item.get_self_href(),
            }
        )

    df = pd.DataFrame(records)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime")

    logger.info(f"Parsed {len(df)} acquisitions")
    return df


def identify_temporal_stacks(df: pd.DataFrame, grid_size_deg: float = 0.5) -> dict:
    """
    Group acquisitions into temporal stacks by geographic proximity.

    Args:
        df: DataFrame from query_temporal_stacks
        grid_size_deg: Grid cell size in degrees for co-location grouping

    Returns:
        Dict mapping stack_id -> list of acquisitions
    """
    if df.empty:
        return {}

    # Round lat/lon to grid cells
    df = df.copy()
    df["lat_grid"] = (df["latitude"] / grid_size_deg).round() * grid_size_deg
    df["lon_grid"] = (df["longitude"] / grid_size_deg).round() * grid_size_deg
    df["grid_key"] = df["lat_grid"].astype(str) + "_" + df["lon_grid"].astype(str)

    stacks = {}
    for grid_key, group in df.groupby("grid_key"):
        if len(group) >= 2:
            stacks[grid_key] = group.to_dict("records")

    logger.info(f"Identified {len(stacks)} temporal stacks with ≥2 acquisitions")
    return stacks


def score_stacks(stacks: dict) -> pd.DataFrame:
    """
    Score and rank temporal stacks by their suitability for change detection.

    Scoring criteria:
    - Temporal depth (number of acquisitions): +2 per acquisition
    - Temporal span (days covered): +1 per month
    - Geometry consistency (low incidence angle variance): +10
    - Mode variety (multiple modes available): +5

    Returns:
        DataFrame of stacks sorted by score
    """
    rows = []
    for stack_id, acquisitions in stacks.items():
        n_acquisitions = len(acquisitions)
        dates = [pd.to_datetime(a["datetime"]) for a in acquisitions]
        temporal_span_days = (max(dates) - min(dates)).days

        incidence_angles = [a["incidence_angle"] for a in acquisitions if a["incidence_angle"]]
        incidence_std = np.std(incidence_angles) if len(incidence_angles) > 1 else 999

        modes = set(a["mode"] for a in acquisitions)
        polarizations = set(str(a["polarization"]) for a in acquisitions)

        # Score
        score = (
            n_acquisitions * 2
            + (temporal_span_days / 30) * 1
            + (10 if incidence_std < 3.0 else 0)
            + (5 if len(modes) > 1 else 0)
        )

        lat = acquisitions[0]["latitude"]
        lon = acquisitions[0]["longitude"]

        rows.append(
            {
                "stack_id": stack_id,
                "latitude": lat,
                "longitude": lon,
                "n_acquisitions": n_acquisitions,
                "temporal_span_days": temporal_span_days,
                "date_start": min(dates).date(),
                "date_end": max(dates).date(),
                "incidence_std": round(incidence_std, 2),
                "modes": list(modes),
                "polarizations": list(polarizations),
                "score": round(score, 1),
            }
        )

    df_scores = pd.DataFrame(rows).sort_values("score", ascending=False)
    return df_scores


def build_pair_manifest(
    stacks: dict,
    min_gap_days: int = 7,
    max_gap_days: int = 180,
    max_incidence_diff: float = 5.0,
) -> list:
    """
    Build all valid temporal pairs from stacks for change detection training.

    Args:
        stacks: Dict from identify_temporal_stacks
        min_gap_days: Minimum temporal gap to ensure potential for change
        max_gap_days: Maximum temporal gap to avoid extreme geometry drift
        max_incidence_diff: Maximum incidence angle difference (degrees)

    Returns:
        List of dicts, each describing a valid temporal pair
    """
    pairs = []
    for stack_id, acquisitions in stacks.items():
        acquisitions_sorted = sorted(
            acquisitions, key=lambda x: pd.to_datetime(x["datetime"])
        )

        for i in range(len(acquisitions_sorted)):
            for j in range(i + 1, len(acquisitions_sorted)):
                a_ref = acquisitions_sorted[i]
                a_sec = acquisitions_sorted[j]

                t_ref = pd.to_datetime(a_ref["datetime"])
                t_sec = pd.to_datetime(a_sec["datetime"])
                gap_days = (t_sec - t_ref).days

                if not (min_gap_days <= gap_days <= max_gap_days):
                    continue

                # Check incidence angle compatibility
                if a_ref["incidence_angle"] and a_sec["incidence_angle"]:
                    incidence_diff = abs(
                        a_ref["incidence_angle"] - a_sec["incidence_angle"]
                    )
                    if incidence_diff > max_incidence_diff:
                        continue

                pairs.append(
                    {
                        "pair_id": f"{a_ref['collect_id']}_{a_sec['collect_id']}",
                        "stack_id": stack_id,
                        "collect_id_ref": a_ref["collect_id"],
                        "collect_id_sec": a_sec["collect_id"],
                        "datetime_ref": str(t_ref.date()),
                        "datetime_sec": str(t_sec.date()),
                        "temporal_gap_days": gap_days,
                        "mode": a_ref["mode"],
                        "polarization": a_ref["polarization"],
                        "incidence_ref": a_ref["incidence_angle"],
                        "incidence_sec": a_sec["incidence_angle"],
                        "latitude": a_ref["latitude"],
                        "longitude": a_ref["longitude"],
                        "stac_url_ref": a_ref["stac_url"],
                        "stac_url_sec": a_sec["stac_url"],
                    }
                )

    logger.info(f"Built {len(pairs)} valid temporal pairs")
    return pairs


def print_summary(df_scores: pd.DataFrame, n_top: int = 20) -> None:
    """Print a human-readable summary of the best temporal stacks."""
    print("\n" + "=" * 80)
    print("TOP TEMPORAL STACKS FOR CHANGE DETECTION (DFC 2026)")
    print("=" * 80)
    print(
        df_scores[
            [
                "stack_id",
                "latitude",
                "longitude",
                "n_acquisitions",
                "temporal_span_days",
                "date_start",
                "date_end",
                "incidence_std",
                "score",
            ]
        ]
        .head(n_top)
        .to_string(index=False)
    )
    print("\n")
    print(f"Total stacks found:  {len(df_scores)}")
    print(f"Total pairs possible: {df_scores['n_acquisitions'].apply(lambda n: n*(n-1)//2).sum()}")


def main():
    parser = argparse.ArgumentParser(description="Explore Capella STAC catalog")
    parser.add_argument("--output_dir", type=str, default="./stac_results")
    parser.add_argument("--min_acquisitions", type=int, default=3)
    parser.add_argument("--max_results", type=int, default=1000)
    parser.add_argument("--min_gap_days", type=int, default=7)
    parser.add_argument("--max_gap_days", type=int, default=180)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Query catalog
    try:
        client = get_stac_client()
        df_acquisitions = query_temporal_stacks(client, max_results=args.max_results)
    except Exception:
        logger.warning("STAC query failed. Generating placeholder output for structure demo.")
        df_acquisitions = pd.DataFrame()

    if df_acquisitions.empty:
        logger.warning("No acquisitions retrieved. Check STAC endpoint or network access.")
        logger.info("Next step: Try direct S3 access with download_capella_data.py --use_s3")
        return

    # Save raw acquisitions
    df_acquisitions.to_csv(out_dir / "all_acquisitions.csv", index=False)
    logger.info(f"Saved acquisitions to {out_dir}/all_acquisitions.csv")

    # Identify and score stacks
    stacks = identify_temporal_stacks(df_acquisitions)
    df_scores = score_stacks(stacks)

    # Filter by minimum acquisitions
    df_scores = df_scores[df_scores["n_acquisitions"] >= args.min_acquisitions]
    df_scores.to_csv(out_dir / "stack_scores.csv", index=False)
    logger.info(f"Saved stack scores to {out_dir}/stack_scores.csv")

    # Build pair manifest
    pairs = build_pair_manifest(stacks, args.min_gap_days, args.max_gap_days)
    with open(out_dir / "pair_manifest.json", "w") as f:
        json.dump(pairs, f, indent=2, default=str)
    logger.info(f"Saved {len(pairs)} pairs to {out_dir}/pair_manifest.json")

    # Print summary
    print_summary(df_scores)
    logger.info(f"Results saved to {out_dir}/")
    logger.info("Next step: python select_stacks.py --scores stac_results/stack_scores.csv")


if __name__ == "__main__":
    main()
