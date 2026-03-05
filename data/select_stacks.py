"""
Select Best Temporal Stacks for DFC 2026 Change Detection
==========================================================
Analyzes scored temporal stacks and selects the most suitable ones
for diverse change detection case studies.

Usage:
    python select_stacks.py --scores stac_results/stack_scores.csv \
        --n_urban 2 --n_agricultural 1 --n_coastal 1 --n_industrial 1
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Approximate geographic regions for scene classification
# (lat_min, lat_max, lon_min, lon_max, scene_type)
KNOWN_REGIONS = [
    # Urban centers
    (40.5, 41.0, -74.5, -73.5, "urban_nyc"),
    (51.3, 51.7, -0.5, 0.3, "urban_london"),
    (48.7, 49.1, 2.0, 2.6, "urban_paris"),
    (37.5, 38.0, 126.8, 127.4, "urban_seoul"),
    (35.4, 35.9, 139.4, 140.0, "urban_tokyo"),
    # Agricultural
    (35.0, 42.0, -100.0, -85.0, "agricultural_midwestUS"),
    (45.0, 55.0, 20.0, 40.0, "agricultural_europe"),
    # Coastal/port
    (1.1, 1.5, 103.5, 104.1, "port_singapore"),
    (22.2, 22.6, 113.8, 114.4, "port_hongkong"),
    # Arid/desert (good SAR contrast)
    (24.0, 27.0, 45.0, 55.0, "desert_arabiapen"),
]


def classify_scene_type(lat: float, lon: float) -> str:
    """Classify a location's likely scene type based on coordinates."""
    if lat is None or lon is None:
        return "unknown"

    for lat_min, lat_max, lon_min, lon_max, scene_type in KNOWN_REGIONS:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return scene_type

    # Fallback classification by latitude
    if abs(lat) < 15:
        return "tropical"
    elif abs(lat) < 40:
        return "subtropical"
    elif abs(lat) < 60:
        return "temperate"
    else:
        return "polar"


def select_diverse_stacks(
    df_scores: pd.DataFrame,
    n_total: int = 10,
    min_acquisitions: int = 3,
    min_temporal_span_days: int = 30,
) -> pd.DataFrame:
    """
    Select a diverse set of temporal stacks for maximal scene coverage.

    Selection strategy:
    1. Filter by minimum quality thresholds
    2. Classify by approximate scene type
    3. Select top-scoring stacks while ensuring geographic diversity

    Args:
        df_scores: Scored stacks from explore_stac.py
        n_total: Total number of stacks to select
        min_acquisitions: Minimum number of acquisitions required
        min_temporal_span_days: Minimum temporal coverage required

    Returns:
        DataFrame with selected stacks
    """
    # Add scene type classification
    df = df_scores.copy()
    df["scene_type"] = df.apply(
        lambda r: classify_scene_type(r["latitude"], r["longitude"]), axis=1
    )

    # Apply quality filters
    df = df[df["n_acquisitions"] >= min_acquisitions]
    df = df[df["temporal_span_days"] >= min_temporal_span_days]
    df = df[df["incidence_std"] < 10.0]  # Reasonable geometry consistency

    logger.info(f"Filtered to {len(df)} stacks meeting quality criteria")

    if df.empty:
        logger.warning("No stacks meet quality criteria. Returning top unfiltered stacks.")
        return df_scores.head(n_total)

    # Greedy diverse selection: pick top-scoring stack per scene type
    selected = []
    scene_types_covered = set()

    # First pass: pick best stack from each scene type
    for scene_type, group in df.groupby("scene_type"):
        best = group.nlargest(1, "score").iloc[0]
        selected.append(best)
        scene_types_covered.add(scene_type)

    # Second pass: fill remaining slots with highest-scoring stacks
    remaining_slots = n_total - len(selected)
    if remaining_slots > 0:
        selected_ids = {s["stack_id"] for s in selected}
        remaining = df[~df["stack_id"].isin(selected_ids)].nlargest(remaining_slots, "score")
        selected.extend(remaining.to_dict("records"))

    result = pd.DataFrame(selected).head(n_total)
    result = result.sort_values("score", ascending=False)

    logger.info(f"Selected {len(result)} stacks covering {len(scene_types_covered)} scene types")
    return result


def print_selection_report(selected: pd.DataFrame) -> None:
    """Print a formatted selection report."""
    print("\n" + "=" * 70)
    print("SELECTED TEMPORAL STACKS FOR DFC 2026")
    print("=" * 70)

    for i, (_, row) in enumerate(selected.iterrows(), 1):
        print(f"\n[Stack {i}] {row.get('scene_type', 'unknown')}")
        print(f"  Stack ID:   {row['stack_id']}")
        print(f"  Location:   {row['latitude']:.2f}°N, {row['longitude']:.2f}°E")
        print(f"  Acquisitions: {row['n_acquisitions']}")
        print(f"  Date range: {row['date_start']} → {row['date_end']}")
        print(f"  Span:       {row['temporal_span_days']} days")
        print(f"  Geometry:   incidence_std={row['incidence_std']}°")
        print(f"  Score:      {row['score']}")

    print(f"\nTotal pairs available: {selected['n_acquisitions'].apply(lambda n: n*(n-1)//2).sum()}")


def main():
    parser = argparse.ArgumentParser(description="Select best temporal stacks for DFC 2026")
    parser.add_argument(
        "--scores",
        type=str,
        default="stac_results/stack_scores.csv",
        help="Path to stack_scores.csv from explore_stac.py",
    )
    parser.add_argument("--n_total", type=int, default=10, help="Total stacks to select")
    parser.add_argument("--min_acquisitions", type=int, default=3)
    parser.add_argument("--min_span_days", type=int, default=30)
    parser.add_argument("--output", type=str, default="stac_results/selected_stacks.json")
    args = parser.parse_args()

    scores_path = Path(args.scores)
    if not scores_path.exists():
        logger.error(f"Scores file not found: {scores_path}")
        logger.info("Run explore_stac.py first to generate stack scores.")
        return

    df_scores = pd.read_csv(scores_path)
    logger.info(f"Loaded {len(df_scores)} stacks from {scores_path}")

    selected = select_diverse_stacks(
        df_scores,
        n_total=args.n_total,
        min_acquisitions=args.min_acquisitions,
        min_temporal_span_days=args.min_span_days,
    )

    print_selection_report(selected)

    # Save selected stacks
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_json(out_path, orient="records", indent=2)
    logger.info(f"Saved selected stacks to {out_path}")

    # Print download command
    print("\nNext step — Download data:")
    print(f"  python download_capella_data.py \\")
    print(f"      --manifest stac_results/pair_manifest.json \\")
    print(f"      --asset_type GEO \\")
    print(f"      --output_dir ../data/raw")


if __name__ == "__main__":
    main()
