"""
Entry point for running a baseline InSAR DEM experiment.

Usage
-----
python experiments/baseline/run_baseline.py --config configs/experiment/baseline_sentinel1.yaml
"""

import argparse
from pathlib import Path

import yaml

from src.insar_processing.baseline import BaselineConfig, run_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline InSAR DEM experiment.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to baseline experiment YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)

    baseline_cfg = BaselineConfig(
        interferogram_path=cfg["interferogram_path"],
        coherence_path=cfg.get("coherence_path"),
        unwrapped_phase_path=cfg["unwrapped_phase_path"],
        output_dem_path=cfg["output_dem_path"],
        wavelength_m=cfg["wavelength_m"],
        incidence_angle_deg=cfg["incidence_angle_deg"],
        perpendicular_baseline_m=cfg["perpendicular_baseline_m"],
    )

    out_dem = run_baseline(baseline_cfg)
    print(f"Baseline DEM written to: {out_dem}")


if __name__ == "__main__":
    main()

