"""
Train a baseline U-Net-style model for InSAR DEM enhancement.

This is a lightweight training script intended as a starting point.
It expects YAML configs for data, model, and training settings.

Usage
-----
python experiments/enhanced/train_unet.py \\
    --data_config configs/data/sentinel1_example.yaml \\
    --model_config configs/model/unet_baseline.yaml \\
    --train_config configs/train/default.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from src.insar_processing.dataset_preparation import TileConfig, prepare_dem_tiles
from src.models.unet_baseline import UNetBaseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-Net baseline for InSAR DEM enhancement.")
    parser.add_argument("--data_config", type=str, required=True, help="Path to data YAML config.")
    parser.add_argument("--model_config", type=str, required=True, help="Path to model YAML config.")
    parser.add_argument("--train_config", type=str, required=True, help="Path to train YAML config.")
    return parser.parse_args()


def load_yaml(path: str):
    with Path(path).open("r") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()

    data_cfg = load_yaml(args.data_config)
    model_cfg = load_yaml(args.model_config)
    train_cfg = load_yaml(args.train_config)

    tile_cfg = TileConfig(
        tile_size=data_cfg.get("tile_size", 256),
        stride=data_cfg.get("stride", 256),
    )

    tiles = prepare_dem_tiles(
        interferogram_path=data_cfg["interferogram_path"],
        coherence_path=data_cfg["coherence_path"],
        reference_dem_path=data_cfg["reference_dem_path"],
        tile_config=tile_cfg,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetBaseline(
        in_channels=model_cfg.get("in_channels", 2),
        out_channels=model_cfg.get("out_channels", 1),
        features=model_cfg.get("features", [32, 64, 128, 256]),
    ).to(device)

    lr = train_cfg.get("learning_rate", 1e-4)
    num_epochs = train_cfg.get("num_epochs", 1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.L1Loss()

    print(f"Loaded {len(tiles)} tiles. Starting training for {num_epochs} epochs.")

    # NOTE: This is a minimal, synchronous loop without DataLoader or batching.
    # It is intended as a conceptual scaffold.
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for tile in tiles:
            # Stack interferogram and coherence as channels.
            x = torch.from_numpy(
                np.stack([tile["interferogram"], tile["coherence"]], axis=0)
            ).unsqueeze(0)  # (1, C, H, W)
            y = torch.from_numpy(tile["dem"]).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

            x = x.float().to(device)
            y = y.float().to(device)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())

        avg_loss = epoch_loss / max(len(tiles), 1)
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.6f}")

    out_dir = Path(train_cfg.get("output_dir", "experiments/enhanced/checkpoints"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "unet_baseline_final.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    main()

