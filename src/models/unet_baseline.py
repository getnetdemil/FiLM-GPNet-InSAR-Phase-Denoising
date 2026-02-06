"""
Baseline U-Net-style model for InSAR DEM enhancement.

This is a compact implementation intended as a reasonable starting point
for experimenting with learning-assisted DEM refinement.
"""

from typing import List

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetBaseline(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        features: List[int] = None,
    ):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down-sampling path
        ch = in_channels
        for feat in features:
            self.downs.append(DoubleConv(ch, feat))
            ch = feat

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Up-sampling path
        rev_feats = list(reversed(features))
        up_ch = features[-1] * 2
        for feat in rev_feats:
            self.ups.append(
                nn.ConvTranspose2d(up_ch, feat, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feat * 2, feat))
            up_ch = feat

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape[-2:] != skip_connection.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip_connection.shape[-2:], mode="bilinear", align_corners=False)

            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)

