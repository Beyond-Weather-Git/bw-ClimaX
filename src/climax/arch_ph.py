# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed, trunc_normal_

from climax.utils.pos_embed import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
)

from .arch import ClimaX


import torch
import torch.nn as nn
from climax.arch import ClimaX


class ClimaXPH(ClimaX):
    """
    ClimaXPH is a variant of ClimaX designed for point-wise prediction tasks.
    It inherits from ClimaX and modifies the prediction head and forward method
    to output a single value instead of spatial maps.
    """

    def __init__(
        self,
        default_vars,
        img_size=[32, 64],
        patch_size=2,
        embed_dim=1024,
        depth=8,
        decoder_depth=2,
        num_heads=16,
        mlp_ratio=4.0,
        drop_path=0.1,
        drop_rate=0.1,
        parallel_patch_embed=False,
        lead_time=None,
    ):
        super().__init__(
            default_vars=default_vars,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            drop_rate=drop_rate,
            parallel_patch_embed=parallel_patch_embed,
        )

        # Redefine the prediction head to output a single value
        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(embed_dim, embed_dim))
            self.head.append(nn.GELU())
        self.head.append(nn.Linear(embed_dim, 1))  # Output dimension is 1
        self.head = nn.Sequential(*self.head)
        self.apply(self._init_weights)

    def forward(self, x, y, lead_times, variables, metric, lat):
        """
        Forward pass for ClimaXPH.

        Args:
            x (torch.Tensor): Input tensor of shape `[B, V, H, W]`.
            y (torch.Tensor): Target tensor of shape `[B]`.
            lead_times (torch.Tensor): Lead times tensor of shape `[B]`.
            variables (list): List of variable names.
            metric (list): List of metric functions.
            lat (torch.Tensor): Latitude tensor.

        Returns:
            loss (list): Computed loss values.
            preds (torch.Tensor): Predictions of shape `[B]`.
        """

        lead_times = lead_times.to(x.device)
        out_transformers = self.forward_encoder(
            x, lead_times, variables
        )  # B, L, D

        # Pool over sequence length
        x = out_transformers.mean(dim=1)  # B, D

        # Pass through the prediction head
        preds = self.head(x).squeeze(-1)  # B

        if metric is None:
            loss = None
        else:

            loss = [m(preds, y, variables, lat) for m in metric]

        return loss, preds
