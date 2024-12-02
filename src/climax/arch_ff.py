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


class ClimaXFF(ClimaX):
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
        # self.head = nn.ModuleList()
        reduced_dim = 4
        self.dim_reduce = nn.Linear(embed_dim, reduced_dim)
        unrolled_dim = self.num_patches * reduced_dim

        head = nn.ModuleList()
        for _ in range(decoder_depth):
            head.append(nn.Linear(embed_dim, embed_dim))
            head.append(nn.GELU())
        head.append(nn.Linear(embed_dim, patch_size**2))
        self.spatial_head = nn.Sequential(*self.head)
        self.apply(self._init_weights)
        self.lead_time = lead_time

    def forward(self, x, y, variables):
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

        lead_times = self.construct_lead_time_tensor(x)
        out_transformers = self.forward_encoder(
            x, lead_times, variables
        )  # B, L, D

        # Pool over sequence length
        # x = out_transformers.mean(dim=1)  # B, D
        reduced = self.dim_reduce(out_transformers)  # B, L, reduced_dim

        # Unroll
        unrolled = reduced.reshape(reduced.shape[0], -1)  # B, L*reduced_dim
        # Pass through the head
        preds = self.spatial_head(out_transformers)  # B, L, V*p*p

        preds = self.unpatchify(preds)

        return preds

    def unpatchify(self, x: torch.Tensor):
        """
        Convert patchified tensor back to spatial tensor for single-field output.

        Args:
            x: Tensor of shape `[B, L, patch_size**2]`.

        Returns:
            imgs: Tensor of shape `[B, 1, H, W]`.
        """
        p = self.patch_size  # Patch size
        B, L, _ = x.shape
        H = self.img_size[0] // p
        W = self.img_size[1] // p

        # Ensure L matches H * W
        assert H * W == L, f"Expected L={H * W}, but got L={L}"

        # Reshape x into `[B, H, W, p, p]`
        x = x.reshape(B, H, W, p, p)

        # Rearrange into `[B, 1, H*p, W*p]`
        imgs = x.permute(0, 3, 4, 1, 2).reshape(B, 1, H * p, W * p)
        return imgs
