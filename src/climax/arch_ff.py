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
        target_config=None,
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
            target_config=target_config,
        )

        # Redefine the prediction head to output a single value
        # self.head = nn.ModuleList()
        self.init_ff_head(target_config, embed_dim, decoder_depth, patch_size)
        self.apply(self._init_weights)
        self.lead_time = lead_time

    def init_ff_head(
        self, target_config, embed_dim, decoder_depth, patch_size
    ):
        # reduced_dim = 4
        # self.dim_reduce = nn.Linear(embed_dim, reduced_dim)

        head = nn.ModuleList()
        for _ in range(decoder_depth):
            head.append(nn.Linear(embed_dim, embed_dim))
            head.append(nn.GELU())
        head.append(
            nn.Linear(embed_dim, len(target_config.variables) * patch_size**2)
        )
        self.spatial_head = nn.Sequential(*head)

    def forward_head(self, x, variables):
        x = self.spatial_head(x)
        x = self.unpatchify(x)
        return x

    def unpatchify(self, x: torch.Tensor, h=None, w=None):
        """
        x: (B, L, V * patch_size**2)
        return imgs: (B, V, H, W)
        """
        p = self.patch_size
        c = len(self.target_config.variables)
        h = self.img_size[0] // p if h is None else h // p
        w = self.img_size[1] // p if w is None else w // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs
