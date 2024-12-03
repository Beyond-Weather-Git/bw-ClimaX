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

from .parallelpatchembed import ParallelVarPatchEmbed


class ClimaX(nn.Module):
    """Implements the ClimaX model as described in the paper,
    https://arxiv.org/abs/2301.10343

    Args:
        default_vars (list): list of default variables to be used for training
        img_size (list): image size of the input data
        patch_size (int): patch size of the input data
        embed_dim (int): embedding dimension
        depth (int): number of transformer layers
        decoder_depth (int): number of decoder layers
        num_heads (int): number of attention heads
        mlp_ratio (float): ratio of mlp hidden dimension to embedding dimension
        drop_path (float): stochastic depth rate
        drop_rate (float): dropout rate
        parallel_patch_embed (bool): whether to use parallel patch embedding
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
        super().__init__()

        # TODO: remove time_history parameter
        self.img_size = img_size
        self.patch_size = patch_size
        self.default_vars = default_vars
        self.parallel_patch_embed = parallel_patch_embed
        self.target_config = target_config
        # variable tokenization: separate embedding layer for each input variable
        if self.parallel_patch_embed:
            self.token_embeds = ParallelVarPatchEmbed(
                len(default_vars), img_size, patch_size, embed_dim
            )
            self.num_patches = self.token_embeds.num_patches
        else:
            self.token_embeds = nn.ModuleList(
                [
                    PatchEmbed(img_size, patch_size, 1, embed_dim)
                    for i in range(len(default_vars))
                ]
            )
            self.num_patches = self.token_embeds[0].num_patches

        # variable embedding to denote which variable each token belongs to
        # helps in aggregating variables
        self.var_embed, self.var_map = self.create_var_embedding(embed_dim)
        self.lead_time = lead_time
        # variable aggregation: a learnable query and a single-layer cross attention
        self.var_query = nn.Parameter(
            torch.zeros(1, 1, embed_dim), requires_grad=True
        )
        self.var_agg = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )

        # positional embedding and lead time embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=True
        )
        self.lead_time_embed = nn.Linear(1, embed_dim)
        self.lead_time = lead_time
        # --------------------------------------------------------------------------

        # ViT backbone
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                    drop=drop_rate,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # --------------------------------------------------------------------------

        # prediction head
        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(embed_dim, embed_dim))
            self.head.append(nn.GELU())
        self.head.append(
            nn.Linear(embed_dim, len(self.default_vars) * patch_size**2)
        )
        self.head = nn.Sequential(*self.head)

        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialize pos_emb and var_emb
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.img_size[0] / self.patch_size),
            int(self.img_size[1] / self.patch_size),
            cls_token=False,
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )

        var_embed = get_1d_sincos_pos_embed_from_grid(
            self.var_embed.shape[-1], np.arange(len(self.default_vars))
        )
        self.var_embed.data.copy_(
            torch.from_numpy(var_embed).float().unsqueeze(0)
        )

        # token embedding layer
        if self.parallel_patch_embed:
            for i in range(len(self.token_embeds.proj_weights)):
                w = self.token_embeds.proj_weights[i].data
                trunc_normal_(w.view([w.shape[0], -1]), std=0.02)
        else:
            for i in range(len(self.token_embeds)):
                w = self.token_embeds[i].proj.weight.data
                trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def create_var_embedding(self, dim):
        var_embed = nn.Parameter(
            torch.zeros(1, len(self.default_vars), dim), requires_grad=True
        )
        # TODO: create a mapping from var --> idx
        var_map = {}
        idx = 0
        for var in self.default_vars:
            var_map[var] = idx
            idx += 1
        return var_embed, var_map

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device):
        ids = np.array([self.var_map[var] for var in vars])
        return torch.from_numpy(ids).to(device)

    def get_var_emb(self, var_emb, vars):
        ids = self.get_var_ids(vars, var_emb.device)
        return var_emb[:, ids, :]

    def unpatchify(self, x: torch.Tensor, h=None, w=None):
        """
        x: (B, L, V * patch_size**2)
        return imgs: (B, V, H, W)
        """
        p = self.patch_size
        c = len(self.default_vars)
        h = self.img_size[0] // p if h is None else h // p
        w = self.img_size[1] // p if w is None else w // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def aggregate_variables(self, x: torch.Tensor):
        """
        x: B, V, L, D
        """
        b, _, l, _ = x.shape
        x = torch.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)  # BxL, V, D

        var_query = self.var_query.repeat_interleave(x.shape[0], dim=0)
        x, _ = self.var_agg(var_query, x, x)  # BxL, D
        x = x.squeeze()

        x = x.unflatten(dim=0, sizes=(b, l))  # B, L, D
        return x

    def forward_encoder(self, x: torch.Tensor, variables):
        lead_times = self.construct_lead_time_tensor(x)
        if isinstance(variables, list):
            variables = tuple(variables)

        # tokenize each variable separately
        embeds = []
        var_ids = self.get_var_ids(variables, x.device)

        if self.parallel_patch_embed:
            x = self.token_embeds(x, var_ids)  # B, V, L, D
        else:
            for i in range(len(var_ids)):
                id = var_ids[i]
                embeds.append(self.token_embeds[id](x[:, i : i + 1]))
            x = torch.stack(embeds, dim=1)  # B, V, L, D

        # add variable embedding
        var_embed = self.get_var_emb(self.var_embed, variables)
        x = x + var_embed.unsqueeze(2)  # B, V, L, D

        # variable aggregation
        x = self.aggregate_variables(x)  # B, L, D

        # add pos embedding
        x = x + self.pos_embed

        # add lead time embedding
        lead_times = lead_times.to(x.device)
        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))  # B, D
        lead_time_emb = lead_time_emb.unsqueeze(1)
        x = x + lead_time_emb  # B, L, D

        x = self.pos_drop(x)

        return x

    def forward_processor(self, x, variables):
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # apply all head layers except the last one
        for layer in self.head[:-1]:
            x = layer(x)

        return x

    def forward_head(self, x, variables):
        # apply only the last layer of the head
        x = self.head[-1](x)  # B, L, V*p*p
        x = self.unpatchify(x)
        out_var_ids = self.get_var_ids(tuple(variables), x.device)
        x = x[:, out_var_ids]
        return x

    def forward(self, x, y, variables):
        # Encoder
        x = self.forward_encoder(x, variables)

        # Processor
        x = self.forward_processor(x, variables)

        # Decoder
        preds = self.forward_head(x, variables)

        return preds

    def construct_lead_time_tensor(self, x):
        lead_times = torch.FloatTensor(
            [self.lead_time for _ in range(x.shape[0])]
        ).to(x.device)
        return lead_times
