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
        ds_name2variable_tuples,
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
        super().__init__()

        # TODO: remove time_history parameter
        self.img_size = img_size
        self.patch_size = patch_siz
        self.ds_name2variable_tuples = ds_name2variable_tuples
        self.parallel_patch_embed = parallel_patch_embed
        # variable tokenization: separate embedding layer for each input variable
        if self.parallel_patch_embed:
            self.ds_name2token_embeds = {
                ds_name: ParallelVarPatchEmbed(
                    len(variable_tuples), img_size, patch_size, embed_dim
                )
                for ds_name, variable_tuples in ds_name2variable_tuples.items()
            }
            self.ds_name2num_patches = {
                self.ds_name2token_embeds[ds_name].num_patches
                for ds_name, token_embed in self.ds_name2token_embeds.items()
            }

            # self.num_patches = self.token_embeds.num_patches
        else:
            self.token_embeds = nn.ModuleList(
                [
                    PatchEmbed(img_size, patch_size, 1, embed_dim)
                    for i in range(len(default_vars))
                ]
            )
            self.ds_name2token_embeds = {
                ds_name: nn.ModuleList(
                    [
                        PatchEmbed(img_size, patch_size, 1, embed_dim)
                        for i in range(len(variable_tuples))
                    ]
                )
                for ds_name, variable_tuples in ds_name2variable_tuples.items()
            }
            self.ds_name2num_patches = {
                ds_name: token_embeds[0].num_patches
                for ds_name, token_embeds in self.ds_name2token_embeds.items()
            }

        # variable embedding to denote which variable each token belongs to
        # helps in aggregating variables
        (
            self.ds_name2var_embed,
            self.ds_name2var_map,
        ) = self.create_var_embedding(embed_dim)
        self.lead_time = lead_time
        # variable aggregation: a learnable query and a single-layer cross attention
        self.var_query = nn.Parameter(
            torch.zeros(1, 1, embed_dim), requires_grad=True
        )
        self.var_agg = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )

        # positional embedding and lead time embedding
        self.ds_name2pos_embed = {
            ds_name: nn.Parameter(
                torch.zeros(1, num_patches, embed_dim), requires_grad=True
            )
            for ds_name, num_patches in self.ds_name2num_patches.items()
        }
        # self.pos_embed = nn.Parameter(
        #     torch.zeros(1, self.num_patches, embed_dim), requires_grad=True
        # )
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
        for ds_name, variable_tuples in self.ds_name2variable_tuples.items():
            pos_embed = get_2d_sincos_pos_embed(
                self.ds_name2pos_embed[ds_name].shape[-1],
                int(self.img_size[0] / self.patch_size),
                int(self.img_size[1] / self.patch_size),
                cls_token=False,
            )
            self.ds_name2pos_embed[ds_name].data.copy_(
                torch.from_numpy(pos_embed).float().unsqueeze(0)
            )

            var_embed = get_1d_sincos_pos_embed_from_grid(
                self.ds_name2var_embed[ds_name].shape[-1],
                np.arange(len(variable_tuples)),
            )
            self.ds_name2var_embed[ds_name].data.copy_(
                torch.from_numpy(var_embed).float().unsqueeze(0)
            )

            # token embedding layer
            if self.parallel_patch_embed:
                for i in range(
                    len(self.ds_name2token_embeds[ds_name].proj_weights)
                ):
                    w = self.ds_name2token_embeds[ds_name].proj_weights[i].data
                    trunc_normal_(w.view([w.shape[0], -1]), std=0.02)
            else:
                for i in range(len(self.ds_name2token_embeds[ds_name])):
                    w = self.ds_name2token_embeds[ds_name][i].proj.weight.data
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
        ds_name2var_embed = {
            ds_name: nn.Parameter(
                torch.zeros(1, len(var_tuples), dim), requires_grad=True
            )
            for ds_name, var_tuples in self.ds_name2variable_tuples.items()
        }
        # TODO: create a mapping from var --> idx
        ds_name2var_map = {}
        var_map = {}
        idx = 0
        for ds_name, var_tuples in self.ds_name2variable_tuples.items():
            for var in var_tuples:
                var_map[var] = idx
                idx += 1
            ds_name2var_map[ds_name] = var_map
        return ds_name2var_embed, ds_name2var_map

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device, ds_name):
        ids = np.array([self.ds_name2var_map[ds_name][var] for var in vars])
        return torch.from_numpy(ids).to(device)

    def get_var_emb(self, var_emb, vars, ds_name):
        ids = self.get_var_ids(vars, var_emb.device, ds_name)
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

    def forward_encoder(
        self,
        x: torch.Tensor,
        lead_times: torch.Tensor,
        variables,
        ds_name,
    ):
        # x: `[B, V, H, W]` shape.

        if isinstance(variables, list):
            variables = tuple(variables)

        # tokenize each variable separately
        embeds = []
        var_ids = self.get_var_ids(variables, x.device, ds_name)

        if self.parallel_patch_embed:
            x = self.ds_name2token_embeds[ds_name](x, var_ids)  # B, V, L, D
        else:
            for i in range(len(var_ids)):
                id = var_ids[i]
                embeds.append(
                    self.ds_name2token_embeds[ds_name][id](x[:, i : i + 1])
                )
            x = torch.stack(embeds, dim=1)  # B, V, L, D

        # add variable embedding
        var_embed = self.get_var_emb(self.var_embed, variables, ds_name)
        x = x + var_embed.unsqueeze(2)  # B, V, L, D

        # variable aggregation
        x = self.aggregate_variables(x)  # B, L, D

        # add pos embedding
        x = x + self.ds_name2pos_embed[ds_name]

        # add lead time embedding using self.lead_time
        # breakpoint()
        lead_times = lead_times.to(x.device)
        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))  # B, D
        lead_time_emb = lead_time_emb.unsqueeze(1)
        x = x + lead_time_emb  # B, L, D

        x = self.pos_drop(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def construct_lead_time_tensor(self, x):
        lead_times = torch.FloatTensor(
            [self.lead_time for _ in range(x.shape[0])]
        ).to(x.device)
        return lead_times

    def forward(self, x, y, dataset_names):
        """Forward pass through the model.

        Args:
            x: `[B, Vi, H, W]` shape. Input weather/climate variables
            y: `[B, Vo, H, W]` shape. Target weather/climate variables
            dataset_names: `[B]` shape. Forecasting lead times of each element of the batch.

        Returns:
            preds (torch.Tensor): `[B, Vo, H, W]` shape. Predicted weather/climate variables.
        """
        dataset_name = dataset_names[0]
        lead_times = self.construct_lead_time_tensor(x)
        out_transformers = self.forward_encoder(
            x, lead_times, dataset_name
        )  # B, L, D
        preds = self.head(out_transformers)  # B, L, V*p*p

        preds = self.unpatchify(preds)
        variable_tuples = self.ds_name2variable_tuples[dataset_name]
        out_var_ids = self.get_var_ids(tuple(variable_tuples), preds.device)
        preds = preds[:, out_var_ids]
        return preds
