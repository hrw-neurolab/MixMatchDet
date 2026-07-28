# ------------------------------------------------------------------------
# MixMatchDet
# Copyright (c) 2026 Hochschule Ruhr West
# Modified from https://github.com/hustvl/YOLOS/blob/main/models/backbone.py
# Licensed under the Apache License, Version 2.0 [See LICENSE for details]
# ------------------------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from src.models.detr.patch_candidate_head import PatchCandidateHead
from src.models.detr.head import DETRHead
from src.models.yolos.attention import Attention, MemEffAttention
from src.models.yolos.layers import HybridEmbed, PatchEmbed, Block
from src.util.batch import BatchInfo, PackedBatch, PaddedBatch


class YOLOS(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        hybrid_backbone=None,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        is_distill=False,
        base_hw=(224, 224),
        mid_hw=None,
        attn_class="Attention",
        num_det_tokens=0,
        aux_indices=[],
        **kwargs,
    ):
        super().__init__()

        self.img_size = img_size

        self.depth = depth
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone,
                img_size=img_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if is_distill:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches + 2, embed_dim)
            )
        else:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches + 1, embed_dim)
            )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        attn = Attention
        if attn_class == "MemEffAttention":
            attn = MemEffAttention

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,  # type: ignore
                    attn_class=attn,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        # self.repr = nn.Linear(embed_dim, representation_size)
        # self.repr_act = nn.Tanh()

        # Classifier head
        # self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        # det settings
        self.head_indices = aux_indices + [depth - 1]
        self.num_det_tokens = num_det_tokens
        self.base_hw = (base_hw[0] // self.patch_size, base_hw[1] // self.patch_size)
        self.mid_hw = None
        self.mid_pos_embed = None

        if mid_hw is not None:
            self.mid_hw = (mid_hw[0] // self.patch_size, mid_hw[1] // self.patch_size)
            self.mid_pos_embed = nn.Parameter(
                torch.zeros(
                    self.depth - 1,
                    1,
                    1 + (self.mid_hw[0] * self.mid_hw[1]) + num_det_tokens,
                    self.embed_dim,
                )
            )
            trunc_normal_(self.mid_pos_embed, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore  # type: ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "det_token"}

    def interpolate_base_pos(self, patch_hw: tuple[int, int]):
        if patch_hw == self.base_hw:
            return self.pos_embed

        cls_pos = self.pos_embed[:, :1, :]
        det_pos = self.pos_embed[:, -self.num_det_tokens :, :]

        patch_pos = self.pos_embed[:, 1 : -self.num_det_tokens, :]
        patch_pos = patch_pos.transpose(1, 2)  # [1, C, num_patch_tokens]
        patch_pos = patch_pos.view(1, self.embed_dim, *self.base_hw)

        patch_pos = nn.functional.interpolate(patch_pos, size=patch_hw, mode="bicubic")

        # [1, new_Ph*new_Pw, C]
        patch_pos = patch_pos.flatten(2).transpose(1, 2)
        return torch.cat((cls_pos, patch_pos, det_pos), dim=1)

    def interpolate_mid_pos(self, patch_hw: tuple[int, int]):
        assert self.mid_hw is not None and self.mid_pos_embed is not None
        if patch_hw == self.mid_hw:
            return self.mid_pos_embed

        cls_pos = self.mid_pos_embed[:, :, :1, :]
        det_pos = self.mid_pos_embed[:, :, -self.num_det_tokens :, :]

        patch_pos = self.mid_pos_embed[:, :, 1 : -self.num_det_tokens, :]
        patch_pos = patch_pos.transpose(2, 3)  # [D, 1, C, num_patch_tokens]
        D, B, C, _ = patch_pos.shape
        patch_pos = patch_pos.view(D * B, C, *self.mid_hw)

        patch_pos = nn.functional.interpolate(patch_pos, size=patch_hw, mode="bicubic")

        # [D, 1, new_Ph*new_Pw, C]
        patch_pos = patch_pos.flatten(2).transpose(1, 2).unsqueeze(1)
        return torch.cat((cls_pos, patch_pos, det_pos), dim=1)

    def prepare_tokens(
        self,
        x: torch.Tensor,
        det_tokens: torch.Tensor | None = None,
        det_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        B = x.shape[0]
        x, patch_hw = self.patch_embed(x)

        tokens: list[torch.Tensor] = [self.cls_token.expand(B, -1, -1)]
        tokens.append(x)

        if det_tokens is not None:
            det_tokens = det_tokens.expand(B, -1, -1)
            tokens.append(det_tokens)

        x = torch.cat(tokens, dim=1)

        # interpolate init pe
        x = x + self.interpolate_base_pos(patch_hw)
        x = self.pos_drop(x)

        return x, patch_hw

    def get_mid_pos(self, batch_info: BatchInfo):
        if self.mid_pos_embed is None:
            return None

        if isinstance(batch_info, PaddedBatch):
            patch_hw = batch_info.feature_hws[0]
            return self.interpolate_mid_pos(patch_hw)

        assert isinstance(batch_info, PackedBatch)
        mid_pos = [self.interpolate_mid_pos(hw) for hw in batch_info.feature_hws]
        return torch.cat(mid_pos, dim=2)

    def forward_det_tokens(
        self,
        x: torch.Tensor,
        batch_info: BatchInfo,
    ):
        if self.training:
            head_indices = self.head_indices
        else:
            head_indices = [self.head_indices[-1]]

        # interpolate mid pe
        temp_mid_pos_embed = self.get_mid_pos(batch_info)
        det_tokens_list: list[torch.Tensor] = []

        for i, blk in enumerate(self.blocks):
            x = blk(x, batch_info)

            if i in head_indices:
                det_tokens = x.index_select(1, batch_info.det_idx)
                det_tokens_list.append(det_tokens)

            if temp_mid_pos_embed is not None and i < (self.depth - 1):
                x = x + temp_mid_pos_embed[i]

        det_tokens = torch.stack(det_tokens_list)
        det_tokens = self.norm(det_tokens)

        x = x.index_select(1, batch_info.patch_idx)
        x = self.norm(x)

        return det_tokens, x

    def forward_patch_candidates(self, *args, **kwargs):
        raise NotImplementedError

    def forward_mixed_candidates(self, *args, **kwargs):
        raise NotImplementedError

    def load_pretrained(self, state_dict: dict[str, torch.Tensor]):
        if "model" in state_dict:
            state_dict = state_dict["model"]  # type: ignore

        pretrained_pos_embed = state_dict.pop("pos_embed")
        self.load_state_dict(state_dict, strict=False)

        pos_cls, pos_patches = pretrained_pos_embed[:, :1], pretrained_pos_embed[:, 1:]

        # reshape the pos_patches to a 2D map based on the original image size and patch size
        PREV_H = PREV_W = self.img_size // self.patch_size
        if (PREV_H, PREV_W) != self.base_hw:
            pos_patches = pos_patches.transpose(1, 2)  # [1, C, Np]
            pos_patches = pos_patches.view(1, self.embed_dim, PREV_H, PREV_W)

            # interpolate the pos_patches to the higher image resolution
            pos_patches = nn.functional.interpolate(
                pos_patches, size=self.base_hw, mode="bicubic"
            )

            # reshape pos_patches to the original shape and set it as a new parameter
            # [1, new_Ph*new_Pw, C]
            pos_patches = pos_patches.flatten(2).transpose(1, 2)

        pos_det = torch.zeros(1, self.num_det_tokens, self.embed_dim)
        trunc_normal_(pos_det, std=0.02)
        self.pos_embed = nn.Parameter(torch.cat((pos_cls, pos_patches, pos_det), dim=1))
