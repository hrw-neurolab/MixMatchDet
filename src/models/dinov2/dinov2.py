# ------------------------------------------------------------------------
# MixMatchDet
# Copyright (c) 2026 Hochschule Ruhr West
# This file has been modified by Hochschule Ruhr West.
# Licensed under the Apache License, Version 2.0 [See LICENSE for details]
# ------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from functools import partial
import math

import torch
import torch.nn as nn

from src.models.detr.patch_candidate_head import PatchCandidateHead
from src.models.detr.head import DETRHead
from src.util.batch import BatchInfo

from .attention import MemEffSelfAttention, SelfAttention
from .mlp import Mlp
from .patch_embed import PatchEmbed
from .block import Block


class DINOv2(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        attn_class="SelfAttention",
        *,
        aux_indices: list[int] = [],
        img_side_minmax=(224, 224),
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
            attn_class: (str) attention class to use ("SelfAttention" | "MemEffSelfAttention")
            aux_indices: (list) list of block indices to extract auxiliary outputs from
            img_side_minmax: (tuple) min and max side length for geometric-mean sizing
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.img_size = img_size
        self.patch_size = patch_size
        self.interpolate_antialias = interpolate_antialias
        self.embed_dim = embed_dim

        # geometric-mean side length
        S = math.sqrt(img_side_minmax[0] * img_side_minmax[1])
        self.base_hw = (math.ceil(S / self.patch_size), math.ceil(S / self.patch_size))

        self.head_indices = aux_indices + [depth - 1]

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.zeros(1, num_register_tokens, embed_dim)
            )
        else:
            self.register_tokens = None

        block_classes = [Block] * depth

        if attn_class == "SelfAttention":
            attn_classes = [SelfAttention] * depth
        elif attn_class == "MemEffSelfAttention":
            attn_classes = [MemEffSelfAttention] * depth
        else:
            raise ValueError(f"Unknown attn_class: {attn_class}")

        ffn_layers = [Mlp] * depth

        blocks_list = [
            block_classes[i](
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                norm_layer=norm_layer,
                act_layer=nn.GELU,
                init_values=init_values,
                attn_class=attn_classes[i],
                ffn_layer=ffn_layers[i],
            )
            for i in range(depth)
        ]

        self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer(embed_dim)

    def interpolate_pos_encoding(self, x: torch.Tensor, patch_hw: tuple[int, int]):
        if patch_hw == self.base_hw:
            return self.pos_embed

        previous_dtype = x.dtype

        patch_pos = self.pos_embed.float()
        patch_pos = patch_pos.transpose(1, 2)  # [1, C, num_patch_tokens]
        patch_pos = patch_pos.view(1, self.embed_dim, *self.base_hw)

        patch_pos = nn.functional.interpolate(
            patch_pos,
            size=patch_hw,
            mode="bicubic",
            antialias=self.interpolate_antialias,
        )

        # [1, new_Ph*new_Pw, C]
        patch_pos = patch_pos.flatten(2).transpose(1, 2)
        return patch_pos.to(previous_dtype)

    def patchify(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        x, feature_hw = self.patch_embed(images)
        x += self.interpolate_pos_encoding(x, feature_hw).expand(images.size(0), -1, -1)
        return x, feature_hw

    def prepare_tokens(
        self,
        x: torch.Tensor,
        det_tokens: torch.Tensor | None = None,
        det_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        B = x.shape[0]
        patch_tokens, feature_hw = self.patchify(x)

        tokens: list[torch.Tensor] = [self.cls_token.expand(B, -1, -1)]

        if self.register_tokens is not None:
            tokens.append(self.register_tokens.expand(B, -1, -1))

        tokens.append(patch_tokens)

        if det_tokens is not None:
            if det_pos is not None:
                det_tokens = det_tokens + det_pos

            det_tokens = det_tokens.expand(B, -1, -1)
            tokens.append(det_tokens)

        return torch.cat(tokens, dim=1), feature_hw

    def forward_det_tokens(
        self,
        x: torch.Tensor,
        batch_info: BatchInfo,
    ):
        if self.training:
            head_indices = self.head_indices
        else:
            head_indices = [self.head_indices[-1]]

        det_tokens_list: list[torch.Tensor] = []

        for i, blk in enumerate(self.blocks):
            x = blk(x, batch_info)

            if i in head_indices:
                det_tokens = x.index_select(1, batch_info.det_idx)
                det_tokens_list.append(det_tokens)

        det_tokens = torch.stack(det_tokens_list)
        det_tokens = self.norm(det_tokens)

        x = x.index_select(1, batch_info.patch_idx)
        x = self.norm(x)

        return det_tokens, x

    def forward_patch_candidates(
        self,
        x: torch.Tensor,
        batch_info: BatchInfo,
        patch_candidate_head: PatchCandidateHead,
    ):
        enc_outputs: list[dict] = []
        outputs: list[dict] = []

        if self.training:
            head_indices = self.head_indices
        else:
            head_indices = [self.head_indices[-1]]

        for i, blk in enumerate(self.blocks):
            x = blk(x, batch_info)

            if i not in head_indices:
                continue

            patch_tokens = x.index_select(1, batch_info.patch_idx)
            patch_tokens = self.norm(patch_tokens)
            patch_candidate_boxes, patch_candidates, out, enc_out = (
                patch_candidate_head(patch_tokens, batch_info)
            )

            enc_outputs.append(enc_out)
            outputs.append(out)

        if len(outputs) > 1:
            out["aux_outputs"] = outputs[:-1]
            out["enc_outputs"] = enc_outputs

        return patch_candidates, patch_candidate_boxes, patch_tokens, out

    def forward_mixed_candidates(
        self,
        x: torch.Tensor,
        batch_info: BatchInfo,
        patch_candidate_head: PatchCandidateHead,
        detr_head: DETRHead,
    ):
        enc_outputs: list[dict] = []
        outputs: list[dict] = []

        if self.training:
            head_indices = self.head_indices
        else:
            head_indices = [self.head_indices[-1]]

        for i, blk in enumerate(self.blocks):
            x = blk(x, batch_info)

            if i not in head_indices:
                continue

            patch_tokens = x.index_select(1, batch_info.patch_idx)
            patch_tokens = self.norm(patch_tokens)

            det_tokens = x.index_select(1, batch_info.det_idx)
            det_tokens = det_tokens.view(batch_info.B, batch_info.num_det_tokens, -1)
            det_tokens = self.norm(det_tokens)

            patch_candidate_boxes, patch_candidates, out_patch, enc_out = (
                patch_candidate_head(patch_tokens, batch_info)
            )
            out_det, det_candidate_boxes = detr_head(det_tokens[None])

            out = {
                "pred_logits": torch.cat(
                    (out_patch["pred_logits"], out_det["pred_logits"]), dim=1
                ),
                "pred_boxes": torch.cat(
                    (out_patch["pred_boxes"], out_det["pred_boxes"]), dim=1
                ),
            }

            if self.training:
                enc_outputs.append(enc_out)
                outputs.append(out)

        if len(outputs) > 1:
            out["aux_outputs"] = outputs[:-1]  # type: ignore
            out["enc_outputs"] = enc_outputs  # type: ignore

        candidates = torch.cat((patch_candidates, det_tokens), dim=1)
        candidate_boxes = torch.cat((patch_candidate_boxes, det_candidate_boxes), dim=1)

        return candidates, candidate_boxes, patch_tokens, out

    def load_pretrained(self, state_dict: dict[str, torch.Tensor]):
        pretrained_pos_embed = state_dict.pop("pos_embed")
        self.load_state_dict(state_dict, strict=False)

        # extract the pretrained positional embeddings for the cls token and the patches
        pos_cls, pos_patches = pretrained_pos_embed[:, :1], pretrained_pos_embed[:, 1:]
        self.cls_token.data += pos_cls.data

        # reshape the pos_patches to a 2D map based on the original image size and patch size
        PREV_H = PREV_W = self.img_size // self.patch_size
        if (PREV_H, PREV_W) != self.base_hw:
            pos_patches = pos_patches.transpose(1, 2)  # [1, C, Np]
            pos_patches = pos_patches.view(1, self.embed_dim, PREV_H, PREV_W)

            # interpolate the pos_patches to the higher image resolution
            pos_patches = nn.functional.interpolate(
                pos_patches,
                size=self.base_hw,
                mode="bicubic",
                antialias=self.interpolate_antialias,
            )

            # reshape pos_patches to the original shape and set it as a new parameter
            # [1, new_Ph*new_Pw, C]
            pos_patches = pos_patches.flatten(2).transpose(1, 2)

        self.pos_embed = nn.Parameter(pos_patches)
