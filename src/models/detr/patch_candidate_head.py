# ------------------------------------------------------------------------
# MixMatchDet
# Copyright (c) 2026 Hochschule Ruhr West
# This file has been modified by Hochschule Ruhr West.
# Licensed under the Apache License, Version 2.0 [See LICENSE for details]
# ------------------------------------------------------------------------
# Plain-DETR
# Copyright (c) 2023 Xi'an Jiaotong University & Microsoft Research Asia.
# Licensed under The MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.detr.head import DETRHead
from src.util.batch import BatchInfo, PackedBatch, PaddedBatch


class LayerNorm2D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: N C H W"""
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class PatchCandidateHead(DETRHead):
    def __init__(
        self,
        topk: int,
        in_stride: int,
        target_strides: list[int],
        hidden_dim: int,
        num_classes: int,
    ):
        super().__init__(hidden_dim, num_classes)

        self.topk = topk
        self.in_stride = in_stride
        self.target_strides = target_strides
        self.num_feature_levels = len(target_strides)

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.score_embed = nn.Linear(hidden_dim, 1)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.score_embed.bias.data = torch.tensor([bias_value])

        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)  # type: ignore
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)  # type: ignore

        if self.num_feature_levels == 1 and target_strides[0] == in_stride:
            # In this case, no feature expansion is needed
            self.feature_projs = None
            return

        self.feature_projs = nn.ModuleList()
        for stride in target_strides:
            if stride == in_stride:
                self.feature_projs.append(nn.Identity())
                continue

            layers = []

            if stride > in_stride:
                scale = int(math.log2(stride / in_stride))
                conv_layer = nn.Conv2d
            else:
                scale = int(math.log2(in_stride / stride))
                conv_layer = nn.ConvTranspose2d

            for _ in range(scale - 1):
                layers += [
                    conv_layer(hidden_dim, hidden_dim, kernel_size=2, stride=2),
                    LayerNorm2D(hidden_dim),
                    nn.GELU(),
                ]

            layers.append(conv_layer(hidden_dim, hidden_dim, kernel_size=2, stride=2))
            self.feature_projs.append(nn.Sequential(*layers))

    def expand_features(self, x: torch.Tensor, padded_batch: PaddedBatch):
        assert self.feature_projs is not None, "No feature expansion is needed."
        B, _, C = x.shape
        H, W = padded_batch.feature_hws[0]

        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        if padded_batch.padding_mask is not None:
            padding_mask = padded_batch.padding_mask.view(B, 1, H, W)
        else:
            padding_mask = None

        out_x, out_padding_mask, out_hws = [], [], []
        for proj in self.feature_projs:
            x_proj = proj(x)
            out_hw = x_proj.shape[-2:]
            out_hws.append(out_hw)
            out_x.append(x_proj.flatten(2).transpose(1, 2))

            if padding_mask is not None:
                mask = F.interpolate(padding_mask.float(), size=out_hw)
                mask = mask.to(torch.bool).squeeze(1)
                out_padding_mask.append(mask)

        out_x = torch.cat(out_x, dim=1)
        out_padding_mask = out_padding_mask or None
        return out_x, out_padding_mask, out_hws

    def expand_features_packed(self, x: torch.Tensor, packed_batch: PackedBatch):
        assert self.feature_projs is not None, "No feature expansion is needed."
        x_list = list(torch.split(x, packed_batch.patch_lens, dim=1))
        out_x, out_hws, seq_lens = [], [], []

        for x, (h, w) in zip(x_list, packed_batch.feature_hws):
            _, _, c = x.shape
            x = x.view(1, h, w, c).permute(0, 3, 1, 2)
            seq_len = 0

            for proj in self.feature_projs:
                x_proj = proj(x)
                out_hws.append(x_proj.shape[-2:])
                seq_len += x_proj.shape[-2] * x_proj.shape[-1]
                out_x.append(x_proj.flatten(2).transpose(1, 2))

            seq_lens.append(seq_len)

        out_x = torch.cat(out_x, dim=1)
        return out_x, out_hws, seq_lens

    def generate_candidate_boxes(self, x: torch.Tensor, padded_batch: PaddedBatch):
        if self.num_feature_levels > 1:
            x, padding_masks, hws = self.expand_features(x, padded_batch)
        else:
            hws = padded_batch.feature_hws[:1]

            if padded_batch.padding_mask is not None:
                mask = padded_batch.padding_mask.view(
                    padded_batch.B, hws[0][0], hws[0][1]
                )
                padding_masks = [mask]
            else:
                padding_masks = None

        candidate_boxes = []
        for level_idx, (H, W) in enumerate(hws):
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, H - 1, H, dtype=torch.float32, device=x.device),
                torch.linspace(0, W - 1, W, dtype=torch.float32, device=x.device),
                indexing="ij",
            )
            grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
            grid = grid.expand(padded_batch.B, -1, -1, -1)  # [B, H, W, 2]

            if padding_masks is not None:
                valid_H = torch.sum(~padding_masks[level_idx][:, :, 0], 1)
                valid_W = torch.sum(~padding_masks[level_idx][:, 0, :], 1)

                # [B, 1, 1, 2]
                scale = torch.stack([valid_W, valid_H], dim=-1)[:, None, None, :]
                grid = (grid + 0.5) / scale
                wh = torch.ones_like(grid) * 0.05 * (2.0**level_idx)
            else:
                scale = torch.tensor([W, H], dtype=torch.float32, device=x.device)
                grid = (grid + 0.5) / scale
                wh = torch.ones_like(grid) * 0.05 * (2.0**level_idx)

            candidate_box = torch.cat((grid, wh), -1)
            candidate_box = candidate_box.view(padded_batch.B, -1, 4)  # [B, H*W, 4]
            candidate_boxes.append(candidate_box)  # [B, H*W, 4]

        candidate_boxes = torch.cat(candidate_boxes, dim=1)  # [B, sum_i(H_i*W_i), 4]

        valid = (candidate_boxes > 0.01) & (candidate_boxes < 0.99)
        sentinel = torch.tensor(float("inf"), device=x.device)  # scalar
        candidate_boxes = torch.log(candidate_boxes / (1 - candidate_boxes))

        valid = valid.all(-1, keepdim=True)  # [B, sum_i(H_i*W_i), 1]

        if padding_masks is not None:
            padding_masks = [m.flatten(1) for m in padding_masks]
            padding_masks = torch.cat(padding_masks, dim=1)  # [B, sum_i(H_i*W_i)]
            padding_masks = padding_masks.unsqueeze(-1)  # [B, sum_i(H_i*W_i), 1]
            valid = valid & ~padding_masks

        candidate_boxes = torch.where(valid, candidate_boxes, sentinel)
        x = torch.where(valid, x, torch.tensor(0.0, device=x.device))
        x = self.out_proj(x)

        return x, candidate_boxes

    def generate_candidate_boxes_packed(
        self, x: torch.Tensor, packed_batch: PackedBatch
    ):
        if self.num_feature_levels > 1:
            x, hws, seq_lens = self.expand_features_packed(x, packed_batch)
        else:
            hws = packed_batch.feature_hws
            seq_lens = packed_batch.patch_lens

        candidate_boxes = []
        for i, (H, W) in enumerate(hws):
            level_idx = i % self.num_feature_levels
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, H - 1, H, dtype=torch.float32, device=x.device),
                torch.linspace(0, W - 1, W, dtype=torch.float32, device=x.device),
                indexing="ij",
            )
            grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]

            scale = torch.tensor([W, H], dtype=torch.float32, device=x.device)
            grid = (grid + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0**level_idx)

            candidate_boxes.append(
                torch.cat((grid, wh), -1).view(1, -1, 4)
            )  # [1, H*W, 4]

        candidate_boxes = torch.cat(candidate_boxes, dim=1)  # [1, sum_i(H_i*W_i), 4]

        valid = (candidate_boxes > 0.01) & (candidate_boxes < 0.99)
        candidate_boxes = torch.log(candidate_boxes / (1 - candidate_boxes))
        sentinel = torch.tensor(float("inf"), device=x.device)  # scalar

        valid = valid.all(-1, keepdim=True)  # [1, sum_i(H_i*W_i), 1]
        candidate_boxes = torch.where(valid, candidate_boxes, sentinel)
        x = torch.where(valid, x, torch.tensor(0.0, device=x.device))
        x = self.out_proj(x)

        return x, candidate_boxes, seq_lens

    def _forward(self, x: torch.Tensor, padded_batch: PaddedBatch):
        # [B, sum_i(H_i*W_i), C], [1, sum_i(H_i*W_i), 4]
        x, candidate_boxes = self.generate_candidate_boxes(x, padded_batch)

        # [B, sum_i(H_i*W_i), 1]
        scores = self.score_embed(x)

        topk = min(self.topk, candidate_boxes.shape[1])
        # [B, topk]
        indices = torch.topk(scores[..., 0], topk, dim=1)[1]
        # [B, topk, 1]
        indices = indices.unsqueeze(-1).detach()

        # [B, sum_i(H_i*W_i), D] -> [B, topk, D]
        x = torch.gather(x, 1, indices.expand(-1, -1, x.shape[-1]))

        # [B, sum_i(H_i*W_i), 4] -> [B, topk, 4]
        topk_candidate_boxes = torch.gather(
            candidate_boxes, 1, indices.expand(-1, -1, candidate_boxes.shape[-1])
        )

        # [B, sum_i(H_i*W_i), num_classes]
        pred_logits = self.class_embed(x)
        # [B, sum_i(H_i*W_i), 4]
        deltas = self.bbox_embed(x)

        box_logits = deltas + topk_candidate_boxes

        enc_output = {"pred_logits": scores, "pred_boxes": candidate_boxes.sigmoid()}
        output = {"pred_logits": pred_logits, "pred_boxes": box_logits.sigmoid()}

        return box_logits, x, output, enc_output

    def _forward_packed(self, x: torch.Tensor, packed_batch: PackedBatch):
        # [1, sum_i(H_i*W_i), C], [1, sum_i(H_i*W_i), 4]
        x, candidate_boxes, seq_lens = self.generate_candidate_boxes_packed(
            x, packed_batch
        )

        # [1, sum_i(H_i*W_i), 1]
        scores = self.score_embed(x)

        all_indices = []
        start_idx = 0

        # each seq_len contains all positions from all feature levels for one image
        for seq_len in seq_lens:
            end_idx = start_idx + seq_len

            topk = min(self.topk, seq_len)
            # [1, topk]
            indices = torch.topk(scores[:, start_idx:end_idx, 0], topk, dim=1)[1]
            indices = indices + start_idx
            all_indices.append(indices)

            start_idx = end_idx

        # [B*num_candidates]
        indices = torch.cat(all_indices, dim=1).squeeze(0).detach()

        # [1, sum_i(H_i*W_i), D] -> [B, num_candidates, D]
        x = x.index_select(1, indices)
        x = x.view(-1, self.topk, x.shape[-1])

        # [1, sum_i(H_i*W_i), 4] -> [B, num_candidates, 4]
        topk_candidate_boxes = candidate_boxes.index_select(1, indices)
        topk_candidate_boxes = topk_candidate_boxes.view(-1, self.topk, 4)

        # [B, num_candidates, num_classes]
        pred_logits = self.class_embed(x)

        # [B, num_candidates, 4]
        deltas = self.bbox_embed(x)
        box_logits = deltas + topk_candidate_boxes

        enc_output = {
            "pred_logits": scores,
            "pred_boxes": candidate_boxes.sigmoid(),
            "seq_lens": seq_lens,
        }
        output = {"pred_logits": pred_logits, "pred_boxes": box_logits.sigmoid()}

        return box_logits, x, output, enc_output

    def forward(self, x: torch.Tensor, batch: BatchInfo):
        if isinstance(batch, PaddedBatch):
            return self._forward(x, batch)
        elif isinstance(batch, PackedBatch):
            return self._forward_packed(x, batch)
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")
