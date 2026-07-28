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

from torch import nn
import torch

from src.models.detr.mlp import MLP


class DETRHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()

        # For focal loss, we don't use the no-object class
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, x: torch.Tensor):
        assert x.dim() == 4

        # [num_outputs, batch_size, num_queries, num_classes]
        logits = self.class_embed(x)

        # [num_outputs, batch_size, num_queries, 4]
        box_logits = self.bbox_embed(x)
        boxes = box_logits.sigmoid()

        out = {"pred_logits": logits[-1], "pred_boxes": boxes[-1]}

        if x.shape[0] == 1:
            return out, box_logits[-1]

        # Add auxiliary outputs if there is more than one output
        out["aux_outputs"] = [
            {"pred_logits": l, "pred_boxes": b} for l, b in zip(logits[:-1], boxes[:-1])
        ]

        return out, box_logits[-1]
