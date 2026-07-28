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

import torch


class NestedTensor(object):
    def __init__(
        self,
        tensors: torch.Tensor,
        shapes: list[tuple[int, int]],
        mask: torch.Tensor | None = None,
    ):
        self.tensors = tensors
        self.shapes = shapes
        self.mask = mask

    def to(self, device, non_blocking=False):
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            mask = mask.to(device, non_blocking=non_blocking)
        return NestedTensor(cast_tensor, self.shapes, mask)

    def record_stream(self, *args, **kwargs):
        self.tensors.record_stream(*args, **kwargs)
        if self.mask is not None:
            self.mask.record_stream(*args, **kwargs)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

    def __len__(self):
        return self.tensors.shape[0]

    @classmethod
    def from_tensor_list(cls, tensor_list: list[torch.Tensor]) -> "NestedTensor":
        assert tensor_list[0].ndim == 3

        B = len(tensor_list)
        C = tensor_list[0].shape[0]
        device, dtype = tensor_list[0].device, tensor_list[0].dtype
        hs = [t.shape[-2] for t in tensor_list]
        ws = [t.shape[-1] for t in tensor_list]
        H = max(hs)
        W = max(ws)
        shapes = list(zip(hs, ws))

        if all([h == H and w == W for h, w in shapes]):
            batch = torch.stack(tensor_list, dim=0)
            return cls(batch, shapes)

        batch = torch.zeros((B, C, H, W), dtype=dtype, device=device)
        # masks = torch.zeros((B, H, W), dtype=torch.bool, device=device)

        for i, t in enumerate(tensor_list):
            c, h, w = t.shape
            batch[i, :c, :h, :w] = t
            # masks[i, :h, :w] = True

        return cls(batch, list(zip(hs, ws)))
