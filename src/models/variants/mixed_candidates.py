# ------------------------------------------------------------------------
# MixMatchDet
# Copyright (c) 2026 Hochschule Ruhr West
# Licensed under the Apache License, Version 2.0 [See LICENSE for details]
# ------------------------------------------------------------------------

import torch

from src.config.mixmatchdet import MixMatchDetConfig
from src.models.detr.patch_candidate_head import PatchCandidateHead
from src.models.detr.head import DETRHead
from src.models.variants.base import MixMatchDetBase
from src.util.batch import PackedBatch, PaddedBatch


class MixMatchDet(MixMatchDetBase):
    det_tokens: torch.Tensor

    def __init__(self, config: MixMatchDetConfig, num_classes: int) -> None:
        super().__init__(config, num_classes)
        assert config.det_tokens is not None, "[DET] token config is required."
        assert (
            config.patch_candidates is not None
        ), "Patch candidate config is required."

        self._init_det_tokens(config.det_tokens)

        self.patch_candidate_head = PatchCandidateHead(
            topk=config.patch_candidates.topk,
            in_stride=self.patch_size,
            target_strides=config.patch_candidates.target_strides,
            hidden_dim=self.embed_dim,
            num_classes=num_classes,
        )

        self.detr_head = DETRHead(self.embed_dim, num_classes)

    def __postprocess(self, out: dict):
        out["num_patch_candidates"] = self.patch_candidate_head.topk
        return out

    def _forward_packed(self, x: torch.Tensor, packed_batch: PackedBatch):
        *_, out = self.backbone.forward_mixed_candidates(
            x,
            packed_batch,
            self.patch_candidate_head,
            self.detr_head,
        )

        return self.__postprocess(out)

    def _forward(self, x: torch.Tensor, padded_batch: PaddedBatch):
        *_, out = self.backbone.forward_mixed_candidates(
            x,
            padded_batch,
            self.patch_candidate_head,
            self.detr_head,
        )

        return self.__postprocess(out)
