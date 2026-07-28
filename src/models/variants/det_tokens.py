# ------------------------------------------------------------------------
# MixMatchDet
# Copyright (c) 2026 Hochschule Ruhr West
# Licensed under the Apache License, Version 2.0 [See LICENSE for details]
# ------------------------------------------------------------------------

import torch

from src.config.mixmatchdet import MixMatchDetConfig
from src.models.detr.head import DETRHead
from src.models.variants.base import MixMatchDetBase
from src.util.batch import PackedBatch, PaddedBatch


class MixMatchDet(MixMatchDetBase):
    det_tokens: torch.Tensor

    def __init__(self, config: MixMatchDetConfig, num_classes: int) -> None:
        super().__init__(config, num_classes)
        assert config.det_tokens is not None, "[DET] token config is required."

        self._init_det_tokens(config.det_tokens)
        self.detr_head = DETRHead(self.embed_dim, num_classes)

    def _forward_packed(self, x: torch.Tensor, packed_batch: PackedBatch):
        det_tokens, _ = self.backbone.forward_det_tokens(x, packed_batch)

        # [num_aux+1, B, num_det_tokens, C]
        det_tokens = det_tokens.view(
            -1, packed_batch.B, self.num_det_tokens, self.embed_dim
        )

        out, _ = self.detr_head(det_tokens)
        return out

    def _forward(self, x: torch.Tensor, padded_batch: PaddedBatch):
        det_tokens, _ = self.backbone.forward_det_tokens(x, padded_batch)

        out, _ = self.detr_head(det_tokens)
        return out
