# ------------------------------------------------------------------------
# MixMatchDet
# Copyright (c) 2026 Hochschule Ruhr West
# Licensed under the Apache License, Version 2.0 [See LICENSE for details]
# ------------------------------------------------------------------------

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn

from src.config.mixmatchdet import MixMatchDetConfig, DetTokenConfig
from src.models.dinov2.dinov2 import DINOv2
from src.models.yolos.yolos import YOLOS
from src.util.batch import PackedBatch, PaddedBatch
from src.util.nested_tensor import NestedTensor
from src.util.weights import load_weights_from_url_or_path


class MixMatchDetBase(nn.Module, ABC):
    def __init__(self, config: MixMatchDetConfig, num_classes: int) -> None:
        super().__init__()
        self.__init_backbone(config)
        self.num_classes = num_classes
        self.num_det_tokens = 0
        self.det_tokens = None
        self.det_pos = None

    def __init_backbone(self, config: MixMatchDetConfig):
        kwargs = {}

        if config.vit.dinov2 is not None:
            backbone_cls = DINOv2
            self.num_register_tokens = config.vit.dinov2.num_register_tokens
            self.embed_dim = config.vit.dinov2.embed_dim
            self.patch_size = config.vit.dinov2.patch_size
            kwargs.update(config.vit.dinov2.model_dump())
        elif config.vit.yolos is not None:
            backbone_cls = YOLOS
            self.num_register_tokens = 0
            self.embed_dim = config.vit.yolos.embed_dim
            self.patch_size = config.vit.yolos.patch_size
            kwargs["num_det_tokens"] = (
                config.det_tokens.num if config.det_tokens is not None else 0
            )
            kwargs.update(config.vit.yolos.model_dump())
        else:
            raise ValueError("No valid config provided.")

        self.backbone = backbone_cls(
            aux_indices=config.vit.aux_indices,
            **kwargs,
        )

        state_dict = load_weights_from_url_or_path(config.vit.weights)
        self.backbone.load_pretrained(state_dict)

    def _init_det_tokens(self, config: DetTokenConfig):
        self.num_det_tokens = config.num

        self.det_tokens = nn.Parameter(torch.zeros(1, config.num, self.embed_dim))
        nn.init.trunc_normal_(self.det_tokens, std=0.02)

        if config.pos:
            self.det_pos = nn.Parameter(torch.zeros(1, config.num, self.embed_dim))
            nn.init.trunc_normal_(self.det_pos, std=0.02)

    @abstractmethod
    def _forward(
        self, x: torch.Tensor, padded_batch: PaddedBatch
    ) -> tuple[dict, dict | None]:
        """Forward pass for tensor input - must be implemented by subclasses."""
        pass

    @abstractmethod
    def _forward_packed(
        self, x: torch.Tensor, packed_batch: PackedBatch
    ) -> tuple[dict, dict | None]:
        """Forward pass for packed input - must be implemented by subclasses."""
        pass

    def forward(self, x: list[torch.Tensor] | NestedTensor) -> tuple[dict, dict | None]:
        if isinstance(x, NestedTensor):
            args = self.prepare_batch(x)
            return self._forward(*args)
        else:
            args = self.prepare_batch_packed(x)
            return self._forward_packed(*args)

    def prepare_batch_packed(
        self, image_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, PackedBatch]:
        """Prepare batch with packed sequence format."""
        batch_sequence = []
        feature_hws = []
        image_hws = []

        for x in image_list:
            image_hws.append((x.shape[-2], x.shape[-1]))
            x, feature_hw = self.backbone.prepare_tokens(
                x[None], self.det_tokens, self.det_pos
            )
            feature_hws.append(feature_hw)
            batch_sequence.append(x)

        x = torch.cat(batch_sequence, dim=1)
        x = x.contiguous()

        packed_batch = PackedBatch(
            image_hws=image_hws,
            feature_hws=feature_hws,
            num_register_tokens=self.num_register_tokens,
            num_det_tokens=self.num_det_tokens,
            device=x.device,
        )

        return x, packed_batch

    def prepare_batch(self, images: NestedTensor) -> tuple[torch.Tensor, PaddedBatch]:
        x, feature_hw = self.backbone.prepare_tokens(
            images.tensors, self.det_tokens, self.det_pos
        )

        if images.mask is not None:
            # [B, 1, H, W]
            mask = F.interpolate(images.mask.unsqueeze(1).float(), size=feature_hw)
            # [B, H*W]
            mask = mask.to(torch.bool).squeeze(1).flatten(1)
        else:
            mask = None

        padded_batch = PaddedBatch(
            image_hws=images.shapes,
            feature_hws=[feature_hw] * len(images),
            num_register_tokens=self.num_register_tokens,
            num_det_tokens=self.num_det_tokens,
            padding_mask=mask,
            device=x.device,
        )

        return x, padded_batch
