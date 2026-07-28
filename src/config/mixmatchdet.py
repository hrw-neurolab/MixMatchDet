# ------------------------------------------------------------------------
# MixMatchDet
# Copyright (c) 2026 Hochschule Ruhr West
# Licensed under the Apache License, Version 2.0 [See LICENSE for details]
# ------------------------------------------------------------------------

from pydantic import BaseModel

from src.config.dinov2 import DINOv2Config
from src.config.yolos import YOLOSConfig


class PatchCandidateConfig(BaseModel):
    target_strides: list[int]
    topk: int


class DetTokenConfig(BaseModel):
    num: int
    pos: bool


class ViTConfig(BaseModel):
    weights: str
    aux_indices: list[int]
    dinov2: DINOv2Config | None = None
    yolos: YOLOSConfig | None = None


class MixMatchDetConfig(BaseModel):
    vit: ViTConfig
    det_tokens: DetTokenConfig | None = None
    patch_candidates: PatchCandidateConfig | None = None
