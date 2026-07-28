# ------------------------------------------------------------------------
# MixMatchDet
# Copyright (c) 2026 Hochschule Ruhr West
# Licensed under the Apache License, Version 2.0 [See LICENSE for details]
# ------------------------------------------------------------------------

from typing import Literal

from pydantic import BaseModel


class FocalLossConfig(BaseModel):
    alpha: float
    gamma: float


class ScaleAwareMatchingConfig(BaseModel):
    allowed_patch: tuple[bool, bool, bool]  # Small, Medium, Large
    allowed_det: tuple[bool, bool, bool]  # Small, Medium, Large


class MatcherConfig(BaseModel):
    cost_class: float
    cost_bbox: float
    cost_giou: float
    focal_cost: FocalLossConfig
    scale_aware_matching: ScaleAwareMatchingConfig | None = None


class SetCriterionConfig(BaseModel):
    matcher: MatcherConfig
    weight_dict: dict[str, float]
    focal_loss: FocalLossConfig
    losses: list[str]


class LayerDecayConfig(BaseModel):
    rate: float
    regex: str = r"(?:^|\.)blocks\.(\d+)\."
    default_layer: int = 0
    split_blocks: bool = True


class ParamGroupConfig(BaseModel):
    regex: str
    lr: float = 0.0
    weight_decay: float = 0.0
    freeze: bool = False
    layer_decay: LayerDecayConfig | None = None


class OptimizerConfig(BaseModel):
    optimizer: Literal["adamw"]
    param_groups: list[ParamGroupConfig]
    warmup_steps: int | None = None
    warmup_epochs: int | None = None
    step_lr_milestones: list[int] | None = None
    cos_lr_min_ratio: float | None = None


class DataConfig(BaseModel):
    image_dir_train: str
    ann_file_train: str
    image_dir_val: str
    ann_file_val: str
    num_classes: int
    """Without background class!"""
    batch_size: int
    image_size: tuple[int, int] | Literal["list", "padded"]
    num_workers: int


class TrainingConfig(BaseModel):
    data: DataConfig
    criterion: SetCriterionConfig
    optimizer: OptimizerConfig
