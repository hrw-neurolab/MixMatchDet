# ------------------------------------------------------------------------
# MixMatchDet
# Copyright (c) 2026 Hochschule Ruhr West
# Licensed under the Apache License, Version 2.0 [See LICENSE for details]
# ------------------------------------------------------------------------

from typing import Literal

from pydantic import BaseModel

AttnClass = Literal["Attention", "MemEffAttention"]


class YOLOSConfig(BaseModel):
    img_size: int
    embed_dim: int
    depth: int
    num_heads: int
    base_hw: tuple[int, int]
    patch_size: int = 16
    in_chans: int = 3
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    qk_scale: float | None = None
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    is_distill: bool = False
    mid_hw: tuple[int, int] | None = None
    attn_class: AttnClass = "Attention"
