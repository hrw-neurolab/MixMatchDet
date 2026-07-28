# ------------------------------------------------------------------------
# MixMatchDet
# Copyright (c) 2026 Hochschule Ruhr West
# Licensed under the Apache License, Version 2.0 [See LICENSE for details]
# ------------------------------------------------------------------------

from typing import Literal

from pydantic import BaseModel

AttnClass = Literal["SelfAttention", "MemEffSelfAttention"]


class DINOv2Config(BaseModel):
    img_size: int
    patch_size: int
    depth: int
    num_heads: int
    embed_dim: int
    in_chans: int = 3
    mlp_ratio: int = 4
    qkv_bias: bool = True
    ffn_bias: bool = True
    proj_bias: bool = True
    init_values: float | None = 1.0
    num_register_tokens: int = 0
    interpolate_antialias: bool = False
    interpolate_offset: float = 0.1
    attn_class: AttnClass = "SelfAttention"
    img_side_minmax: tuple[int, int]
