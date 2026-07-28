# ------------------------------------------------------------------------
# MixMatchDet
# Copyright (c) 2026 Hochschule Ruhr West
# Licensed under the Apache License, Version 2.0 [See LICENSE for details]
# ------------------------------------------------------------------------

import yaml
from typing import Any
from pathlib import Path

from pydantic import BaseModel, Field

from src.config.mixmatchdet import MixMatchDetConfig
from src.config.training import TrainingConfig


def deep_merge(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            out[k] = deep_merge(out.get(k), v) if k in out else v
        return out
    return b


def load_with_extends(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    base = {}
    if "extends" in data:
        base_path = (path.parent / data.pop("extends")).resolve()
        base = load_with_extends(base_path)

    return deep_merge(base, data)


class WandbConfig(BaseModel):
    project_name: str
    group: str | None = None
    tags: list[str] = Field(default_factory=list)


class RunConfig(BaseModel):
    run_name: str
    model: MixMatchDetConfig
    training: TrainingConfig
    trainer_kwargs: dict[str, Any]
    save_dir: str = "logs"
    seed: int = 42
    wandb: WandbConfig | None = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "RunConfig":
        config = load_with_extends(Path(yaml_path))
        return cls.model_validate(config)
