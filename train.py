# ------------------------------------------------------------------------
# MixMatchDet
# Copyright (c) 2026 Hochschule Ruhr West
# Licensed under the Apache License, Version 2.0 [See LICENSE for details]
# ------------------------------------------------------------------------

import argparse
import warnings

import torch

torch.set_float32_matmul_precision("medium")

from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    OnExceptionCheckpoint,
)

from src.config.run import RunConfig
from src.data import COCO
from src.lit_mixmatchdet import LitMixMatchDet


def suppress_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r".*It is recommended to use .* when logging on epoch level in distributed setting to accumulate the metric across devices.*",
    )
    warnings.filterwarnings(
        "ignore", message=r"^Grad strides do not match bucket view strides.*"
    )


def build_loggers(config: RunConfig, args: argparse.Namespace):
    loggers = []
    csv_logger = CSVLogger(version=config.run_name, save_dir=config.save_dir)
    loggers.append(csv_logger)

    if config.wandb and args.resume_id is not None:
        wandb_logger = WandbLogger(
            project=config.wandb.project_name,
            id=args.resume_id,
            resume="must",
            save_dir=config.save_dir,
        )
        loggers.append(wandb_logger)

    elif config.wandb:
        wandb_logger = WandbLogger(
            project=config.wandb.project_name,
            name=config.run_name,
            save_dir=config.save_dir,
            group=config.wandb.group,
            tags=config.wandb.tags,
        )
        loggers.append(wandb_logger)

    return loggers


def build_callbacks(config: RunConfig):
    checkpoint_callback = ModelCheckpoint(
        monitor="val/map",
        save_top_k=1,
        mode="max",
        save_last=True,
        auto_insert_metric_name=False,
    )

    exception_checkpoint_callback = OnExceptionCheckpoint(
        ".", f"on_exception_{config.run_name}"
    )

    lr_monitor_callback = LearningRateMonitor(logging_interval="step")

    callbacks = [
        checkpoint_callback,
        lr_monitor_callback,
        exception_checkpoint_callback,
    ]

    return callbacks


def update_criterion(config: RunConfig):
    weight_dict = config.training.criterion.weight_dict
    extra_weight_dict = {}
    num_aux_layers = len(config.model.vit.aux_indices)

    for i in range(num_aux_layers):
        extra_weight_dict.update({f"{k}_{i}": v for k, v in weight_dict.items()})

    if config.model.patch_candidates is not None:
        for i in range(num_aux_layers + 1):
            extra_weight_dict.update(
                {f"{k}_enc_{i}": v for k, v in weight_dict.items()}
            )

    config.training.criterion.weight_dict.update(extra_weight_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MixMatchDet model")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, performs a dry run without saving logs or checkpoints",
    )
    parser.add_argument(
        "--resume-id",
        type=str,
        default=None,
        help="WandB run ID to resume from",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    args = parser.parse_args()

    print(f"Loading configuration from: {args.config}")
    config = RunConfig.from_yaml(args.config)

    suppress_warnings()
    seed_everything(config.seed, workers=True)

    trainer_kwargs = {**config.trainer_kwargs}

    if not args.dry_run:
        trainer_kwargs["logger"] = build_loggers(config, args)
        trainer_kwargs["callbacks"] = build_callbacks(config)
    else:
        trainer_kwargs["devices"] = [0]
        trainer_kwargs["strategy"] = "auto"
        trainer_kwargs["logger"] = False
        trainer_kwargs["fast_dev_run"] = 5
        trainer_kwargs["detect_anomaly"] = True

    trainer = Trainer(**trainer_kwargs)

    update_criterion(config)
    model = LitMixMatchDet(config.model, config.training)

    data = COCO(
        config=config.training.data,
        patch_size=model.model.patch_size,
        seed=config.seed,
    )

    if args.ckpt_path:
        trainer.fit(
            model, datamodule=data, ckpt_path=args.ckpt_path, weights_only=False
        )
    else:
        trainer.fit(model, datamodule=data, weights_only=False)
