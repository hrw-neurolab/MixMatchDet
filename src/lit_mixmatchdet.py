# ------------------------------------------------------------------------
# MixMatchDet
# Copyright (c) 2026 Hochschule Ruhr West
# Licensed under the Apache License, Version 2.0 [See LICENSE for details]
# ------------------------------------------------------------------------

from collections import defaultdict

import torch
import lightning as L
import math
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR, SequentialLR
from torchmetrics import MetricCollection
from torchmetrics.detection import MeanAveragePrecision as MAP

from src.config.mixmatchdet import MixMatchDetConfig
from src.config.training import TrainingConfig
from src.models.mixmatchdet import mixmatchdet
from src.models.detr.criterion import SetCriterion
from src.util.misc import aggregate_match_stats, reduce_dict
from src.util.nested_tensor import NestedTensor
from src.util.param_groups import create_param_groups, freeze_param_groups
from src.models.detr.postprocess import PostProcessor
from src.util.warmup_lr import WarmupLR

BatchType = tuple[list[torch.Tensor] | NestedTensor, list[dict[str, torch.Tensor]]]


class LitMixMatchDet(L.LightningModule):
    def __init__(
        self,
        model_config: MixMatchDetConfig | dict,
        training_config: TrainingConfig | dict,
    ):
        super().__init__()

        # Necessary for checkpointing
        if isinstance(model_config, dict):
            model_config = MixMatchDetConfig.model_validate(model_config)

        if isinstance(training_config, dict):
            training_config = TrainingConfig.model_validate(training_config)

        self.tc = training_config

        self.save_hyperparameters(
            {
                "model_config": model_config.model_dump(),
                "training_config": training_config.model_dump(),
            }
        )

        self.model = mixmatchdet(
            config=model_config, num_classes=self.tc.data.num_classes
        )

        freeze_param_groups(self.model, self.tc.optimizer.param_groups)

        self.criterion = SetCriterion(
            num_classes=self.tc.data.num_classes,
            matcher=self.tc.criterion.matcher,
            losses=self.tc.criterion.losses,
            focal_loss=self.tc.criterion.focal_loss,
        )

        self.postprocess = PostProcessor()
        self.val_metrics = MetricCollection(MAP(), prefix="val/")
        self.match_stats = defaultdict(
            lambda: torch.tensor(0, device=self.device, dtype=torch.long)
        )

    def forward(self, x: list[torch.Tensor] | NestedTensor):
        return self.model.forward(x)

    def training_step(self, batch: BatchType):
        images, targets = batch
        outputs = self(images)

        loss_dict, _ = self.criterion(outputs, targets)
        weight_dict = self.tc.criterion.weight_dict

        loss = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )

        loss_dict_unscaled = {f"train/{k}_unscaled": v for k, v in loss_dict.items()}

        self.log(
            "train/loss",
            loss,
            batch_size=len(images),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        self.log_dict(
            loss_dict_unscaled,
            batch_size=len(images),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch: BatchType):
        images, targets = batch
        outputs = self(images)

        loss_dict, match_stats = self.criterion(outputs, targets, match_stats=True)
        weight_dict = self.tc.criterion.weight_dict
        loss = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )

        self.log(
            "val/loss", loss, batch_size=len(images), sync_dist=True, on_epoch=True
        )

        outputs, targets = self.postprocess(outputs, targets)
        self.val_metrics.update(outputs, targets)

        if match_stats is not None:
            for k, v in match_stats.items():
                self.match_stats[k] += v

        return loss

    def on_validation_epoch_end(self):
        results = self.val_metrics.compute()
        # Remove the 'classes' key, because it is not a single scalar value
        results.pop("val/classes", None)
        self.log_dict(results)
        self.val_metrics.reset()

        if len(self.match_stats) == 0:
            return

        stats = reduce_dict(self.match_stats, average=False)
        if self.global_rank == 0:
            stats = aggregate_match_stats(stats, prefix="match/")
            self.log_dict(stats, rank_zero_only=True)

        self.match_stats.clear()

    def configure_optimizers(self):
        cfg = self.tc.optimizer
        param_groups = create_param_groups(
            self.model, cfg.param_groups, self.global_rank == 0
        )

        if cfg.optimizer == "adamw":
            optimizer = torch.optim.AdamW(param_groups)
        else:
            raise NotImplementedError

        if cfg.warmup_steps is not None:
            warmup_steps = cfg.warmup_steps
        elif cfg.warmup_epochs is not None:
            assert self.trainer.max_epochs is not None
            steps_per_epoch = (
                self.trainer.estimated_stepping_batches / self.trainer.max_epochs
            )
            warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)

        warmup_lr = WarmupLR(optimizer, num_warmup=warmup_steps, warmup_strategy="cos")

        if cfg.step_lr_milestones is not None:
            warmup_lr = {"scheduler": warmup_lr, "interval": "step", "frequency": 1}
            main_lr = MultiStepLR(optimizer, milestones=cfg.step_lr_milestones)
            scheduler = [warmup_lr, main_lr]

        elif cfg.cos_lr_min_ratio is not None:
            T_max = int(self.trainer.estimated_stepping_batches) - warmup_steps
            r = cfg.cos_lr_min_ratio
            lr_lambda = (
                lambda step, r=r, T=T_max: r
                + (1 - r) * (1 + math.cos(math.pi * step / T)) / 2
            )
            main_lr = LambdaLR(optimizer, lr_lambda=lr_lambda)
            scheduler = {
                "scheduler": SequentialLR(
                    optimizer,
                    schedulers=[warmup_lr, main_lr],
                    milestones=[warmup_steps],
                ),
                "interval": "step",
                "frequency": 1,
            }
            scheduler = [scheduler]
        else:
            raise NotImplementedError

        return [optimizer], scheduler
