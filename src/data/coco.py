# ------------------------------------------------------------------------
# MixMatchDet
# Copyright (c) 2026 Hochschule Ruhr West
# Licensed under the Apache License, Version 2.0 [See LICENSE for details]
# ------------------------------------------------------------------------

from functools import partial

import torch
import lightning as L
from torchvision import datasets

from src.config.training import DataConfig
from src.data.transforms import COCOTransforms, FixedSizeTransforms
from src.util.nested_tensor import NestedTensor


class CocoDetection(datasets.CocoDetection):
    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        image_id = torch.tensor([image_id])
        target = {"image_id": image_id, **target}
        return img, target


def collate_fn_list(batch):
    return list(zip(*batch))


def collate_fn_nested(batch):
    images, targets = list(zip(*batch))
    images = NestedTensor.from_tensor_list(list(images))
    return images, targets


class COCO(L.LightningDataModule):
    def __init__(self, config: DataConfig, patch_size: int, seed: int):
        super().__init__()

        self.image_dir_train = config.image_dir_train
        self.ann_file_train = config.ann_file_train
        self.image_dir_val = config.image_dir_val
        self.ann_file_val = config.ann_file_val
        self.patch_size = patch_size
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers

        self.transforms_class = partial(
            COCOTransforms, patch_size=patch_size, seed=seed
        )

        if config.image_size == "list":
            self.collate_fn = collate_fn_list
        elif config.image_size == "padded":
            self.collate_fn = collate_fn_nested
        else:
            self.transforms_class = partial(
                FixedSizeTransforms,
                image_size=config.image_size,
                patch_size=patch_size,
                seed=seed,
            )
            self.collate_fn = collate_fn_nested

    def train_dataloader(self):
        transforms = self.transforms_class(train=True)

        train_dataset = CocoDetection(
            root=self.image_dir_train,
            annFile=self.ann_file_train,
            transforms=transforms,
        )

        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        transforms = self.transforms_class(train=False)

        val_dataset = CocoDetection(
            root=self.image_dir_val,
            annFile=self.ann_file_val,
            transforms=transforms,
        )

        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=True,
        )
