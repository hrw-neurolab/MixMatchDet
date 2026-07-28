# ------------------------------------------------------------------------
# MixMatchDet
# Copyright (c) 2026 Hochschule Ruhr West
# Licensed under the Apache License, Version 2.0 [See LICENSE for details]
# ------------------------------------------------------------------------

import torch
import albumentations as A
import numpy as np
from PIL.Image import Image

from src.data.ensure_divisible import EnsureDivisible
from src.util import box_ops


class BaseTransforms:
    transforms: A.Compose

    def __init__(self, train: bool, seed: int, label_map: dict | None = None):
        self.train = train
        self.seed = seed
        self.label_map = label_map

    def __call__(self, image: Image, target: list[dict]) -> tuple[torch.Tensor, dict]:
        target = [obj for obj in target if "iscrowd" not in obj or obj["iscrowd"] == 0]
        bboxes = [obj["bbox"] for obj in target]

        if self.label_map is not None:
            category_ids = [self.label_map[obj["category_id"]] for obj in target]
        else:
            category_ids = [obj["category_id"] for obj in target]

        H, W = image.height, image.width
        orig_size = torch.tensor([H, W], dtype=torch.float32)

        transformed = self.transforms(
            image=np.array(image), bboxes=bboxes, category_ids=category_ids
        )
        _, H, W = transformed["image"].shape
        aug_size = torch.tensor([H, W], dtype=torch.float32)

        bboxes = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
        bboxes = box_ops.box_xywh_to_cxcywh(bboxes)
        aug_areas = bboxes[:, 2] * bboxes[:, 3]

        # Normalize bboxes to [0, 1] range in cxcywh format
        bboxes = bboxes / torch.tensor([W, H, W, H], dtype=torch.float32)

        labels = torch.as_tensor(transformed["category_ids"], dtype=torch.int64)
        target_out = dict(
            boxes=bboxes,
            labels=labels,
            orig_size=orig_size,
            aug_size=aug_size,
            aug_areas=aug_areas,
        )

        if not self.train:
            orig_boxes = [obj["bbox"] for obj in target]
            orig_boxes = torch.tensor(orig_boxes, dtype=torch.float32).reshape(-1, 4)
            orig_boxes = box_ops.box_xywh_to_xyxy(orig_boxes)
            target_out["orig_boxes"] = orig_boxes

        return transformed["image"], target_out


class FixedSizeTransforms(BaseTransforms):
    def __init__(
        self,
        train: bool,
        image_size: tuple[int, int],
        patch_size: int,
        seed: int,
        label_map: dict | None = None,
    ):
        super().__init__(train, seed, label_map)

        assert (
            image_size[0] % patch_size == 0 and image_size[1] % patch_size == 0
        ), "Image size must be divisible by patch size."

        if train:
            transforms: list[A.BasicTransform | A.BaseCompose] = [
                A.HorizontalFlip(p=0.5),
                A.SmallestMaxSize(max_size_hw=image_size),
                A.OneOf(
                    [
                        A.Resize(*image_size, p=0.2),
                        A.RandomSizedBBoxSafeCrop(*image_size, p=0.2),
                        A.AtLeastOneBBoxRandomCrop(
                            *image_size, erosion_factor=0.5, p=0.6
                        ),
                    ],
                    p=1.0,
                ),
            ]
            bbox_params = A.BboxParams(
                format="coco",
                label_fields=["category_ids"],
                filter_invalid_bboxes=True,
                clip=True,
            )
        else:
            transforms = [A.Resize(*image_size)]
            bbox_params = A.BboxParams(format="coco", label_fields=["category_ids"])

        transforms.append(A.Normalize())
        transforms.append(A.ToTensorV2())

        self.transforms = A.Compose(
            transforms, bbox_params=bbox_params, seed=seed, telemetry=False
        )


class COCOTransforms(BaseTransforms):
    def __init__(
        self,
        train: bool,
        patch_size: int,
        seed: int,
        label_map: dict | None = None,
    ):
        super().__init__(train, seed, label_map)

        if train:
            transforms: list[A.BasicTransform | A.BaseCompose] = [
                A.HorizontalFlip(p=0.5)
            ]

            scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
            random_resize = A.Compose(
                [
                    A.OneOf([A.SmallestMaxSize(x) for x in scales], p=1.0),
                    A.LongestMaxSize(1333),
                ]
            )

            path_a = random_resize

            scales = [400, 500, 600]
            path_b = A.Compose(
                [
                    A.OneOf([A.SmallestMaxSize(x) for x in scales], p=1.0),
                    A.RandomSizedBBoxSafeCrop(384, 600),
                    random_resize,
                ]
            )

            transforms.append(A.OneOf([path_a, path_b], p=1.0))

            bbox_params = A.BboxParams(
                format="coco",
                label_fields=["category_ids"],
                filter_invalid_bboxes=True,
                clip=True,
            )

        else:
            transforms = [A.SmallestMaxSize(800), A.LongestMaxSize(1333)]
            bbox_params = A.BboxParams(format="coco", label_fields=["category_ids"])

        transforms.append(EnsureDivisible(divisor=patch_size))
        transforms.append(A.Normalize())
        transforms.append(A.ToTensorV2())

        self.transforms = A.Compose(
            transforms, bbox_params=bbox_params, seed=seed, telemetry=False
        )
