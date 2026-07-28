# ------------------------------------------------------------------------
# MixMatchDet
# Copyright (c) 2026 Hochschule Ruhr West
# Licensed under the Apache License, Version 2.0 [See LICENSE for details]
# ------------------------------------------------------------------------

from typing import Any

import cv2
import numpy as np
from pydantic import Field
from albumentations.core.transforms_interface import (
    BaseTransformInitSchema,
    DualTransform,
)
from albumentations.core.type_definitions import ALL_TARGETS
from albumentations.augmentations.geometric import functional as fgeometric


class EnsureDivisible(DualTransform):
    """Resize the input so that height and width are divisible by a certain factor."""

    _targets = ALL_TARGETS

    class InitSchema(BaseTransformInitSchema):
        divisor: int = Field(ge=1)

    def __init__(self, divisor: int, p: float = 1):
        super().__init__(p=p)
        self.divisor = divisor

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        """Apply resizing to the image.

        Args:
            img (np.ndarray): Image to resize.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Resized image.

        """
        height, width = img.shape[:2]

        new_height = ((height + self.divisor - 1) // self.divisor) * self.divisor
        new_width = ((width + self.divisor - 1) // self.divisor) * self.divisor

        return fgeometric.resize(
            img, (new_height, new_width), interpolation=cv2.INTER_LINEAR
        )

    def apply_to_mask(self, mask: np.ndarray, **params: Any) -> np.ndarray:
        """Apply resizing to the mask.

        Args:
            mask (np.ndarray): Mask to resize.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Resized mask.

        """
        height, width = mask.shape[:2]
        new_height = ((height + self.divisor - 1) // self.divisor) * self.divisor
        new_width = ((width + self.divisor - 1) // self.divisor) * self.divisor

        return fgeometric.resize(
            mask, (new_height, new_width), interpolation=cv2.INTER_NEAREST
        )

    def apply_to_bboxes(self, bboxes: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the transform to bounding boxes.

        Args:
            bboxes (np.ndarray): Bounding boxes to transform.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Transformed bounding boxes which are scale invariant.

        """
        # Bounding box coordinates are scale invariant
        return bboxes

    def apply_to_keypoints(self, keypoints: np.ndarray, **params: Any) -> np.ndarray:
        """Apply resizing to keypoints.

        Args:
            keypoints (np.ndarray): Keypoints to resize.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Resized keypoints.

        """
        height, width = params["shape"][:2]
        new_height = ((height + self.divisor - 1) // self.divisor) * self.divisor
        new_width = ((width + self.divisor - 1) // self.divisor) * self.divisor
        return fgeometric.keypoints_scale(
            keypoints, new_width / width, new_height / height
        )
