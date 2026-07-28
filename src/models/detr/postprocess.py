# ------------------------------------------------------------------------
# MixMatchDet
# Copyright (c) 2026 Hochschule Ruhr West
# This file has been modified by Hochschule Ruhr West.
# Licensed under the Apache License, Version 2.0 [See LICENSE for details]
# ------------------------------------------------------------------------
# Plain-DETR
# Copyright (c) 2023 Xi'an Jiaotong University & Microsoft Research Asia.
# Licensed under The MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch

from src.util.box_ops import box_cxcywh_to_xyxy


class PostProcessor:
    """This module converts the model's output into the format expected by the COCO API"""

    TOPK = 100

    def process_outputs(
        self, logits: torch.Tensor, boxes: torch.Tensor, orig_sizes: torch.Tensor
    ):
        B, N, C = logits.shape
        # convert to [x0, y0, x1, y1] format
        boxes = box_cxcywh_to_xyxy(boxes)

        prob = logits.sigmoid()
        topk = min(self.TOPK, N * C)
        scores, topk_indexes = torch.topk(prob.view(B, -1), topk, dim=1)
        labels = topk_indexes % C
        topk_indexes = topk_indexes // C
        topk_boxes = topk_indexes.unsqueeze(-1).expand(-1, -1, 4)
        boxes = torch.gather(boxes, 1, topk_boxes)

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = orig_sizes.unbind(1)
        scale = torch.stack([img_w, img_h, img_w, img_h], dim=1)  # [B, 4]
        boxes.mul_(scale[:, None, :].to(boxes.device))

        return scores, labels, boxes

    def __call__(self, outputs: dict, targets: list[dict[str, torch.Tensor]]):
        """Post-process the outputs of the model for COCO evaluation.

        Args:
            outputs: a dict containing the keys "pred_logits" and "pred_boxes"
                with the model outputs.
            targets: a list of dicts, one for each image in the batch, containing
                the keys "boxes" and "labels" with the ground truth annotations.
        Returns:
            tuple[list, list]:
                processed_outputs: a list of dicts, one for each image in the batch,
                    containing the keys "scores", "labels", and "boxes" with the
                    post-processed model outputs.
                processed_targets: a list of dicts, one for each image in the batch,
                    containing the keys "boxes", "labels" and "image_id" with the post-processed
                    targets.
        """
        scores, labels, boxes = self.process_outputs(
            outputs["pred_logits"],
            outputs["pred_boxes"],
            torch.stack([t["orig_size"] for t in targets]),
        )

        processed_outputs = [
            {"scores": s, "labels": l, "boxes": b}
            for s, l, b in zip(scores, labels, boxes)
        ]

        processed_targets = [
            dict(boxes=t["orig_boxes"], labels=t["labels"], image_id=t["image_id"])
            for t in targets
        ]

        return processed_outputs, processed_targets
