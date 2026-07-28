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
from scipy.optimize import linear_sum_assignment
from torch import nn

from src.config.training import FocalLossConfig, ScaleAwareMatchingConfig
from src.util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        focal_cost: FocalLossConfig | None = None,
        scale_aware_matching: ScaleAwareMatchingConfig | None = None,
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
            focal_cost: Focal loss configuration (if None, standard cross-entropy is used)
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_cost = focal_cost
        self.scale_aware_matching = scale_aware_matching
        if scale_aware_matching is not None:
            self.register_buffer(
                "allowed_patch", torch.as_tensor(scale_aware_matching.allowed_patch)
            )
            self.register_buffer(
                "allowed_det", torch.as_tensor(scale_aware_matching.allowed_det)
            )

        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"

    @torch.no_grad()
    def compute_geometry_cost(self, outputs, targets):
        """Compute the weighted L1 and GIoU matching costs."""
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )

        return self.cost_bbox * cost_bbox + self.cost_giou * cost_giou

    @torch.no_grad()
    def forward(
        self,
        outputs,
        targets,
        num_patch_candidates=None,
        geometry_cost=None,
        target_size_classes=None,
    ):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # Concat the target labels.
        tgt_ids = torch.cat([v["labels"] for v in targets])

        # Compute the classification cost.
        if self.focal_cost is None:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
            cost_class = -out_prob[:, tgt_ids]
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            alpha = self.focal_cost.alpha
            gamma = self.focal_cost.gamma
            neg_cost_class = (
                (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
            )
            pos_cost_class = (
                alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            )
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        if geometry_cost is None:
            geometry_cost = self.compute_geometry_cost(outputs, targets)
        elif geometry_cost.shape != cost_class.shape:
            raise ValueError(
                "Cached geometry cost shape does not match classification cost: "
                f"{geometry_cost.shape} != {cost_class.shape}"
            )

        # Final cost matrix
        C = geometry_cost + self.cost_class * cost_class

        sizes = [len(v["boxes"]) for v in targets]

        if "seq_lens" in outputs:
            # Handle patch candidate head format with seq_lens (enc output)
            seq_lens = outputs["seq_lens"]

            cost_blocks = []
            source_start = target_start = 0
            for source_size, target_size in zip(seq_lens, sizes):
                cost_blocks.append(
                    C[
                        source_start : source_start + source_size,
                        target_start : target_start + target_size,
                    ].flatten()
                )
                source_start += source_size
                target_start += target_size
            C = torch.cat(cost_blocks).cpu()

            indices = []
            start = 0
            for source_size, target_size in zip(seq_lens, sizes):
                end = start + source_size * target_size
                c = C[start:end].view(source_size, target_size)
                idx_i, idx_j = linear_sum_assignment(c)
                indices.append(
                    (
                        torch.as_tensor(idx_i, dtype=torch.int64),
                        torch.as_tensor(idx_j, dtype=torch.int64),
                    )
                )
                start = end
        else:
            # Standard format
            C = C.view(bs, num_queries, -1)
            if target_size_classes is not None:
                target_size_classes = torch.cat(target_size_classes)
                allowed_patch = self.allowed_patch[target_size_classes]
                allowed_det = self.allowed_det[target_size_classes]
                C[:, :num_patch_candidates].masked_fill_(
                    ~allowed_patch[None, None], 1e6
                )
                C[:, num_patch_candidates:].masked_fill_(~allowed_det[None, None], 1e6)

            C = C.cpu()

            indices = [
                linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
            ]
            indices = [
                (
                    torch.as_tensor(i, dtype=torch.int64),
                    torch.as_tensor(j, dtype=torch.int64),
                )
                for i, j in indices
            ]

        return indices
