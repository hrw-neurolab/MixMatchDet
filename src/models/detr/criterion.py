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

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.training import FocalLossConfig, MatcherConfig
from src.models.detr.matcher import HungarianMatcher
from src.util import box_ops
from src.util.misc import (
    accuracy,
    compute_match_stats,
    get_target_sizes,
    get_world_size,
    is_dist_avail_and_initialized,
    sigmoid_focal_loss,
)


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        num_classes: int,
        matcher: MatcherConfig,
        losses: list[str],
        focal_loss: FocalLossConfig,
    ):
        """Create the criterion.

        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and candidates
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_loss: configuration for focal loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher(
            cost_class=matcher.cost_class,
            cost_bbox=matcher.cost_bbox,
            cost_giou=matcher.cost_giou,
            focal_cost=matcher.focal_cost,
            scale_aware_matching=matcher.scale_aware_matching,
        )
        self.losses = losses
        self.focal_loss = focal_loss

    def loss_labels(self, outputs, targets, indices, idx, num_boxes, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )

        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )

        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        loss_ce = sigmoid_focal_loss(
            src_logits,
            target_classes_onehot,
            num_boxes,
            alpha=self.focal_loss.alpha,
            gamma=self.focal_loss.gamma,
        )

        seq_lens = outputs.get("seq_lens", None)
        if seq_lens is not None:
            loss_ce = loss_ce * (sum(seq_lens) / len(seq_lens))
        else:
            loss_ce = loss_ce * src_logits.shape[1]

        losses = {"loss_ce": loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    # This loss actually boosts performance, but was not included in our paper
    def loss_labels_vfl(self, outputs, targets, indices, idx, num_boxes, log=True):
        assert "pred_boxes" in outputs
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        ious, _ = box_ops.box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes),
        )
        ious = torch.diag(ious).detach()

        src_logits = outputs["pred_logits"]
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = src_logits.sigmoid().detach()
        weight = (
            self.focal_loss.alpha * pred_score.pow(self.focal_loss.gamma) * (1 - target)
            + target_score
        )

        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction="none"
        )

        loss_ce = loss.mean(1).sum() / num_boxes

        # Apply the same seq_lens scaling logic as the original loss_labels
        seq_lens = outputs.get("seq_lens", None)
        if seq_lens is not None:
            loss_ce = loss_ce * (sum(seq_lens) / len(seq_lens))
        else:
            loss_ce = loss_ce * src_logits.shape[1]

        # Use the "loss_ce" key to map seamlessly to the existing weight_dict
        losses = {"loss_ce": loss_ce}

        if log:
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    def loss_boxes(self, outputs, targets, indices, idx, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices, seq_lens=None):
        # permute predictions following indices
        if seq_lens is not None:
            # Patch candidate head format: predictions are [1, sum_i(seq_lens_i), ...]
            # All batch_idx are 0, src_idx needs cumulative offsets
            batch_idx = torch.zeros(
                sum(len(src) for src, _ in indices), dtype=torch.int64
            )

            src_indices = []
            offset = 0
            for i, (src, _) in enumerate(indices):
                src_indices.append(src + offset)
                offset += seq_lens[i]

            src_idx = torch.cat(src_indices)
            return batch_idx, src_idx
        else:
            # Standard format: predictions are [B, num_queries, ...]
            batch_idx = torch.cat(
                [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
            )
            src_idx = torch.cat([src for (src, _) in indices])
            return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, idx, num_boxes, **kwargs):
        loss_map = {
            # "labels": self.loss_labels_vfl,
            "labels": self.loss_labels,
            "boxes": self.loss_boxes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, idx, num_boxes, **kwargs)

    def forward(self, outputs, targets, match_stats=False):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        aux_outputs = outputs.get("aux_outputs", None)
        enc_outputs = outputs.get("enc_outputs", None)
        outputs = {
            k: v for k, v in outputs.items() if k not in ["aux_outputs", "enc_outputs"]
        }

        # Retrieve the matching between the outputs of the last layer and the targets
        num_patch_candidates = outputs.get("num_patch_candidates", None)
        target_size_classes = None
        if self.matcher.scale_aware_matching is not None:
            assert (
                num_patch_candidates is not None
            ), "Scale-aware matching requires num_patch_candidates in outputs"
            target_size_classes = [get_target_sizes(target) for target in targets]

        indices = self.matcher(
            outputs,
            targets,
            num_patch_candidates,
            target_size_classes=target_size_classes,
        )

        if match_stats and num_patch_candidates is not None:
            match_stats = compute_match_stats(indices, targets, num_patch_candidates)
        else:
            match_stats = None

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute source permutation indices once for all losses
        idx = self._get_src_permutation_idx(indices, outputs.get("seq_lens", None))

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {"log": True} if loss == "labels" else {}
            losses.update(
                self.get_loss(loss, outputs, targets, indices, idx, num_boxes, **kwargs)
            )

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if aux_outputs is not None:
            for i, aux_output in enumerate(aux_outputs):
                indices = self.matcher(
                    aux_output,
                    targets,
                    num_patch_candidates,
                    target_size_classes=target_size_classes,
                )
                seq_lens = aux_output.get("seq_lens", None)
                idx = self._get_src_permutation_idx(indices, seq_lens)
                for loss in self.losses:
                    l_dict = self.get_loss(
                        loss, aux_output, targets, indices, idx, num_boxes
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if enc_outputs is not None:
            bin_targets = copy.deepcopy(targets)

            if target_size_classes is not None:
                for target, size_classes in zip(bin_targets, target_size_classes):
                    keep = self.matcher.allowed_patch[size_classes]
                    for key in ("labels", "boxes", "aug_areas"):
                        if key in target:
                            target[key] = target[key][keep]

            for bt in bin_targets:
                bt["labels"] = torch.zeros_like(bt["labels"])

            orig_num_classes = self.num_classes
            self.num_classes = 1

            # Encoder heads score the same fixed candidate boxes, so their geometric
            # matching cost is identical within this criterion forward.
            geometry_cost = self.matcher.compute_geometry_cost(
                enc_outputs[0], bin_targets
            )

            for i, enc_output in enumerate(enc_outputs):
                indices = self.matcher(
                    enc_output,
                    bin_targets,
                    None,
                    geometry_cost=geometry_cost,
                )
                seq_lens = enc_output.get("seq_lens", None)
                idx = self._get_src_permutation_idx(indices, seq_lens)
                l_dict = self.get_loss(
                    "labels",
                    enc_output,
                    bin_targets,
                    indices,
                    idx,
                    num_boxes,
                )
                l_dict = {k + f"_enc_{i}": v for k, v in l_dict.items()}
                losses.update(l_dict)

            self.num_classes = orig_num_classes

        return losses, match_stats
