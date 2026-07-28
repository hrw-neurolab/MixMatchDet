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

import pickle
import torch
import torch.distributed as dist
import torch.nn.functional as F


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def sigmoid_focal_loss(
    inputs,
    targets,
    num_boxes,
    alpha: float = 0.25,
    gamma: float = 2,
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(
            size=(max_size - local_size,), dtype=torch.uint8, device="cuda"
        )
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def get_target_sizes(target):
    """Returns a tensor of size classes (0=small, 1=medium, 2=large) based on COCO area thresholds.

    Note: the area thresholds are scaled based on the original and augmented image sizes, to account for resizing.

    Args:
        target: dict with keys "aug_areas" (tensor of shape [num_targets_i]) and "orig_size" and "aug_size" (tuples of (H, W))

    Returns:
        size_cls: tensor of shape [num_targets_i] with values 0/1/2 for small/medium/large
    """
    areas = target["aug_areas"]  # [num_targets_i]
    H0, W0 = target["orig_size"]
    H1, W1 = target["aug_size"]
    scale_factor = ((H1 * W1) / (H0 * W0)).sqrt()  # Geometric mean
    thr_small = (32 * scale_factor) ** 2
    thr_large = (96 * scale_factor) ** 2
    cls = torch.full_like(areas, 1, dtype=torch.long)  # 1 = medium
    cls[areas < thr_small] = 0
    cls[areas >= thr_large] = 2
    return cls


@torch.no_grad()
def compute_match_stats(indices, targets, num_patch_candidates):
    """
    indices: list of (src_idx, tgt_idx) per batch element
    targets: list of dicts with key "aug_areas" containing area of each GT box in pixels
    num_patch_candidates: number of patch candidates (P)

    Returns: flat dict with stats
    """

    # counters
    stats = {
        "gt_small": 0,
        "gt_medium": 0,
        "gt_large": 0,
        "patch_small": 0,
        "patch_medium": 0,
        "patch_large": 0,
        "det_small": 0,
        "det_medium": 0,
        "det_large": 0,
    }

    for (src_idx, tgt_idx), tgt in zip(indices, targets):
        if len(tgt_idx) == 0:
            continue

        size_cls = get_target_sizes(tgt)  # [num_targets_i] 0/1/2 (s/m/l)

        # matched pairs
        matched_sizes = size_cls[tgt_idx]
        is_patch = (src_idx < num_patch_candidates).to(matched_sizes.device)

        for size_val, size_name in zip([0, 1, 2], ["small", "medium", "large"]):
            stats[f"gt_{size_name}"] += (size_cls == size_val).sum()
            mask = matched_sizes == size_val
            stats[f"patch_{size_name}"] += (is_patch & mask).sum()
            stats[f"det_{size_name}"] += (~is_patch & mask).sum()

    return stats


@torch.no_grad()
def aggregate_match_stats(stats, prefix=""):
    out = {}

    for size in ["small", "medium", "large"]:
        gt = stats[f"gt_{size}"]
        if gt > 0:
            out[f"{prefix}pct_patch_{size}"] = stats[f"patch_{size}"] / gt * 100
            out[f"{prefix}pct_det_{size}"] = stats[f"det_{size}"] / gt * 100
        else:
            out[f"{prefix}pct_patch_{size}"] = 0.0
            out[f"{prefix}pct_det_{size}"] = 0.0

        out[f"{prefix}gt_{size}"] = stats[f"gt_{size}"].float()

    return out
