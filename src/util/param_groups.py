# ------------------------------------------------------------------------
# MixMatchDet
# Copyright (c) 2026 Hochschule Ruhr West
# Licensed under the Apache License, Version 2.0 [See LICENSE for details]
# ------------------------------------------------------------------------

import re
from collections import defaultdict

import torch.nn as nn

from src.config.training import LayerDecayConfig, ParamGroupConfig


def _get_block_layer_decay_index(name: str, config: LayerDecayConfig) -> int | None:
    pattern = re.compile(config.regex)
    match = pattern.search(name)
    if match is None:
        return None

    block_idx = int(match.group(1))
    if not config.split_blocks:
        return block_idx + 1

    suffix = name[match.end() :]
    first_block_layer = 2 * block_idx + 1
    if suffix.startswith(("norm1", "attn", "ls1")):
        return first_block_layer
    if suffix.startswith(("norm2", "mlp", "ffn", "ls2")):
        return first_block_layer + 1

    raise ValueError(f"Could not assign block parameter to a split layer: {name}")


def _infer_max_layer_decay_index(
    matched_params: list[tuple[str, nn.Parameter]], config: LayerDecayConfig
) -> int:
    block_layer_indices = [
        layer_idx
        for name, _ in matched_params
        if (layer_idx := _get_block_layer_decay_index(name, config)) is not None
    ]
    if block_layer_indices:
        return max(block_layer_indices)

    return config.default_layer


def _get_layer_decay_index(name: str, config: LayerDecayConfig) -> int:
    block_layer_idx = _get_block_layer_decay_index(name, config)
    if block_layer_idx is not None:
        return block_layer_idx

    return config.default_layer


def create_param_groups(
    model: nn.Module,
    param_group_configs: list[ParamGroupConfig],
    summary: bool = False,
) -> list[dict]:
    """
    Create optimizer parameter groups from configs.

    Args:
        model (nn.Module): Model containing parameters
        param_group_configs (list[ParamGroupConfig]): List of ParamGroupConfig objects
        summary (bool): Whether to print a summary of parameter groups

    Returns:
        list[dict]: List of parameter group dicts for optimizer
    """

    # Start with all parameters
    remaining_params = {name: param for name, param in model.named_parameters()}
    param_groups = []

    for i, config in enumerate(param_group_configs):
        matched_params = []
        pattern = re.compile(config.regex)

        # Find matching parameters
        names_to_remove = []
        for name, param in remaining_params.items():
            if pattern.match(name):
                matched_params.append((name, param))
                names_to_remove.append(name)

        # Remove matched parameters from remaining stack
        for name in names_to_remove:
            del remaining_params[name]

        # If no parameters matched, skip this group
        if not matched_params:
            if summary:
                print(f"Group {i + 1} - No parameters matched regex: {config.regex}")
            continue

        # Frozen group
        if config.freeze:
            if summary:
                print(f"Group {i + 1} - Freezing parameters:")
                for name in names_to_remove:
                    print(f"  - {name}")
            continue

        if config.layer_decay is None:
            if summary:
                print(f"Group {i + 1} - LR: {config.lr} WD: {config.weight_decay}")
                for name in names_to_remove:
                    print(f"  - {name}")

            param_group = {
                "params": [param for _, param in matched_params],
                "lr": config.lr,
                "weight_decay": config.weight_decay,
            }
            param_groups.append(param_group)
            continue

        layer_decay = config.layer_decay
        if layer_decay.rate <= 0:
            raise ValueError("layer_decay.rate must be greater than 0")

        max_layer_idx = _infer_max_layer_decay_index(matched_params, layer_decay)

        params_by_layer = defaultdict(list)
        names_by_layer = defaultdict(list)
        for name, param in matched_params:
            layer_idx = _get_layer_decay_index(name, layer_decay)
            params_by_layer[layer_idx].append(param)
            names_by_layer[layer_idx].append(name)

        for layer_idx in sorted(params_by_layer):
            if layer_idx > max_layer_idx:
                raise ValueError(
                    f"Layer index {layer_idx} exceeds inferred max layer "
                    f"{max_layer_idx} for regex: {config.regex}"
                )

            lr_scale = layer_decay.rate ** (max_layer_idx - layer_idx)
            lr = config.lr * lr_scale

            if summary:
                print(
                    f"Group {i + 1} layer {layer_idx} - LR: {lr} "
                    f"WD: {config.weight_decay} scale: {lr_scale}"
                )
                for name in names_by_layer[layer_idx]:
                    print(f"  - {name}")

            param_groups.append(
                {
                    "params": params_by_layer[layer_idx],
                    "lr": lr,
                    "weight_decay": config.weight_decay,
                }
            )

    # Raise Error if there are unmatched trainable parameters
    unmatched = {name: p for name, p in remaining_params.items() if p.requires_grad}
    if unmatched:
        msg = f"{len(unmatched)} trainable parameters not matched by any config:"
        for name in list(unmatched.keys()):
            msg += f"\n  - {name}"
        raise ValueError(msg)

    return param_groups


def freeze_param_groups(
    model: nn.Module,
    param_group_configs: list[ParamGroupConfig],
):
    """
    Freeze parameters in the model based on param group configs.

    Args:
        model (nn.Module): Model containing parameters
        param_group_configs (list[ParamGroupConfig]): List of ParamGroupConfig objects
    """
    remaining_params = {name: param for name, param in model.named_parameters()}

    for config in param_group_configs:
        pattern = re.compile(config.regex)

        matched_names: list[str] = []
        for name, param in remaining_params.items():
            if pattern.match(name):
                matched_names.append(name)
                if config.freeze:
                    param.requires_grad = False

        for name in matched_names:
            del remaining_params[name]

        if config.freeze and not matched_names:
            raise ValueError(f"No parameters matched regex: {config.regex}")
