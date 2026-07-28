# ------------------------------------------------------------------------
# MixMatchDet
# Copyright (c) 2026 Hochschule Ruhr West
# Licensed under the Apache License, Version 2.0 [See LICENSE for details]
# ------------------------------------------------------------------------

import argparse

import torch
from torch.utils.data import DataLoader
from lightning import seed_everything
from tqdm import tqdm

from src.data.coco import (
    COCOTransforms,
    CocoDetection,
    collate_fn_list,
    collate_fn_nested,
)
from src.data.transforms import FixedSizeTransforms
from src.util.nested_tensor import NestedTensor


def _get_transforms(image_size: str, patch_size: int, seed: int):
    if image_size == "list":
        collate_fn = collate_fn_list
        transforms = COCOTransforms(train=True, patch_size=patch_size, seed=seed)
    elif image_size == "padded":
        collate_fn = collate_fn_nested
        transforms = COCOTransforms(train=True, patch_size=patch_size, seed=seed)
    else:
        size = (int(image_size), int(image_size))
        transforms = FixedSizeTransforms(
            train=True,
            image_size=size,
            patch_size=patch_size,
            seed=seed,
        )
        collate_fn = collate_fn_nested
    return transforms, collate_fn


def _batch_hw(samples: list[torch.Tensor] | NestedTensor) -> list[tuple[int, int]]:
    if isinstance(samples, NestedTensor):
        return list(samples.shapes)
    return [(int(s.shape[-2]), int(s.shape[-1])) for s in samples]


def _count_stats_for_batch(
    hws: list[tuple[int, int]],
    patch_size: int,
    num_register_tokens: int,
    num_det_tokens: int,
) -> dict[str, float]:
    B = len(hws)
    hs = [h for h, _ in hws]
    ws = [w for _, w in hws]
    H = max(hs)
    W = max(ws)

    total_pixels = float(B * H * W)
    real_pixels = float(sum(h * w for h, w in hws))
    pad_pixels = total_pixels - real_pixels

    grid_H = H // patch_size
    grid_W = W // patch_size
    total_patch_tokens = float(B * grid_H * grid_W)
    real_patch_tokens = float(
        sum((h // patch_size) * (w // patch_size) for h, w in hws)
    )
    pad_patch_tokens = total_patch_tokens - real_patch_tokens

    # Sequence lengths used by the ViT blocks (cls + registers + patches + [DET] tokens)
    per_img_overhead = 1 + num_register_tokens + num_det_tokens
    total_seq_tokens = float(B * per_img_overhead + total_patch_tokens)
    real_seq_tokens = float(B * per_img_overhead + real_patch_tokens)
    pad_seq_tokens = total_seq_tokens - real_seq_tokens

    return {
        "B": float(B),
        "total_pixels": total_pixels,
        "pad_pixels": pad_pixels,
        "total_patch_tokens": total_patch_tokens,
        "real_patch_tokens": real_patch_tokens,
        "pad_patch_tokens": pad_patch_tokens,
        "total_seq_tokens": total_seq_tokens,
        "real_seq_tokens": real_seq_tokens,
        "pad_seq_tokens": pad_seq_tokens,
    }


def count_padding_tokens(args) -> None:
    seed_everything(args.seed, workers=True)

    transforms, collate_fn = _get_transforms(
        args.image_size, patch_size=args.patch_size, seed=args.seed
    )
    dataset = CocoDetection(
        args.image_dir,
        args.ann_file,
        transforms=transforms,
    )

    data_loader = DataLoader(
        dataset,
        args.batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        shuffle=True,
    )

    sum_pad_pixels_pct = 0.0
    sum_pad_patch_pct = 0.0
    sum_pad_seq_pct = 0.0

    sum_total_patch_tokens = 0.0
    sum_real_patch_tokens = 0.0
    sum_total_seq_tokens = 0.0
    sum_real_seq_tokens = 0.0

    sum_packed_savings_pct = 0.0
    batches = 0

    for samples, _ in tqdm(data_loader, desc=f"Counting ({args.image_size})"):
        batches += 1

        hws = _batch_hw(samples)
        stats = _count_stats_for_batch(
            hws,
            patch_size=args.patch_size,
            num_register_tokens=args.num_register_tokens,
            num_det_tokens=args.num_det_tokens,
        )

        total_pixels = stats["total_pixels"]
        sum_pad_pixels_pct += (stats["pad_pixels"] / total_pixels) * 100.0

        total_patch = stats["total_patch_tokens"]
        sum_pad_patch_pct += (stats["pad_patch_tokens"] / total_patch) * 100.0
        sum_total_patch_tokens += total_patch
        sum_real_patch_tokens += stats["real_patch_tokens"]

        total_seq = stats["total_seq_tokens"]
        sum_pad_seq_pct += (stats["pad_seq_tokens"] / total_seq) * 100.0
        sum_total_seq_tokens += total_seq
        sum_real_seq_tokens += stats["real_seq_tokens"]

        # For packed/list mode, report the within-batch savings vs padding-to-max.
        if args.image_size == "list":
            # In list mode, the model uses packed sequences with real_patch_tokens.
            # The above `total_patch_tokens` corresponds to "pad to max in batch".
            sum_packed_savings_pct += (
                1.0 - (stats["real_patch_tokens"] / total_patch)
            ) * 100.0

    print(f"Image size mode: {args.image_size}")
    print(f"Batches: {batches}")

    avg_total_patch = sum_total_patch_tokens / batches
    avg_real_patch = sum_real_patch_tokens / batches
    avg_total_seq = sum_total_seq_tokens / batches
    avg_real_seq = sum_real_seq_tokens / batches

    if args.image_size == "list":
        print(f"Average packed patch tokens / batch: {avg_real_patch:.1f}")
        print(f"Average packed seq tokens / batch: {avg_real_seq:.1f}")
        print(f"Equivalent pad-to-max patch tokens / batch: {avg_total_patch:.1f}")
        print(f"Equivalent pad-to-max seq tokens / batch: {avg_total_seq:.1f}")
        print(
            f"Average savings vs pad-to-max (patch tokens): {sum_packed_savings_pct / batches:.2f}%"
        )
    else:
        print(f"Average pixel padding: {sum_pad_pixels_pct / batches:.2f}%")
        print(f"Average patch-token padding: {sum_pad_patch_pct / batches:.2f}%")
        print(
            f"Average seq-token padding (incl cls/reg/[DET]): {sum_pad_seq_pct / batches:.2f}%"
        )
        print(f"Average patch tokens / batch (padded): {avg_total_patch:.1f}")
        print(f"Average patch tokens / batch (real): {avg_real_patch:.1f}")
        print(f"Average seq tokens / batch (padded): {avg_total_seq:.1f}")
        print(f"Average seq tokens / batch (real): {avg_real_seq:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count padding tokens in COCO dataset")

    parser.add_argument(
        "--image-dir",
        type=str,
        default="coco/train2017",
        help="Path to COCO images directory",
    )

    parser.add_argument(
        "--ann-file",
        type=str,
        default="coco/annotations/instances_train2017.json",
        help="Path to COCO annotations json",
    )

    parser.add_argument(
        "--image-size",
        type=str,
        default="padded",
        help='Batching mode: "padded" (NestedTensor), "list" (packed), or a fixed integer size like "672".',
    )

    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for the dataloader"
    )

    parser.add_argument(
        "--patch-size", type=int, default=14, help="Patch size for the transforms"
    )

    parser.add_argument(
        "--num-register-tokens",
        type=int,
        default=0,
        help="Number of register tokens per image (for seq token counts)",
    )

    parser.add_argument(
        "--num-det-tokens",
        type=int,
        default=0,
        help="Number of [DET] tokens per image (for seq token counts)",
    )

    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of workers for data loading"
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    count_padding_tokens(args)
