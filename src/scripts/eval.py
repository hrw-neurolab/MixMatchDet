# ------------------------------------------------------------------------
# MixMatchDet
# Copyright (c) 2026 Hochschule Ruhr West
# Licensed under the Apache License, Version 2.0 [See LICENSE for details]
# ------------------------------------------------------------------------

import contextlib
import time
import datetime
import argparse
import os

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode
from lightning import seed_everything
from tqdm import tqdm

from src.data.coco import (
    COCOTransforms,
    CocoDetection,
    collate_fn_list,
    collate_fn_nested,
)
from src.data.transforms import FixedSizeTransforms
from src.models.detr.coco_eval import CocoEvaluator
from src.models.detr.postprocess import PostProcessor
from src.lit_mixmatchdet import LitMixMatchDet


def get_transforms(args, patch_size):
    if args.image_size == "list":
        collate_fn = collate_fn_list
        transforms = COCOTransforms(train=False, patch_size=patch_size, seed=args.seed)
    elif args.image_size == "padded":
        collate_fn = collate_fn_nested
        transforms = COCOTransforms(train=False, patch_size=patch_size, seed=args.seed)
    else:
        image_size = (int(args.image_size), int(args.image_size))
        transforms = FixedSizeTransforms(
            train=False,
            image_size=image_size,
            patch_size=patch_size,
            seed=args.seed,
        )
        collate_fn = collate_fn_nested

    return transforms, collate_fn


def _to_device(samples, device):
    if isinstance(samples, list):
        return [s.to(device) for s in samples]
    else:
        return samples.to(device)


def setup(args):
    """Common setup: distributed init, model, dataset, dataloader."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    seed_everything(args.seed, workers=True)

    model = LitMixMatchDet.load_from_checkpoint(args.resume, weights_only=False).model
    model.to(device)
    model.eval()

    if local_rank == 0:
        n_parameters = sum(p.numel() for p in model.parameters())
        print("number of params:", n_parameters)

    transforms, collate_fn = get_transforms(args, patch_size=model.backbone.patch_size)

    dataset = CocoDetection(
        "coco/val2017",
        "coco/annotations/instances_val2017.json",
        transforms=transforms,
    )

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    sampler = DistributedSampler(dataset, shuffle=False)

    data_loader = DataLoader(
        dataset,
        args.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.bf16
        else contextlib.nullcontext()
    )

    return local_rank, device, model, dataset, data_loader, autocast_ctx


def benchmark_fps(args):
    """Benchmark FPS (inference-only and inference+postprocess). No eval."""
    local_rank, device, model, dataset, data_loader, autocast_ctx = setup(args)
    postprocessor = PostProcessor()

    # --- Warmup ---
    warmup_samples, _ = next(iter(data_loader))
    warmup_samples = _to_device(warmup_samples, device)
    for _ in range(10):
        with autocast_ctx, torch.no_grad():
            model(warmup_samples)
    torch.cuda.synchronize()

    # --- Benchmark loop ---
    total_images = 0
    total_infer_time = 0.0
    total_infer_post_time = 0.0
    pbar = tqdm(data_loader, desc="Benchmarking FPS", disable=local_rank != 0)
    for samples, targets in pbar:
        samples = _to_device(samples, device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with autocast_ctx, torch.no_grad():
            outputs = model(samples)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        postprocessor(outputs, targets)
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        total_infer_time += t1 - t0
        total_infer_post_time += t2 - t0
        total_images += len(samples)

        pbar.set_postfix(
            infer=f"{total_images / total_infer_time:.1f}",
            infer_post=f"{total_images / total_infer_post_time:.1f}",
            max_mem=f"{torch.cuda.max_memory_allocated() / 1024.0**2:.0f} MB",
        )

    # Reduce totals across all ranks for aggregate FPS
    stats = torch.tensor([total_images], device=device, dtype=torch.float64)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    all_images = stats.item()
    # Use max time across ranks (wall-clock bottleneck)
    time_stats = torch.tensor(
        [total_infer_time, total_infer_post_time], device=device, dtype=torch.float64
    )
    dist.all_reduce(time_stats, op=dist.ReduceOp.MAX)
    max_infer_time, max_infer_post_time = time_stats.tolist()

    if local_rank == 0:
        world_size = dist.get_world_size()
        print(
            f"Inference FPS:       {all_images / max_infer_time:.1f} img/s  (aggregate, {world_size} GPU(s))"
        )
        print(
            f"Infer + Postprocess: {all_images / max_infer_post_time:.1f} img/s  (aggregate, {world_size} GPU(s))"
        )
        print(f"Total images:        {int(all_images)}")

    dist.barrier()
    dist.destroy_process_group()


def benchmark_flops(args):
    """Compute average FLOPs per image across the full dataset."""
    local_rank, device, model, dataset, data_loader, autocast_ctx = setup(args)

    total_images = 0
    total_iters = 0
    total_flops = 0
    pbar = tqdm(data_loader, desc="Counting FLOPs", disable=local_rank != 0)
    for samples, _ in pbar:
        samples = _to_device(samples, device)
        with FlopCounterMode(display=False) as fc, autocast_ctx, torch.no_grad():
            model(samples)
        total_flops += fc.get_total_flops()
        total_images += len(samples)
        total_iters += 1
        pbar.set_postfix(gflops_iter=f"{total_flops / total_iters / 1e9:.2f}")

    # Reduce across ranks
    stats = torch.tensor(
        [total_flops, total_images, total_iters], device=device, dtype=torch.float64
    )
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    all_flops, all_images, all_iters = stats.tolist()

    if local_rank == 0:
        print(
            f"Avg FLOPs/iter:  {all_flops / all_iters / 1e9:.2f} G  (batch_size={args.batch_size})"
        )
        print(
            f"Avg FLOPs/image: {all_flops / all_images / 1e9:.2f} G  (over {int(all_images)} images)"
        )

    dist.barrier()
    dist.destroy_process_group()


def evaluate(args):
    """Full COCO evaluation with FPS and FLOPs tracking."""
    local_rank, device, model, dataset, data_loader, autocast_ctx = setup(args)
    postprocessor = PostProcessor()
    coco_evaluator = CocoEvaluator(dataset.coco, ("bbox",))

    # --- Evaluation loop  ---
    start_time = time.time()
    pbar = tqdm(data_loader, desc="Evaluating", disable=local_rank != 0)
    for samples, targets in pbar:
        samples = [s.to(device) for s in samples]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with autocast_ctx, torch.no_grad():
            outputs = model(samples)

        max_mem = torch.cuda.max_memory_allocated() / 1024.0**2
        pbar.set_postfix(max_mem=f"{max_mem:.0f} MB")

        results, targets = postprocessor(outputs, targets)
        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, results)
        }
        coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()

    if local_rank == 0:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"Time {total_time_str}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluation/Benchmark script")

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", help="resume from checkpoint")
    parser.add_argument("--image-size", default="list")
    parser.add_argument("--batch-size", default=1, type=int, help="batch size per GPU")
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--fps", action="store_true", help="benchmark FPS only (no eval)"
    )
    parser.add_argument(
        "--flops", action="store_true", help="compute avg FLOPs/image over dataset"
    )

    args = parser.parse_args()
    if args.fps:
        benchmark_fps(args)
    elif args.flops:
        benchmark_flops(args)
    else:
        evaluate(args)
