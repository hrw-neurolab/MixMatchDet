# ------------------------------------------------------------------------
# MixMatchDet
# Copyright (c) 2026 Hochschule Ruhr West
# Licensed under the Apache License, Version 2.0 [See LICENSE for details]
# ------------------------------------------------------------------------

import argparse
import hashlib
import random
import re
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.coco import CocoDetection, collate_fn_list, collate_fn_nested
from src.data.transforms import COCOTransforms, FixedSizeTransforms
from src.lit_mixmatchdet import LitMixMatchDet
from src.models.detr.postprocess import PostProcessor


def get_transforms(image_size, patch_size, seed):
    if image_size == "list":
        collate_fn = collate_fn_list
        transforms = COCOTransforms(train=False, patch_size=patch_size, seed=seed)
    elif image_size == "padded":
        collate_fn = collate_fn_nested
        transforms = COCOTransforms(train=False, patch_size=patch_size, seed=seed)
    else:
        image_size = (int(image_size), int(image_size))
        transforms = FixedSizeTransforms(
            train=False, image_size=image_size, patch_size=patch_size, seed=seed
        )
        collate_fn = collate_fn_nested

    return transforms, collate_fn


def _to_device(samples, device):
    if isinstance(samples, (list, tuple)):
        return [s.to(device) for s in samples]
    else:
        return samples.to(device)


def _draw_dashed_rectangle(draw, box, *, outline, width):
    x0, y0, x1, y1 = box
    dash_length = max(2 * width, 6)
    step = 2 * dash_length

    for x in range(round(x0), round(x1), step):
        draw.line((x, y0, min(x + dash_length, x1), y0), fill=outline, width=width)
        draw.line((x, y1, min(x + dash_length, x1), y1), fill=outline, width=width)
    for y in range(round(y0), round(y1), step):
        draw.line((x0, y, x0, min(y + dash_length, y1)), fill=outline, width=width)
        draw.line((x1, y, x1, min(y + dash_length, y1)), fill=outline, width=width)


def _checkpoint_output_dir(out_dir, checkpoint):
    checkpoint = Path(checkpoint).expanduser().resolve()
    run_dir = (
        checkpoint.parent.parent.name
        if checkpoint.parent.name == "checkpoints"
        else checkpoint.parent.name
    )
    label = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{run_dir}_{checkpoint.stem}")
    label = label.strip("._-")[:48]
    checkpoint_id = hashlib.sha256(str(checkpoint).encode()).hexdigest()[:8]
    return out_dir / f"{label}_{checkpoint_id}"


def main(args):
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    module = LitMixMatchDet.load_from_checkpoint(args.checkpoint, weights_only=False)
    model = module.model.to(device).eval()
    data = module.tc.data

    transforms, collate_fn = get_transforms(
        data.image_size, model.backbone.patch_size, args.seed
    )
    dataset = CocoDetection(
        data.image_dir_val, data.ann_file_val, transforms=transforms
    )
    if args.image_id is not None:
        try:
            indices = [dataset.ids.index(args.image_id)]
        except ValueError:
            raise ValueError(f"COCO image id {args.image_id} not found") from None
    else:
        indices = random.sample(range(len(dataset)), min(args.num_images, len(dataset)))
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=1,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_out_dir = _checkpoint_output_dir(out_dir, args.checkpoint)
    checkpoint_out_dir.mkdir(parents=True, exist_ok=True)
    postprocessor = PostProcessor()

    for samples, targets in tqdm(loader):
        samples = _to_device(samples, device)
        target = targets[0]
        with torch.no_grad():
            outputs = model(samples)

        scores, _, boxes = postprocessor.process_outputs(
            outputs["pred_logits"],
            outputs["pred_boxes"],
            target["orig_size"].to(device)[None],
        )

        # Recover the source token for each entry selected by the postprocessor.
        logits = outputs["pred_logits"]
        token_ids = (
            torch.topk(
                logits.sigmoid().flatten(1),
                min(postprocessor.TOPK, logits.shape[1] * logits.shape[2]),
                dim=1,
            ).indices
            // logits.shape[2]
        )
        num_patch_candidates = outputs.get("num_patch_candidates")
        if num_patch_candidates is None:
            num_patch_candidates = logits.shape[1] if model.num_det_tokens == 0 else 0
        num_patch_candidates = int(num_patch_candidates)

        image_id = int(target["image_id"].item())
        info = dataset.coco.loadImgs(image_id)[0]
        image = Image.open(Path(data.image_dir_val) / info["file_name"]).convert("RGB")
        draw = ImageDraw.Draw(image)

        for score, box, token_id in zip(scores[0], boxes[0], token_ids[0]):
            if score < args.score_threshold:
                continue
            box = box.cpu().tolist()
            if token_id < num_patch_candidates:
                draw.rectangle(box, outline="red", width=args.line_width)
            else:
                _draw_dashed_rectangle(draw, box, outline="blue", width=args.line_width)

        output_path = checkpoint_out_dir / info["file_name"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("out_dir")
    parser.add_argument("-n", "--num-images", type=int, default=10)
    parser.add_argument("--image-id", type=int)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--line-width", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
