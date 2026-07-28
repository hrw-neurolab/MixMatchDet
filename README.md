<!--
MixMatchDet
Copyright (c) 2026 Hochschule Ruhr West
Licensed under the Apache License, Version 2.0 [See LICENSE for details]
-->

<h1 align="center">MixMatchDet: Mixing and Matching Tokens for Plain ViT Detectors</h1>

<p align="center">
    <a href="https://scholar.google.com/citations?user=VmjTjOUAAAAJ">
        D. Rohrschneider
    </a>
    *, 
    <a href="https://scholar.google.com/citations?user=cEBWyEQAAAAJ">
        A. Haselhoff
    </a>
    , 
    <a href="https://scholar.google.com/citations?user=kwlsSk4AAAAJ">
        U. Handmann
    </a>
</p>

<p align="center">
    <a href="https://github.com/hrw-neurolab/MixMatchDet/blob/master/LICENSE">
        <img alt="license" src="https://img.shields.io/badge/LICENSE-Apache%202.0-blue">
    </a>
    <!-- <a href="https://arxiv.org/abs/...">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-...-red">
    </a> -->
    <a href="mailto:david.rohrschneider@hs-ruhrwest.de">
        <img alt="Contact Us" src="https://img.shields.io/badge/Contact-Email-black">
    </a>
</p>

This is the official codebase for our paper `MixMatchDet: Mixing and Matching Tokens for Plain ViT Detectors` [GCPR 2026].

The repository contains a flexible configuration, training based on torch lightning with optional Weights and Biases logging, COCO evaluation, throughput/FLOP
measurement, and prediction visualization utilities.

## Results
Evaluated on the COCO val2017 benchmark using the official [pycocotools library](https://github.com/ppwwyyxx/cocoapi/tree/master/PythonAPI).

| Backbone | # Epochs | #Params (M) | AP | Config |
|---|---:|---:|---:|---|
| DINOv2-S | 20 | 24.2 | 42.0 | [config](configs/3-mixed-candidates/dinov2s_mixed.yml) |
| DINOv2-S | 36 | 24.2 | 43.5 | [config](configs/4-scaling/dinov2s_mixed_36ep.yml) |
| DINOv2-S | 50 | 24.2 | 44.1 | [config](configs/4-scaling/dinov2s_mixed_50ep.yml) |
| DINOv2-B | 20 | 93.1 | 47.5 | [config](configs/4-scaling/dinov2b_mixed.yml) |
| DINOv2-L | 20 | 314 | 51.1 | [config](configs/4-scaling/dinov2l_mixed.yml) |
||||||
| DINOv3-S | 20 | 22.5 | 40.0 | - |
| DINOv3-S | 36 | 22.5 | 42.8 | - |
| DINOv3-S | 50 | 22.5 | 44.6 | - |
| DINOv3-B | 20 | 88.9 | 51.2 | - |
| DINOv3-L | 20 | 308 | **55.2** | - |

## Reproduction
> [!IMPORTANT]
> We currently do not provide open source code for DINOv3 derived components due to its special license.
> For now, please contact us for requests regarding these components.

<details>
<summary><b>Installation</b></summary>
The code targets Python 3.12 and CUDA-capable PyTorch.
We highly recommend verifying the local CUDA version via `nvidia-smi` to match the installed torch and xformers kernels to your machine.

Examples for CUDA 13.0:
```bash
# uv
uv venv --python 3.12

uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130

# pip (ensure python is at 3.12)
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
```
</details>

<details>
<summary><b>COCO Dataset</b></summary>
COCO 2017 can be downloaded and extracted with:

```bash
# uv
uv run src/scripts/prepare_coco coco

# local python
python -m src.scripts.prepare_coco coco
```

This creates the conventional structure:

```text
coco/
  annotations/
    instances_train2017.json
    instances_val2017.json
  train2017/
  val2017/
```

In case your COCO path differs from the default path set up above, update the four dataset paths under `training.data` in `configs/base.yml` as follows:

```yaml
training:
  data:
    image_dir_train: "path/to/train2017"
    ann_file_train: "path/to/annotations/instances_train2017.json"
    image_dir_val: "path/to/val2017"
    ann_file_val: "path/to/annotations/instances_val2017.json"
```
</details>

<details>
<summary><b>Backbone Weights</b></summary>
DINOv2 and YOLOS (DeiT) configurations reference publicly available pretrained weights, which are downloaded and cached automatically upon their first invocation:

```yaml
# DINOv2, ViT-S, Patch 14, 4 Register Tokens 
model:
  vit:
    weights: "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth"

# DINOv2, ViT-B, Patch 14, 4 Register Tokens 
model:
  vit:
    weights: "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth"

# DINOv2, ViT-L, Patch 14, 4 Register Tokens 
model:
  vit:
    weights: "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth"

# DeiT, ViT-S, Patch 16
model:
  vit:
    weights: "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth"
```
</details>

<details>
<summary><b>Configurations</b></summary>
Configurations use YAML inheritance through the `extends` field.

- `configs/1-baseline/`: learned `[DET]` token baselines
- `configs/2-pad-vs-pack/`: padded, packed, and fixed-size batching
- `configs/3-mixed-candidates/`: patch-only and mixed-candidate models
- `configs/4-scaling/`: epoch scaling and larger DINOv2 backbones

The main candidate settings are:

```yaml
model:
  det_tokens:
    num: 150
    pos: true

  patch_candidates:
    topk: 150
    target_strides: [14]
```

Set either section to `null` to disable that candidate type. At least one candidate type must be enabled.

Scale-Aware Matching can be enabled through:

```yaml
training:
  criterion:
      matcher:
        scale_aware_matching:
          # This is the best config for dinov2 -- restricting extremes, mixing medium
          allowed_patch: [true, true, false]
          allowed_det: [false, true, true]
```
</details>

<details>
<summary><b>Training</b></summary>
The default trainer configuration uses two GPUs with DDP. Adjust `trainer_kwargs.devices` and `trainer_kwargs.strategy` in `configs/base.yml` for the available hardware. `trainer.data.{batch_size, num_workers}` are per GPU, and we found a total batch size of 8 to perform best (eg., `batch_size=4` with `devices=2`).

Run a short validation of the complete training pipeline:

```bash
uv run train.py \
  --config configs/3-mixed-candidates/dinov2s_mixed.yml \
  --dry-run
```

Start a normal training run:

```bash
uv run train.py \
  --config configs/3-mixed-candidates/dinov2s_mixed.yml
```

Resume from a Lightning checkpoint:

```bash
uv run train.py \
  --config configs/3-mixed-candidates/dinov2s_mixed.yml \
  --ckpt-path /path/to/last.ckpt \
  --resume-id <<WANDB_RUNID>>
```

Weights & Biases logging is enabled by default. Set `wandb: null` in a configuration to use only the local CSV and Checkpoint loggers.
</details>

<details>
<summary><b>Evaluation</b></summary>
The evaluation command uses distributed launch even for a single GPU and expects COCO validation data under `coco/` (need to change `src/scripts/eval.py` if the COCO path differs).

```bash
uv run torchrun --standalone --nproc_per_node=1 \
  -m src.scripts.eval \
  --resume /path/to/checkpoint.ckpt \
  --image-size list \
  --bf16
```

Use `--fps` to measure throughput without COCO evaluation, or `--flops` to
measure average FLOPs per image:

```bash
uv run torchrun --standalone --nproc_per_node=1 \
  -m src.scripts.eval \
  --resume /path/to/checkpoint.ckpt \
  --image-size list \
  --bf16 \
  --fps
```

For multi-GPU evaluation, increase `--nproc_per_node`.
</details>

<details>
<summary><b>Visualizing Predictions</b></summary>

```bash
uv run python -m src.scripts.visualize_predictions \
  /path/to/checkpoint.ckpt \
  visualizations/
```

Useful options include `--num-images`, `--image-id`, and
`--score-threshold`. Patch-candidate boxes are drawn in red and `[DET]` candidate boxes in blue with dashed border.
</details>

## Citation

If you use this repository, please cite the accompanying MixMatchDet paper.
The final BibTeX entry will be added with the publication metadata.

## Acknowledgement

MixMatchDet is build upon these awesome works:

| Name | Links |
|---|---|
| You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection (YOLOS) | [[arXiv]](https://arxiv.org/abs/2106.00666) [[GitHub]](https://github.com/hustvl/YOLOS) |
| DINOv2: Learning Robust Visual Features without Supervision | [[arXiv]](https://arxiv.org/abs/2304.07193) [[GitHub]](https://github.com/facebookresearch/dinov2) |
| DINOv3 | [[arXiv]](https://arxiv.org/abs/2508.10104) [[GitHub]](https://github.com/facebookresearch/dinov3) |
| DETR Doesn’t Need Multi-Scale or Locality Design (Plain-DETR) | [[arXiv]](https://arxiv.org/abs/2308.01904) [[GitHub]](https://github.com/impiga/Plain-DETR) |
| Deformable DETR: Deformable Transformers for End-to-End Object Detection | [[arXiv]](https://arxiv.org/abs/2010.04159) [[GitHub]](https://github.com/fundamentalvision/Deformable-DETR) |
| End-to-End Object Detection with Transformers (DETR) | [[arXiv]](https://arxiv.org/abs/2005.12872) [[GitHub]](https://github.com/facebookresearch/detr) |
| Torch-Warmup-LR | [[GitHub]](https://github.com/lehduong/torch-warmup-lr) |
| PyTorch Lightning | [[GitHub]](https://github.com/Lightning-AI/pytorch-lightning) |
| xFormers: A Modular and Hackable Transformer Modelling Library | [[GitHub]](https://github.com/facebookresearch/xformers) |

## License

MixMatchDet is distributed under the Apache License 2.0.

Original MIT notices for incorporated Plain-DETR, YOLOS, and Torch Warmup LR code are retained in `LICENSES/` and `THIRD_PARTY_NOTICES.md`.

## Funding

This work was supported by the project [Transferhub - Digitalisierung & Circular Economy im Prosperkolleg](https://www.transferhub.nrw) funded by the Ministry of Economic Affairs, Industry, Climate Action and Energy of the State of North Rhine-Westphalia and co-funded by the European Union under the EFRE/JTF Programme North Rhine-Westphalia 2021-2027 (EFRE-20800649).
