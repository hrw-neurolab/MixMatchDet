# ------------------------------------------------------------------------
# MixMatchDet
# Copyright (c) 2026 Hochschule Ruhr West
# Licensed under the Apache License, Version 2.0 [See LICENSE for details]
# ------------------------------------------------------------------------

from abc import ABC, abstractmethod
from functools import cached_property
import torch
from xformers.ops import fmha


class BatchInfo(ABC):
    """A container for storing batch information for variable size batches."""

    num_register_tokens: int
    """Number of register/storage tokens per image."""
    num_det_tokens: int
    """Number of [DET] tokens per image."""
    image_hws: torch.Tensor
    """Heights and widths of the images in the batch."""
    feature_hws: list[tuple[int, int]]
    """Heights and widths of the features for each image in the batch."""
    patch_lens: list[int]
    """Number of patch tokens for each image in the batch."""
    seq_lens: list[int]
    """Sequence lengths for each image in the batch."""
    B: int
    """Batch size."""
    device: torch.device
    """Device where the batch is stored."""
    patch_idx: torch.Tensor
    """Int64 indices of patch tokens within the packed/padded sequence dimension."""
    det_idx: torch.Tensor
    """Int64 indices of [DET] tokens within the packed/padded sequence dimension."""
    attention_mask: torch.Tensor | None
    """Attention mask for ViT self-attention as [B, 1, N, N] where N is total sequence length.
    It has False for padded tokens and True for non-padded tokens. None if no padding."""
    attention_bias: torch.Tensor | fmha.AttentionBias | None
    """Attention bias for ViT self-attention."""

    def __init__(
        self,
        image_hws: list[tuple[int, int]],
        feature_hws: list[tuple[int, int]],
        num_register_tokens: int,
        num_det_tokens: int,
        device: torch.device,
    ) -> None:
        self.num_register_tokens = num_register_tokens
        self.num_det_tokens = num_det_tokens
        self.image_hws = torch.tensor(image_hws, device=device)
        self.feature_hws = feature_hws

        self.patch_lens = [h * w for h, w in feature_hws]
        self.seq_lens = [
            1 + num_register_tokens + l + num_det_tokens for l in self.patch_lens
        ]

        self.B = len(self.seq_lens)
        self.device = device

        self._init_indices()

    @abstractmethod
    def _init_indices(self) -> None:
        """Initialize integer indices for patch and [DET] tokens."""
        pass


class PackedBatch(BatchInfo):
    def _init_indices(self):
        self.image_start_idx = [
            sum(self.seq_lens[:i]) for i in range(len(self.seq_lens))
        ]

        patch_idx: list[int] = []
        det_idx: list[int] = []

        for img_start, patch_len in zip(self.image_start_idx, self.patch_lens):
            patch_start = img_start + 1 + self.num_register_tokens
            patch_end = patch_start + patch_len
            patch_idx.extend(list(range(patch_start, patch_end)))

            if self.num_det_tokens > 0:
                det_end = patch_end + self.num_det_tokens
                det_idx.extend(list(range(patch_end, det_end)))

        dd = {"dtype": torch.int64, "device": self.device}
        self.patch_idx = torch.tensor(patch_idx, **dd)
        self.det_idx = torch.tensor(det_idx, **dd)

    @cached_property
    def attention_bias(self):
        return fmha.BlockDiagonalMask.from_seqlens(self.seq_lens, device=self.device)

    @cached_property
    def attention_mask(self):
        N = sum(self.seq_lens)
        return self.attention_bias.materialize((1, 1, N, N), device=self.device)


class PaddedBatch(BatchInfo):
    padding_mask: torch.Tensor | None
    """Interpolated mask for the patch tokens in the padded tensor as [B, H*W]."""

    def __init__(
        self,
        image_hws: list[tuple[int, int]],
        feature_hws: list[tuple[int, int]],
        num_register_tokens: int,
        num_det_tokens: int,
        padding_mask: torch.Tensor | None,
        device: torch.device,
    ) -> None:
        super().__init__(
            image_hws, feature_hws, num_register_tokens, num_det_tokens, device
        )

        self.padding_mask = padding_mask

    def _init_indices(self) -> None:
        patch_start = 1 + self.num_register_tokens
        patch_end = patch_start + self.patch_lens[0]
        self.patch_idx = torch.arange(patch_start, patch_end, device=self.device)
        if self.num_det_tokens > 0:
            det_end = patch_end + self.num_det_tokens
            self.det_idx = torch.arange(patch_end, det_end, device=self.device)

    @cached_property
    def attention_mask(self) -> torch.Tensor | None:
        """Get the attention mask for ViT self-attention via SDPA.

        Returns:
            A bool tensor of shape [B, 1, 1, total_seq_len] or None if no padding mask.
        """
        if self.padding_mask is None:
            return None

        cls_reg_keep = torch.ones(
            (self.B, 1 + self.num_register_tokens), dtype=torch.bool, device=self.device
        )
        det_keep = torch.ones(
            (self.B, self.num_det_tokens), dtype=torch.bool, device=self.device
        )

        full_keep = torch.cat([cls_reg_keep, self.padding_mask, det_keep], dim=1)

        return full_keep[:, None, None, :]

    @cached_property
    def attention_bias(self) -> torch.Tensor | None:
        """Get the attention bias for ViT self-attention.

        Returns:
            A tensor of shape [B, 1, total_seq_len, total_seq_len] or None if no padding mask.
        """
        if self.padding_mask is None:
            return None

        total_seq_len = self.seq_lens[0]

        dtype = torch.float32
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_gpu_dtype()

        # Pad to multiple of 8 for Cutlass (xformers requirement)
        total_seq_len8 = (total_seq_len + 7) // 8 * 8

        attn_bias = torch.zeros(
            self.B, 1, total_seq_len8, total_seq_len8, dtype=dtype, device=self.device
        )[:, :, :total_seq_len, :total_seq_len]

        # Mask padded *patch keys* for all [DET] tokens.
        # `padding_mask` is expected to be [B, patch_len] with True for padded patches.
        patch_start = 1 + self.num_register_tokens
        patch_end = patch_start + self.patch_lens[0]
        pad_mask_keys = self.padding_mask.view(self.B, 1, 1, -1)

        attn_bias[:, :, :, patch_start:patch_end].masked_fill_(
            ~pad_mask_keys, float("-inf")
        )

        return attn_bias
