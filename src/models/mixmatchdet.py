# ------------------------------------------------------------------------
# MixMatchDet
# Copyright (c) 2026 Hochschule Ruhr West
# Licensed under the Apache License, Version 2.0 [See LICENSE for details]
# ------------------------------------------------------------------------

from src.config.mixmatchdet import MixMatchDetConfig
from src.models.variants.base import MixMatchDetBase


def mixmatchdet(config: MixMatchDetConfig, num_classes: int) -> MixMatchDetBase:
    """
    Factory function to create the appropriate MixMatchDet variant.

    Args:
        config: Configuration object specifying the model variant and parameters
        num_classes: Number of object classes to detect

    Returns:
        An instance of the appropriate MixMatchDet variant

    Raises:
        ValueError: If the configuration is invalid or inconsistent
    """
    has_det_tokens = config.det_tokens is not None
    has_patch_candidates = config.patch_candidates is not None

    if has_det_tokens and not has_patch_candidates:
        from src.models.variants.det_tokens import MixMatchDet

    elif not has_det_tokens and has_patch_candidates:
        from src.models.variants.patch_candidates import MixMatchDet

    elif has_det_tokens and has_patch_candidates:
        from src.models.variants.mixed_candidates import MixMatchDet

    else:
        raise ValueError("Unknown MixMatchDet variant")

    return MixMatchDet(config, num_classes)
