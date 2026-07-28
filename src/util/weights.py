# ------------------------------------------------------------------------
# MixMatchDet
# Copyright (c) 2026 Hochschule Ruhr West
# Licensed under the Apache License, Version 2.0 [See LICENSE for details]
# ------------------------------------------------------------------------

import torch


def load_weights_from_url_or_path(url_or_path: str) -> dict:
    """Load a state dict from a URL or local file path.

    Args:
        url_or_path (str): URL or local file path to the weights.

    Returns:
        dict: The loaded state dictionary.
    """
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        state_dict = torch.hub.load_state_dict_from_url(url_or_path, map_location="cpu")
    else:
        state_dict = torch.load(url_or_path, map_location="cpu")
    return dict(state_dict)
