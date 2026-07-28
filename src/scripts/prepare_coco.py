# ------------------------------------------------------------------------
# MixMatchDet
# Copyright (c) 2026 Hochschule Ruhr West
# Licensed under the Apache License, Version 2.0 [See LICENSE for details]
# ------------------------------------------------------------------------

import zipfile
import sys
from pathlib import Path

import urllib.request


def download_coco_dataset(target_dir):
    """
    Download and extract the official COCO dataset.

    Args:
        target_dir (str): Path where the dataset will be saved
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # COCO dataset URLs
    datasets = {
        "train2017": "http://images.cocodataset.org/zips/train2017.zip",
        "val2017": "http://images.cocodataset.org/zips/val2017.zip",
        "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    }

    for name, url in datasets.items():
        zip_path = target_dir / f"{name}.zip"

        # Download
        if not zip_path.exists():
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(url, zip_path)
            print(f"Downloaded {name}")

        # Extract
        print(f"Extracting {name}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)
        print(f"Extracted {name}")

        # Remove zip to save space
        zip_path.unlink()
        print(f"Removed {name}.zip\n")

    print("COCO dataset download and extraction complete!")


if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "./coco"
    download_coco_dataset(target_dir)
