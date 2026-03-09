#!/usr/bin/env python3
"""Convert all PNG images in the dataset's rgb/ folders to JPEG for nvjpeg compatibility."""

import os
import sys
from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

DATASET = Path("/home/timnogga/bachelorthesis/biwi_kinect_head_pose/RIFTCast/2026_02_03_orbbec_test")
QUALITY = 95


def convert_one(png_path: Path):
    jpg_path = png_path.with_suffix(".jpg")
    if jpg_path.exists():
        return f"skip {jpg_path.name}"
    img = Image.open(png_path)
    img.save(jpg_path, "JPEG", quality=QUALITY)
    return f"done {jpg_path.name}"


def main():
    pngs = sorted(DATASET.glob("frame_*/rgb/*.png"))
    print(f"Found {len(pngs)} PNG files to convert")
    
    with ProcessPoolExecutor(max_workers=8) as pool:
        for i, result in enumerate(pool.map(convert_one, pngs)):
            if (i + 1) % 20 == 0 or i + 1 == len(pngs):
                print(f"  [{i+1}/{len(pngs)}] {result}")
    
    print("Done.")


if __name__ == "__main__":
    main()
