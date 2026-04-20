#!/usr/bin/env python3
"""Convert all PNG images in the dataset's rgb/ folders to JPEG for nvjpeg compatibility."""

import argparse
from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DATASET = REPO_ROOT / "data" / "2026_02_03_orbbec_test"
DEFAULT_QUALITY = 95


def convert_one(args):
    png_path, quality = args
    jpg_path = png_path.with_suffix(".jpg")
    if jpg_path.exists():
        return f"skip {jpg_path.name}"
    img = Image.open(png_path)
    img.save(jpg_path, "JPEG", quality=quality)
    return f"done {jpg_path.name}"


def main():
    parser = argparse.ArgumentParser(description="Convert frame_*/rgb/*.png files to .jpg")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Dataset root (default: {DEFAULT_DATASET})",
    )
    parser.add_argument("--quality", type=int, default=DEFAULT_QUALITY, help="JPEG quality (default: 95)")
    parser.add_argument("--workers", type=int, default=8, help="Process workers (default: 8)")
    args = parser.parse_args()

    pngs = sorted(args.dataset.glob("frame_*/rgb/*.png"))
    print(f"Found {len(pngs)} PNG files to convert")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = ((png, args.quality) for png in pngs)
        for i, result in enumerate(pool.map(convert_one, jobs)):
            if (i + 1) % 20 == 0 or i + 1 == len(pngs):
                print(f"  [{i+1}/{len(pngs)}] {result}")

    print("Done.")


if __name__ == "__main__":
    main()
