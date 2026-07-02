#!/usr/bin/env python3
"""
Silhouette re-projection consistency analysis.

Compares the rendered mask (mesh reprojected into a camera) against the
original input silhouette mask for a given camera and frame.

Usage:
    python3 scripts/silhouette_iou.py
    python3 scripts/silhouette_iou.py --frame 5 --cam-id 0

The rendered masks are written by the exporter to:
    runtime/outputs/<run>/frame_XXXXX/mask/mask_<id>.png

The input masks are stored in the dataset as raw uint8 binary files:
    data/<dataset>/frame_XXXXX/mask/mask_<cam_name>.bin
"""

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image


BASE = Path(__file__).resolve().parents[1]

RUNS = {
    "none": BASE / "runtime/outputs/vci_classical_dataset_runtime_depth_fusion_mode_none_no_depth",
    "real": BASE / "runtime/outputs/vci_classical_dataset_runtime_depth_fusion_mode_real_data",
}

DATASET = BASE / "data/2026_03_20_orbbec_002_standard"


def load_input_mask(frame: int, cam_name: str, flip: bool = True) -> np.ndarray:
    path = DATASET / f"frame_{frame:05d}" / "mask" / f"mask_{cam_name}.bin"
    raw = np.fromfile(path, dtype=np.uint8)

    n = raw.size
    if n == 5328 * 4608:
        mask = raw.reshape(5328, 4608)
    elif n == 1080 * 1920:
        mask = raw.reshape(1080, 1920)
    else:
        raise ValueError(f"Unexpected mask size {n} for {path}")

    if flip:
        mask = np.flipud(mask)
    return mask > 127


def load_rendered_mask(run_dir: Path, frame: int, cam_id: int) -> np.ndarray:
    path = run_dir / f"frame_{frame:05d}" / "mask" / f"mask_{cam_id}.png"
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr > 127


def iou_stats(gt: np.ndarray, pred: np.ndarray) -> dict:
    tp = int((gt & pred).sum())
    fp = int((~gt & pred).sum())
    fn = int((gt & ~pred).sum())
    union = tp + fp + fn
    return {
        "iou":       tp / union if union > 0 else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "recall":    tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
        "rendered_px": int(pred.sum()),
    }


def get_cam_name_for_id(cam_id: int, run_dir: Path) -> str:
    debug = run_dir / "camera_debug.json"
    with open(debug) as f:
        d = json.load(f)
    for cam in d["cameras"]:
        if cam["id"] == cam_id:
            return cam["name"]
    raise ValueError(f"Camera id {cam_id} not found in {debug}")


def main():
    parser = argparse.ArgumentParser(description="Silhouette reprojection IoU analysis")
    parser.add_argument("--frame",  type=int, default=0, help="Frame index (default: 0)")
    parser.add_argument("--cam-id", type=int, default=0, help="Camera integer ID (default: 0 = C0000)")
    args = parser.parse_args()

    frame  = args.frame
    cam_id = args.cam_id

    # Resolve camera name from the first available run's debug JSON
    first_run = next(iter(RUNS.values()))
    cam_name = get_cam_name_for_id(cam_id, first_run)

    input_mask = load_input_mask(frame, cam_name)
    print(f"Camera:      {cam_name}  (id={cam_id})")
    print(f"Frame:       {frame}")
    print(f"Resolution:  {input_mask.shape[1]} x {input_mask.shape[0]}")
    print(f"Input mask:  {input_mask.sum():,} px  ({100 * input_mask.mean():.2f}%)")
    print()
    print(f"{'Run':>6} | {'IoU':>6} | {'Precision':>9} | {'Recall':>6} | {'TP':>8} | {'FP':>8} | {'FN':>8} | {'Rendered px':>11}")
    print("-" * 75)

    for name, run_dir in RUNS.items():
        rendered = load_rendered_mask(run_dir, frame, cam_id)
        stats = iou_stats(input_mask, rendered)
        print(
            f"{name:>6} | {stats['iou']:>6.4f} | {stats['precision']:>9.4f} | "
            f"{stats['recall']:>6.4f} | {stats['tp']:>8,} | {stats['fp']:>8,} | "
            f"{stats['fn']:>8,} | {stats['rendered_px']:>11,}"
        )


if __name__ == "__main__":
    main()
