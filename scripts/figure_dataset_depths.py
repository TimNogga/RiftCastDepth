#!/usr/bin/env python3
"""Thesis 'dataset showcase' figure: ray-marched depth maps of the dimpled sphere from several
camera angles, in one grid. Pockets appear as the deeper (farther) patches on the sphere face.
Reads data/_dimpled_d30 (60 RGB + 30 depth). Output -> thesis_evaluation/figures/05_dataset_depth_views.png
"""
import json, math
from pathlib import Path
import cv2, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "data/_dimpled_d30"
OUT = ROOT / "output/thesis_evaluation/figures/05_dataset_depth_views.png"
CAM_IDS = ["D000", "D001", "D002", "D003", "D004", "D005", "D006", "D007"]  # nested FPS, maximally spread
NCOL = 4
BG_M = 4.9          # background threshold (object < this)
VMIN, VMAX = 2.02, 2.52   # metric depth colour range (m)
HALF = 430          # crop half-window around image centre (px)

cal = {c["camera_id"]: c for c in json.loads((DS / "calibration_dome.json").read_text())["cameras"]}


def az_el(pos):
    x, y, z = pos
    r = math.sqrt(x * x + y * y + z * z)
    return math.degrees(math.atan2(x, z)), math.degrees(math.asin(y / r))


nrow = (len(CAM_IDS) + NCOL - 1) // NCOL
fig, axes = plt.subplots(nrow, NCOL, figsize=(NCOL * 3.0, nrow * 3.05))
axes = np.atleast_2d(axes)
im = None
for k, cid in enumerate(CAM_IDS):
    ax = axes[k // NCOL][k % NCOL]
    d = cv2.imread(str(DS / f"frame_00000/depth/{cid}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    H, W = d.shape
    cy, cx = H // 2, W // 2
    crop = d[cy - HALF:cy + HALF, cx - HALF:cx + HALF]
    obj = np.ma.masked_where(crop >= BG_M, crop)          # hide background
    im = ax.imshow(obj, cmap="turbo", vmin=VMIN, vmax=VMAX)
    ax.set_facecolor("white")
    az, el = az_el(cal[cid]["extrinsics"]["position"])
    ax.set_title(f"{cid}   az {az:+.0f}°, el {el:+.0f}°", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
for k in range(len(CAM_IDS), nrow * NCOL):
    axes[k // NCOL][k % NCOL].axis("off")

cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cbar.set_label("metric depth (m)")
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("wrote", OUT)
