#!/usr/bin/env python3
"""Dataset showcase, variant: POCKET RESIDUAL. For each depth camera we analytically render the depth"""
import json, math
from pathlib import Path
import cv2, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "data/_dimpled_d30"
OUT = ROOT / "output/thesis_evaluation/figures/05b_dataset_pocket_residual.png"
CAM_IDS = ["D000", "D001", "D002", "D003", "D004", "D005", "D006", "D007"]
NCOL = 4
BG_M = 4.9
R = 0.5                      # base sphere radius (must match generator)
FX = FY = 1400.0
W, H = 1920, 1080
HALF = 430
RES_MAX = 0.10              # colour range of the residual (m); pocket depth ~0.09

cal = {c["camera_id"]: c for c in json.loads((DS / "calibration_dome.json").read_text())["cameras"]}
cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
ray_cam = np.stack([(u - cx) / FX, (v - cy) / FY, np.ones_like(u)], 2)
ray_cam /= np.linalg.norm(ray_cam, axis=2, keepdims=True)


def az_el(pos):
    x, y, z = pos
    r = math.sqrt(x * x + y * y + z * z)
    return math.degrees(math.atan2(x, z)), math.degrees(math.asin(y / r))


def smooth_sphere_depth(cid):
    """Analytic depth of the plain sphere for this camera (same math as the generator)."""
    view = np.array(cal[cid]["extrinsics"]["view_matrix"], np.float32).reshape(4, 4)
    Rm = view[:3, :3]
    pos = np.array(cal[cid]["extrinsics"]["position"], np.float32)
    ray_w = np.einsum("hwc,cd->hwd", ray_cam, Rm)        # cam->world ray dirs
    o = pos.reshape(1, 1, 3)
    b = (ray_w * o).sum(2)
    c = float((pos * pos).sum()) - R * R
    disc = b * b - c
    depth = np.full((H, W), np.nan, np.float32)
    hit = disc >= 0.0
    t_near = -b[hit] - np.sqrt(disc[hit])
    depth[hit] = t_near * ray_cam[..., 2][hit]
    return depth


nrow = (len(CAM_IDS) + NCOL - 1) // NCOL
fig, axes = plt.subplots(nrow, NCOL, figsize=(NCOL * 3.0, nrow * 3.05))
axes = np.atleast_2d(axes)
im = None
for k, cid in enumerate(CAM_IDS):
    ax = axes[k // NCOL][k % NCOL]
    d = cv2.imread(str(DS / f"frame_00000/depth/{cid}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    ref = smooth_sphere_depth(cid)
    resid = d - ref                                       # pockets are deeper -> positive
    obj = (d < BG_M) & np.isfinite(ref)
    resid = np.ma.masked_where(~obj, resid)
    cyi, cxi = H // 2, W // 2
    crop = resid[cyi - HALF:cyi + HALF, cxi - HALF:cxi + HALF]
    im = ax.imshow(crop, cmap="RdBu_r", vmin=-RES_MAX, vmax=RES_MAX)
    ax.set_facecolor("white")
    az, el = az_el(cal[cid]["extrinsics"]["position"])
    ax.set_title(f"{cid}   az {az:+.0f}°, el {el:+.0f}°", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
for k in range(len(CAM_IDS), nrow * NCOL):
    axes[k // NCOL][k % NCOL].axis("off")

cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cbar.set_label("depth residual vs. smooth sphere (m)")
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("wrote", OUT)
