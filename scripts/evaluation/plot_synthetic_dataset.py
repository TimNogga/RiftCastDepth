#!/usr/bin/env python3
"""
Plot of the synthetic open-cube dataset for the thesis.

Four panels:
  A – 3D scene: camera layout + GT open box geometry
  B – silhouette renders seen by 4 colour cameras
  C – depth map seen by D003L (colourised, looking into open face)
  D – GT open box shown in three orthographic views

Run from repo root:
    python3 scripts/evaluation/plot_synthetic_dataset.py
"""

from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET   = REPO_ROOT / "data" / "synthetic_open_cube_8cams" / "frame_00000"
CALIB     = REPO_ROOT / "data" / "synthetic_open_cube_8cams" / "calibration_dome.json"
GT_OBJ    = REPO_ROOT / "scripts" / "ground_truth_open_box.obj"
OUT_DIR   = REPO_ROOT / "output" / "evaluation" / "thesis_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 180, "font.size": 9,
                     "axes.titlesize": 10, "figure.facecolor": "white"})


# ── helpers ────────────────────────────────────────────────────────────────

def load_obj(path: Path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append(list(map(float, line.split()[1:4])))
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
                if len(idx) == 3:
                    faces.append(idx)
                elif len(idx) == 4:
                    faces.append([idx[0], idx[1], idx[2]])
                    faces.append([idx[0], idx[2], idx[3]])
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int64)


def load_depth_pth(path: Path) -> np.ndarray:
    t = torch.load(str(path), map_location="cpu")
    if t.dim() == 3:
        t = t.squeeze(0) if t.shape[0] == 1 else t.squeeze(-1)
    return t.numpy().astype(np.float32) / 1000.0


def colorise_depth(depth_m: np.ndarray, dmin=0.5, dmax=3.5) -> np.ndarray:
    valid = (depth_m > 0.1) & (depth_m < 4.0)
    norm  = np.clip((depth_m - dmin) / (dmax - dmin), 0, 1)
    cmap  = plt.get_cmap("turbo")
    rgb   = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    rgb[~valid] = 20
    return rgb


# ── panel A – 3D scene ─────────────────────────────────────────────────────

def draw_box_3d(ax, bmin, bmax, color, alpha=0.12, edge_color="k", lw=0.6):
    """Draw a solid box as a semi-transparent Poly3DCollection."""
    x0, y0, z0 = bmin
    x1, y1, z1 = bmax
    verts = [
        [[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0]],  # -Z face
        [[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]],  # +Z face
        [[x0,y0,z0],[x0,y0,z1],[x0,y1,z1],[x0,y1,z0]],  # -X
        [[x1,y0,z0],[x1,y0,z1],[x1,y1,z1],[x1,y1,z0]],  # +X
        [[x0,y0,z0],[x1,y0,z0],[x1,y0,z1],[x0,y0,z1]],  # -Y
        [[x0,y1,z0],[x1,y1,z0],[x1,y1,z1],[x0,y1,z1]],  # +Y
    ]
    poly = Poly3DCollection(verts, alpha=alpha, facecolor=color, edgecolor=edge_color, linewidth=lw)
    ax.add_collection3d(poly)


def panel_3d_scene(ax):
    ax.set_proj_type("persp")

    # --- GT open box (5 walls) ---
    walls = [
        ([-0.5,-0.5,-0.5],[-0.42, 0.5, 0.5]),   # left
        ([ 0.42,-0.5,-0.5],[ 0.5, 0.5, 0.5]),   # right
        ([-0.5,-0.5,-0.5],[ 0.5,-0.42, 0.5]),   # bottom
        ([-0.5, 0.42,-0.5],[ 0.5, 0.5, 0.5]),   # top
        ([-0.5,-0.5,-0.5],[ 0.5, 0.5,-0.42]),   # back
    ]
    for bmin, bmax in walls:
        draw_box_3d(ax, bmin, bmax, color="#3498db", alpha=0.18)

    # open-face outline (dashed, +Z side open)
    s = 0.5
    for xs, ys in [([s,s],[-s,s]), ([-s,s],[s,s]), ([-s,-s],[-s,s]), ([-s,s],[-s,-s])]:
        ax.plot(xs, ys, [s,s], "b--", lw=0.8, alpha=0.5)

    # load calibration
    with open(CALIB) as f:
        calib = json.load(f)

    rgb_pos, depth_pos = [], []
    for cam in calib["cameras"]:
        p = cam["extrinsics"]["position"]
        if cam["camera_type"] == "rgb":
            rgb_pos.append(p)
        else:
            depth_pos.append(p)

    rgb_pos   = np.array(rgb_pos)
    depth_pos = np.array(depth_pos)

    # draw gaze lines
    for pos in rgb_pos:
        ax.plot([pos[0], 0], [pos[1], 0], [pos[2], 0],
                color="#e74c3c", lw=0.5, alpha=0.35)
    for pos in depth_pos:
        ax.plot([pos[0], 0], [pos[1], 0], [pos[2], 0],
                color="#e67e22", lw=1.0, alpha=0.7)

    # camera markers
    ax.scatter(rgb_pos[:,0], rgb_pos[:,1], rgb_pos[:,2],
               c="#e74c3c", s=60, marker="^", zorder=5, label="RGB camera")
    ax.scatter(depth_pos[:,0], depth_pos[:,1], depth_pos[:,2],
               c="#e67e22", s=100, marker="D", zorder=5, label="Depth camera")

    # label a few cameras
    for cam in calib["cameras"]:
        p = cam["extrinsics"]["position"]
        ax.text(p[0]*1.08, p[1]*1.08, p[2]*1.08, cam["camera_id"],
                fontsize=6, ha="center", va="center", color="#555555")

    ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2); ax.set_zlim(-3.2, 3.2)
    ax.set_xlabel("X [m]", fontsize=7, labelpad=1)
    ax.set_ylabel("Y [m]", fontsize=7, labelpad=1)
    ax.set_zlabel("Z [m]", fontsize=7, labelpad=1)
    ax.tick_params(labelsize=6)
    ax.set_title("(a) Camera Layout & GT Scene", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left")
    ax.view_init(elev=20, azim=-50)


# ── panel B – silhouette renders ──────────────────────────────────────────

def panel_renders(axes):
    show_cams = ["C000", "C001", "C002", "C003"]
    cam_labels = ["Front (+Z)", "Back (−Z)", "Right (+X)", "Left (−X)"]
    for ax, name, lbl in zip(axes, show_cams, cam_labels):
        p = DATASET / "rgb" / f"{name}.jpg"
        if p.exists():
            img = np.array(Image.open(p).convert("RGB"))
            # rotate 90° CW so the landscape image reads correctly
            img = np.rot90(img, k=3)
            ax.imshow(img, cmap="gray")
        ax.axis("off")
        ax.set_title(f"{name}\n{lbl}", fontsize=8, fontweight="bold")


# ── panel C – depth map ────────────────────────────────────────────────────

def panel_depth(ax):
    pth = DATASET / "rgb" / "D003L.pth"
    if not pth.exists():
        ax.text(0.5, 0.5, "depth not found", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        return
    d   = load_depth_pth(pth)
    rgb = colorise_depth(d)
    ax.imshow(rgb)
    ax.axis("off")
    ax.set_title("(c) Depth Camera D003L\n(looking into open +Z face)",
                 fontsize=10, fontweight="bold")

    # add colourbar manually
    sm = plt.cm.ScalarMappable(cmap="turbo",
                               norm=plt.Normalize(vmin=0.5, vmax=3.5))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Depth [m]", fontsize=7)
    cb.ax.tick_params(labelsize=6)


# ── panel D – GT ortho views ───────────────────────────────────────────────

def panel_gt_ortho(axes):
    gt_v, gt_f = load_obj(GT_OBJ)
    v0, v1, v2 = gt_v[gt_f[:,0]], gt_v[gt_f[:,1]], gt_v[gt_f[:,2]]

    configs = [
        (0, 2, "Front view\n(X–Z plane)", "X [m]", "Z [m]"),
        (0, 1, "Top view\n(X–Y plane)",   "X [m]", "Y [m]"),
        (2, 1, "Side view\n(Z–Y plane)",  "Z [m]", "Y [m]"),
    ]
    for ax, (xi, yi, title, xl, yl) in zip(axes, configs):
        for tri in range(len(gt_f)):
            xs = [v0[tri,xi], v1[tri,xi], v2[tri,xi], v0[tri,xi]]
            ys = [v0[tri,yi], v1[tri,yi], v2[tri,yi], v0[tri,yi]]
            ax.fill(xs, ys, color="#3498db", alpha=0.35)
            ax.plot(xs, ys, color="#2980b9", lw=0.5)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.25)
        ax.set_xlabel(xl, fontsize=8); ax.set_ylabel(yl, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(title, fontsize=9, fontweight="bold")

    axes[0].set_title("(d) GT Mesh – Orthographic Views  (open +Z face visible in front)",
                      fontsize=9, fontweight="bold")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    # Layout:
    #  Row 0: [3D scene (tall)]  [depth map (tall)]
    #  Row 1: [4 silhouette renders]
    #  Row 2: [3 GT ortho views]

    fig = plt.figure(figsize=(14, 14))
    fig.suptitle("Synthetic Open-Cube Dataset Overview", fontweight="bold",
                 fontsize=13, y=0.98)

    gs_top = fig.add_gridspec(1, 2, top=0.96, bottom=0.60, hspace=0.05,
                              wspace=0.08, left=0.04, right=0.96)
    gs_mid = fig.add_gridspec(1, 4, top=0.56, bottom=0.34, hspace=0.0,
                              wspace=0.04, left=0.04, right=0.96)
    gs_bot = fig.add_gridspec(1, 3, top=0.30, bottom=0.04, hspace=0.0,
                              wspace=0.10, left=0.04, right=0.96)

    # 3D scene
    ax3d = fig.add_subplot(gs_top[0, 0], projection="3d")
    panel_3d_scene(ax3d)

    # Depth map
    ax_depth = fig.add_subplot(gs_top[0, 1])
    panel_depth(ax_depth)

    # Silhouette renders
    ax_renders = [fig.add_subplot(gs_mid[0, i]) for i in range(4)]
    panel_renders(ax_renders)
    ax_renders[0].set_title("(b) Silhouette Renders per Colour Camera",
                            fontsize=10, fontweight="bold",
                            loc="left", x=0.0, pad=4)

    # GT ortho views
    ax_ortho = [fig.add_subplot(gs_bot[0, i]) for i in range(3)]
    panel_gt_ortho(ax_ortho)

    out = OUT_DIR / "fig_synthetic_dataset_overview.png"
    fig.savefig(out, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  → {out.name}")


if __name__ == "__main__":
    main()
