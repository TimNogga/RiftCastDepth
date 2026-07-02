#!/usr/bin/env python3
"""
Generate thesis evaluation figures.

output/evaluation/thesis_figures/:
  fig_01_vci_rendered_comparison.png   – cropped body views, 3 configs × 4 cameras
  fig_02_vci_error_maps.png            – error maps, same layout
  fig_03_vci_mesh_projections.png      – ortho projections, 3 configs × 3 views
  fig_04_synth_mesh_projections.png    – ortho projections, 3 synth runs × 3 views (+ GT)
  fig_05_synth_chamfer_heatmap.png     – per-vertex chamfer distance coloured
  fig_06_depth_sensor_view.png         – what the depth camera sees

Run from repo root:
    python3 eval/generate_thesis_figures.py
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR   = REPO_ROOT / "output" / "evaluation" / "thesis_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 180, "font.size": 10,
                     "axes.titlesize": 10, "figure.facecolor": "white"})

VCI_EXPS = {
    "No Depth (Baseline)": REPO_ROOT / "output/comparative_study/exp_01_no_depth",
    "Depth TSDF":          REPO_ROOT / "output/comparative_study/exp_02_depth_tsdf",
    "Depth + Cutter":      REPO_ROOT / "output/comparative_study/exp_03_depth_cutter_020m",
}
VCI_COLORS = ["#e74c3c", "#3498db", "#2ecc71"]

SYNTH_EXPS = {
    "No Depth (Baseline)":      REPO_ROOT / "runtime/outputs/runtime_depth_fusion_mode_none_no_depth",
    "Depth TSDF (real preset)": REPO_ROOT / "runtime/outputs/runtime_depth_fusion_mode_real_data",
    "Depth TSDF (synth preset)":REPO_ROOT / "runtime/outputs/runtime_depth_fusion_mode_synthetic_data",
}
SYNTH_COLORS = ["#e74c3c", "#3498db", "#2ecc71"]

GT_OBJ   = REPO_ROOT / "scripts" / "ground_truth_open_box.obj"
VCI_CAMS = [5, 14, 26, 37]
PAD_FRAC = 0.06   # padding as fraction of bounding-box size


# ── bounding box helpers ───────────────────────────────────────────────────

def content_bbox(img_rgb: np.ndarray, threshold: int = 240):
    """Return (rmin, rmax, cmin, cmax) of non-white pixels."""
    mask = ~np.all(img_rgb > threshold, axis=2)
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return 0, img_rgb.shape[0]-1, 0, img_rgb.shape[1]-1
    return int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])


def padded_bbox(rmin, rmax, cmin, cmax, H, W, pad_frac=PAD_FRAC):
    ph = max(int((rmax - rmin) * pad_frac), 10)
    pw = max(int((cmax - cmin) * pad_frac), 10)
    return (max(0, rmin - ph), min(H - 1, rmax + ph),
            max(0, cmin - pw), min(W - 1, cmax + pw))


def compute_camera_bboxes(cam_ids, reference_dir: Path, pad_frac=PAD_FRAC):
    """Compute one padded bounding box per camera from the reference experiment."""
    bboxes = {}
    for cam_id in cam_ids:
        p = reference_dir / f"frame_00000/rgb/rgb_{cam_id}.png"
        if not p.exists():
            continue
        rgb = np.array(Image.open(p).convert("RGB"))
        r0, r1, c0, c1 = content_bbox(rgb)
        bboxes[cam_id] = padded_bbox(r0, r1, c0, c1, rgb.shape[0], rgb.shape[1], pad_frac)
    return bboxes


def crop_and_resize(img_rgba: np.ndarray, bbox, target_h: int = 400) -> np.ndarray:
    """Crop to bbox and resize to target_h (keeping aspect)."""
    r0, r1, c0, c1 = bbox
    crop = img_rgba[r0:r1+1, c0:c1+1]
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return np.ones((target_h, max(1, target_h // 3), 4), dtype=np.uint8) * 255
    new_w = max(1, int(w * target_h / h))
    pil = Image.fromarray(crop)
    pil = pil.resize((new_w, target_h), Image.LANCZOS)
    return np.array(pil)


# ── mesh helpers ───────────────────────────────────────────────────────────

def load_bin_mesh(path: Path):
    vp = path / "frame_00000/vertices.bin"
    fp = path / "frame_00000/faces.bin"
    if not vp.exists():
        return None, None
    return (np.fromfile(vp, dtype=np.float32).reshape(-1, 3),
            np.fromfile(fp, dtype=np.int64).reshape(-1, 3))


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


def sample_surface(v, f, n=50_000):
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1-v0, v2-v0), axis=1)
    total = areas.sum()
    if total == 0:
        return v[:n]
    probs = areas / total
    tri   = np.random.choice(len(f), n, p=probs)
    r1, r2 = np.random.rand(n, 1), np.random.rand(n, 1)
    mask = (r1+r2) > 1
    r1[mask] = 1 - r1[mask]; r2[mask] = 1 - r2[mask]
    return (r1*v[f[tri,0]] + r2*v[f[tri,1]] + (1-r1-r2)*v[f[tri,2]]).astype(np.float32)


def downsample(v: np.ndarray, n: int = 80_000) -> np.ndarray:
    if len(v) <= n:
        return v
    return v[np.random.choice(len(v), n, replace=False)]


def ortho(ax, pts, xi, yi, color, s=0.04, alpha=0.3):
    ax.scatter(pts[:, xi], pts[:, yi], s=s, c=color, alpha=alpha, rasterized=True)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.2)


def save(fig, name: str):
    p = OUT_DIR / name
    fig.savefig(p, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  → {name}")


# ── Figure 1 – VCI rendered views (cropped) ───────────────────────────────

def fig_vci_rendered(bboxes: dict):
    print("[1] VCI rendered comparison …")
    labels = list(VCI_EXPS.keys())
    dirs   = list(VCI_EXPS.values())
    cams   = [c for c in VCI_CAMS if c in bboxes]

    fig, axes = plt.subplots(len(labels), len(cams),
                             figsize=(2.2 * len(cams), 5.5 * len(labels)))
    fig.suptitle("Rendered Views per Configuration", fontweight="bold", fontsize=12, y=1.01)

    for row, (label, d, color) in enumerate(zip(labels, dirs, VCI_COLORS)):
        for col, cam_id in enumerate(cams):
            ax = axes[row][col]
            p  = d / f"frame_00000/rgb/rgb_{cam_id}.png"
            if p.exists():
                img  = np.array(Image.open(p).convert("RGBA"))
                crop = crop_and_resize(img, bboxes[cam_id])
                ax.imshow(crop); ax.axis("off")
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes); ax.axis("off")
            if col == 0:
                ax.set_ylabel(label, color=color, fontsize=9,
                              fontweight="bold", labelpad=4)
            if row == 0:
                ax.set_title(f"Cam {cam_id}", fontsize=9)
            for spine in ax.spines.values():
                spine.set_edgecolor(color); spine.set_linewidth(1.5)

    plt.tight_layout()
    save(fig, "fig_01_vci_rendered_comparison.png")


# ── Figure 2 – VCI error maps (cropped) ───────────────────────────────────

def fig_vci_errors(bboxes: dict):
    print("[2] VCI error maps …")
    labels = list(VCI_EXPS.keys())
    dirs   = list(VCI_EXPS.values())
    cams   = [c for c in VCI_CAMS if c in bboxes]

    fig, axes = plt.subplots(len(labels), len(cams),
                             figsize=(2.2 * len(cams), 5.5 * len(labels)))
    fig.suptitle("Rendering Error Maps (brighter = higher error)",
                 fontweight="bold", fontsize=12, y=1.01)

    for row, (label, d, color) in enumerate(zip(labels, dirs, VCI_COLORS)):
        for col, cam_id in enumerate(cams):
            ax = axes[row][col]
            p  = d / f"frame_00000/error/error_{cam_id}.png"
            if p.exists():
                err  = np.array(Image.open(p).convert("L"))
                # convert to 3-channel for consistent crop_and_resize
                err3 = np.stack([err, err, err], axis=2)
                crop = crop_and_resize(err3, bboxes[cam_id])
                ax.imshow(crop[:, :, 0], cmap="hot", vmin=0, vmax=255)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(label, color=color, fontsize=9,
                              fontweight="bold", labelpad=4)
            if row == 0:
                ax.set_title(f"Cam {cam_id}", fontsize=9)

    plt.tight_layout()
    save(fig, "fig_02_vci_error_maps.png")


# ── Figure 3 – VCI mesh ortho projections ─────────────────────────────────

def fig_vci_projections():
    print("[3] VCI mesh projections …")
    labels = list(VCI_EXPS.keys())
    proj   = [(0, 1, "Front (X–Y)", "X [m]", "Y [m]"),
              (0, 2, "Top  (X–Z)",  "X [m]", "Z [m]"),
              (2, 1, "Side (Z–Y)",  "Z [m]", "Y [m]")]

    fig, axes = plt.subplots(len(proj), len(labels),
                             figsize=(4 * len(labels), 4 * len(proj)))
    fig.suptitle("Mesh Orthographic Projections – VCI Dataset",
                 fontweight="bold", fontsize=12)

    for col, (label, d, color) in enumerate(zip(labels, VCI_EXPS.values(), VCI_COLORS)):
        v, _ = load_bin_mesh(d)
        pts  = downsample(v) if v is not None else None
        for row, (xi, yi, view, xl, yl) in enumerate(proj):
            ax = axes[row][col]
            if pts is not None:
                ortho(ax, pts, xi, yi, color)
            ax.set_xlabel(xl, fontsize=8)
            if row == 0:
                n = f"{len(v):,}" if v is not None else "?"
                ax.set_title(f"{label}\n{n} verts", color=color,
                             fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{view}\n{yl}", fontsize=8)
            else:
                ax.set_ylabel(yl, fontsize=8)

    plt.tight_layout()
    save(fig, "fig_03_vci_mesh_projections.png")


# ── Figure 4 – Synthetic mesh ortho projections ───────────────────────────

def fig_synth_projections():
    print("[4] Synthetic mesh projections …")
    proj = [(0, 1, "Front (X–Y)", "X [m]", "Y [m]"),
            (0, 2, "Top  (X–Z)",  "X [m]", "Z [m]"),
            (2, 1, "Side (Z–Y)",  "Z [m]", "Y [m]")]

    gt_v, gt_f = load_obj(GT_OBJ)
    gt_pts     = sample_surface(gt_v, gt_f, 30_000)

    all_labels = ["GT (open box)"] + list(SYNTH_EXPS.keys())
    all_dirs   = [None] + list(SYNTH_EXPS.values())
    all_colors = ["#7f8c8d"] + SYNTH_COLORS

    fig, axes = plt.subplots(len(proj), len(all_labels),
                             figsize=(3.8 * len(all_labels), 4 * len(proj)))
    fig.suptitle("Mesh Orthographic Projections – Synthetic Dataset  (grey = GT outline)",
                 fontweight="bold", fontsize=12)

    for col, (label, d, color) in enumerate(zip(all_labels, all_dirs, all_colors)):
        pts = gt_pts if d is None else downsample(load_bin_mesh(d)[0])
        for row, (xi, yi, view, xl, yl) in enumerate(proj):
            ax = axes[row][col]
            if pts is not None:
                ortho(ax, pts, xi, yi, color)
            if col > 0 and pts is not None:   # grey GT overlay
                ax.scatter(gt_pts[:, xi], gt_pts[:, yi], s=0.005,
                           c="#aaaaaa", alpha=0.15, rasterized=True)
            ax.set_xlabel(xl, fontsize=8)
            if row == 0:
                n = f"{len(pts):,}" if pts is not None else "?"
                ax.set_title(f"{label}\n{n} pts", color=color,
                             fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{view}\n{yl}", fontsize=8)
            else:
                ax.set_ylabel(yl, fontsize=8)

    plt.tight_layout()
    save(fig, "fig_04_synth_mesh_projections.png")


# ── Figure 5 – Chamfer heatmap ─────────────────────────────────────────────

def fig_chamfer_heatmap():
    print("[5] Chamfer distance heatmap …")
    gt_v, gt_f = load_obj(GT_OBJ)
    pts_gt     = sample_surface(gt_v, gt_f, 100_000)
    tree_gt    = cKDTree(pts_gt)

    labels = list(SYNTH_EXPS.keys())
    proj   = [(0, 1, "Front (X–Y)"),
              (0, 2, "Top  (X–Z)"),
              (2, 1, "Side (Z–Y)")]
    vmax   = 0.15

    fig, axes = plt.subplots(len(proj), len(labels),
                             figsize=(4.5 * len(labels), 4 * len(proj)))
    fig.suptitle("Chamfer Distance to GT Surface [m]  (blue = accurate, red = far)",
                 fontweight="bold", fontsize=12)

    for col, (label, d) in enumerate(SYNTH_EXPS.items()):
        v, _ = load_bin_mesh(d)
        pts  = downsample(v, 60_000) if v is not None else None
        if pts is None:
            continue
        dists, _ = tree_gt.query(pts, workers=-1)
        dists     = np.clip(dists, 0, vmax)

        for row, (xi, yi, title) in enumerate(proj):
            ax = axes[row][col]
            sc = ax.scatter(pts[:, xi], pts[:, yi], c=dists, cmap="coolwarm_r",
                            s=0.3, alpha=0.5, vmin=0, vmax=vmax, rasterized=True)
            ax.set_aspect("equal"); ax.grid(True, alpha=0.2)
            if row == 0:
                ax.set_title(label, fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(title, fontsize=8)
            if row == len(proj)-1 and col == len(labels)-1:
                plt.colorbar(sc, ax=ax, label="dist to GT [m]",
                             fraction=0.046, pad=0.04)

    plt.tight_layout()
    save(fig, "fig_05_synth_chamfer_heatmap.png")


# ── Figure 6 – Depth sensor view ──────────────────────────────────────────

def fig_depth_views():
    print("[6] Depth sensor views …")
    sources = []

    # Synthetic – depth preview already rendered
    p = REPO_ROOT / "data/synthetic_open_cube_8cams/frame_00000/depth_preview/D003L_from_pth_preview.jpg"
    if p.exists():
        sources.append(("Synthetic – D003L\n(looking into open face)",
                        np.array(Image.open(p).convert("RGB"))))

    # VCI – load raw .pth depth maps and visualise
    import importlib.util, sys
    try:
        import torch
        import matplotlib.cm as mcm
        for name in ["D003L", "D005Z"]:
            pth = REPO_ROOT / f"data/2026_03_20_orbbec_002_standard/frame_00000/rgb/{name}.pth"
            if not pth.exists():
                continue
            d = torch.load(str(pth), map_location="cpu").numpy().astype(np.float32) / 1000.0
            valid = (d > 0.1) & (d < 4.0)
            vis   = np.zeros_like(d)
            if valid.any():
                dmin, dmax = d[valid].min(), d[valid].max()
                vis[valid] = (d[valid] - dmin) / max(dmax - dmin, 1e-6)
            rgb = (mcm.turbo(vis)[:, :, :3] * 255).astype(np.uint8)
            rgb[~valid] = 30
            sources.append((f"VCI – {name}", rgb))
    except (ImportError, Exception):
        pass

    if not sources:
        print("  SKIP – no depth data found"); return

    fig, axes = plt.subplots(1, len(sources), figsize=(6 * len(sources), 4))
    fig.suptitle("Depth Sensor Views", fontweight="bold", fontsize=12)
    if len(sources) == 1:
        axes = [axes]
    for ax, (title, img) in zip(axes, sources):
        ax.imshow(img); ax.axis("off")
        ax.set_title(title, fontsize=10, fontweight="bold")

    plt.tight_layout()
    save(fig, "fig_06_depth_sensor_views.png")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)
    print("=" * 55)
    print("Thesis Figure Generation")
    print("=" * 55)

    # Pre-compute per-camera bounding boxes from baseline experiment
    ref_dir = list(VCI_EXPS.values())[0]
    print("\nComputing body bounding boxes …")
    bboxes = compute_camera_bboxes(VCI_CAMS, ref_dir)
    for cam_id, bb in bboxes.items():
        r0, r1, c0, c1 = bb
        print(f"  cam {cam_id}: rows {r0}-{r1}  cols {c0}-{c1}")

    fig_vci_rendered(bboxes)
    fig_vci_errors(bboxes)
    fig_vci_projections()
    fig_synth_projections()
    fig_chamfer_heatmap()
    fig_depth_views()

    n = len(list(OUT_DIR.glob("*.png")))
    print(f"\nDone. {n} figures → {OUT_DIR}")


if __name__ == "__main__":
    main()
