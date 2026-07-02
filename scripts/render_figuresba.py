#!/usr/bin/env python3
"""
Depth-coloured mesh renders for the figuresba ablation study.
Fixed viewpoint: azim=270°, elev=60°  (chosen from render_test_angles.py sweep).

Produces:
  output/figuresba/renders/<exp>.png   – one clean render per experiment
  output/figuresba/comparison.png      – all 5 side-by-side
  output/figuresba/stats.md            – mesh statistics table
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURESBA = os.path.join(ROOT, "output", "figuresba")
RENDERS   = os.path.join(FIGURESBA, "renders")

EXPERIMENTS = [
    ("01_baseline",     "Baseline\n(no depth)"),
    ("02_tsdf_edge",    "TSDF +\nedge gradient"),
    ("03_tsdf_no_edge", "TSDF, no\nedge gradient"),
    ("04_tsdf_cutter",  "TSDF +\npost-MC cutter"),
    ("05_cutter_only",  "Cutter only\n(no TSDF)"),
]

# ── render settings ───────────────────────────────────────────────────────────
AZIM      = 270
ELEV      = 60
CMAP      = "plasma"
AMBIENT   = 0.25
DIFFUSE   = 0.75
LIGHT_DIR = np.array([0.4, 0.9, 0.3], dtype=np.float64)
LIGHT_DIR /= np.linalg.norm(LIGHT_DIR)
BG        = "#1a1a2e"
DPI       = 150


# ── helpers ───────────────────────────────────────────────────────────────────

def load_mesh(exp_name: str):
    d = os.path.join(FIGURESBA, exp_name, "frame_00000")
    vp, fp = os.path.join(d, "vertices.bin"), os.path.join(d, "faces.bin")
    if not os.path.isfile(vp):
        return None, None
    verts = np.fromfile(vp, dtype=np.float32).reshape(-1, 3).astype(np.float64)
    faces = np.fromfile(fp, dtype=np.int64).reshape(-1, 3)
    return verts, faces


def mesh_stats(verts, faces):
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum()
    return len(verts), len(faces), area


def view_matrix(elev_deg, azim_deg):
    e, a = np.radians(elev_deg), np.radians(azim_deg)
    fwd = -np.array([np.cos(e) * np.sin(a), np.sin(e), np.cos(e) * np.cos(a)])
    fwd /= np.linalg.norm(fwd)
    up = np.array([0., 1., 0.]) if abs(fwd[1]) < 0.99 else np.array([0., 0., -1.])
    right = np.cross(up, fwd); right /= np.linalg.norm(right)
    return np.stack([right, np.cross(fwd, right), fwd])   # (3,3)


def render_ax(ax, verts, faces, R, y_min, y_max):
    """Paint the mesh onto ax with Lambertian shading and height colouring."""
    vc     = verts - verts.mean(axis=0)
    vv     = (R @ vc.T).T
    proj   = vv[:, :2]
    view_z = vv[:, 2]

    # face normals (world-space, for lighting)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn /= (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12)

    # painter sort
    order = np.argsort(view_z[faces].mean(axis=1))

    # height colour (world-Y)
    face_y = verts[faces, 1].mean(axis=1)
    y_norm = (face_y - y_min) / max(y_max - y_min, 1e-8)

    # shading
    shade = np.clip(AMBIENT + DIFFUSE * np.abs(fn @ LIGHT_DIR), 0., 1.)

    rgb    = matplotlib.colormaps[CMAP](y_norm[order])[:, :3]
    colors = np.clip(rgb * shade[order, None], 0., 1.)
    colors = np.concatenate([colors, np.ones((len(colors), 1))], axis=1)

    col = PolyCollection(proj[faces[order]], facecolors=colors,
                         edgecolors="none", linewidths=0)
    ax.add_collection(col)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(BG)


# ── per-experiment single render ──────────────────────────────────────────────

def render_single(exp_name, verts, faces, R, y_min, y_max):
    fig, ax = plt.subplots(figsize=(5, 7), facecolor=BG)
    render_ax(ax, verts, faces, R, y_min, y_max)

    sm = plt.cm.ScalarMappable(cmap=CMAP,
                                norm=plt.Normalize(y_min, y_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("depth (m)", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)

    plt.tight_layout(pad=0.3)
    out = os.path.join(RENDERS, f"{exp_name}.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out


# ── comparison figure ─────────────────────────────────────────────────────────

def render_comparison(exp_data, R, y_min, y_max):
    valid = [(n, l, v, f) for n, l, v, f in exp_data if v is not None]
    n = len(valid)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 7), facecolor=BG)
    if n == 1:
        axes = [axes]

    for ax, (name, label, verts, faces) in zip(axes, valid):
        render_ax(ax, verts, faces, R, y_min, y_max)
        nv, nf, area = mesh_stats(verts, faces)
        ax.set_title(
            f"{label}\n{nv:,}v · {nf:,}f · {area:.2f} m²",
            color="white", fontsize=9, fontweight="bold", pad=4
        )

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(y_min, y_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=list(axes), orientation="horizontal",
                        pad=0.02, shrink=0.5, aspect=30)
    cbar.set_label("depth (m)", color="white", fontsize=10)
    cbar.ax.xaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.xaxis.get_ticklabels(), color="white", fontsize=8)

    plt.tight_layout(pad=0.4)
    out = os.path.join(FIGURESBA, "comparison.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out


# ── comparison with RGB below ─────────────────────────────────────────────────

def render_comparison_with_rgb(exp_data, R, y_min, y_max):
    """
    Two-row comparison: depth render on top, RGB image directly below.
    No mesh statistics in titles.
    """
    import matplotlib.image as mpimg

    valid = [(n, l, v, f) for n, l, v, f in exp_data if v is not None]
    n = len(valid)

    # height_ratios: depth render is taller (7), RGB is shorter (4)
    fig, axes = plt.subplots(
        2, n,
        figsize=(4.5 * n, 13),
        facecolor=BG,
        gridspec_kw={"height_ratios": [7, 4], "hspace": 0.04},
    )
    if n == 1:
        axes = [[axes[0]], [axes[1]]]

    for col, (name, label, verts, faces) in enumerate(valid):
        ax_depth = axes[0][col]
        ax_rgb   = axes[1][col]

        # ── depth render ──
        render_ax(ax_depth, verts, faces, R, y_min, y_max)
        ax_depth.set_title(label, color="white", fontsize=10,
                           fontweight="bold", pad=6)

        # ── RGB photo ──
        rgb_path = os.path.join(
            FIGURESBA, name, "frame_00000", "rgb", "rgb_0.png"
        )
        if os.path.isfile(rgb_path):
            img = mpimg.imread(rgb_path)
            ax_rgb.imshow(img)
        else:
            ax_rgb.set_facecolor(BG)
            ax_rgb.text(0.5, 0.5, "no rgb", color="white",
                        ha="center", va="center", transform=ax_rgb.transAxes)
        ax_rgb.axis("off")

    # shared colorbar
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(y_min, y_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=list(axes[0]), orientation="horizontal",
                        pad=0.01, shrink=0.5, aspect=30)
    cbar.set_label("depth (m)", color="white", fontsize=10)
    cbar.ax.xaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.xaxis.get_ticklabels(), color="white", fontsize=8)

    out = os.path.join(FIGURESBA, "comparison_with_rgb.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out


# ── comparison with pre-cropped RGB below ────────────────────────────────────

def render_comparison_with_cropped_rgb(exp_data, R, y_min, y_max):
    """
    Two-row comparison: depth render on top, pre-cropped RGB image below.
    Reads from output/figuresba/rgb/<exp_name>.png (produced by copy_rgb_images).
    """
    import matplotlib.image as mpimg

    valid = [(n, l, v, f) for n, l, v, f in exp_data if v is not None]
    n = len(valid)

    fig, axes = plt.subplots(
        2, n,
        figsize=(4.5 * n, 13),
        facecolor=BG,
        gridspec_kw={"height_ratios": [7, 4], "hspace": 0.04},
    )
    if n == 1:
        axes = [[axes[0]], [axes[1]]]

    for col, (name, label, verts, faces) in enumerate(valid):
        ax_depth = axes[0][col]
        ax_rgb   = axes[1][col]

        render_ax(ax_depth, verts, faces, R, y_min, y_max)
        ax_depth.set_title(label, color="white", fontsize=10,
                           fontweight="bold", pad=6)

        rgb_path = os.path.join(FIGURESBA, "rgb", f"{name}.png")
        if os.path.isfile(rgb_path):
            img = mpimg.imread(rgb_path)
            ax_rgb.imshow(img)
        else:
            ax_rgb.set_facecolor(BG)
            ax_rgb.text(0.5, 0.5, "no cropped rgb\n(run copy_rgb_images first)",
                        color="white", ha="center", va="center",
                        transform=ax_rgb.transAxes, fontsize=8)
        ax_rgb.axis("off")

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(y_min, y_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=list(axes[0]), orientation="horizontal",
                        pad=0.01, shrink=0.5, aspect=30)
    cbar.set_label("depth (m)", color="white", fontsize=10)
    cbar.ax.xaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.xaxis.get_ticklabels(), color="white", fontsize=8)

    out = os.path.join(FIGURESBA, "comparison_with_cropped_rgb.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out


# ── rgb strip ────────────────────────────────────────────────────────────────

def copy_rgb_images(exp_data):
    import matplotlib.image as mpimg
    from PIL import Image

    rgb_dir = os.path.join(FIGURESBA, "rgb")
    os.makedirs(rgb_dir, exist_ok=True)

    # First pass: collect the union bounding box across all images
    r0_all, r1_all, c0_all, c1_all = [], [], [], []
    sources = {}
    for name, _, v, __ in exp_data:
        src = os.path.join(FIGURESBA, name, "frame_00000", "rgb", "rgb_0.png")
        if not os.path.isfile(src):
            continue
        sources[name] = src
        img = np.array(Image.open(src).convert("RGB"))
        mask = np.any(img < 240, axis=2)
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            continue
        r0_all.append(rows[0]);  r1_all.append(rows[-1])
        c0_all.append(cols[0]);  c1_all.append(cols[-1])

    if not r0_all:
        return []

    PAD = 120
    r0 = max(0, min(r0_all) - PAD)
    c0 = max(0, min(c0_all) - PAD)

    # Second pass: crop every image to the same box
    copied = []
    for name, _, v, __ in exp_data:
        src = sources.get(name)
        if src is None:
            continue
        img = Image.open(src).convert("RGB")
        r1 = min(img.height, max(r1_all) + PAD)
        c1 = min(img.width,  max(c1_all) + PAD)
        cropped = img.crop((c0, r0, c1, r1))
        dst = os.path.join(rgb_dir, f"{name}.png")
        cropped.save(dst)
        copied.append(dst)
    return copied


# ── stats table ───────────────────────────────────────────────────────────────

def write_stats(exp_data):
    lines = ["# figuresba – Mesh Statistics (frame 0)\n",
             "| Experiment | Vertices | Faces | Surface Area (m²) |",
             "|---|---:|---:|---:|"]
    for _, label, verts, faces in exp_data:
        tag = label.replace("\n", " ")
        if verts is None:
            lines.append(f"| **{tag}** | – | – | – |")
        else:
            nv, nf, area = mesh_stats(verts, faces)
            lines.append(f"| **{tag}** | {nv:,} | {nf:,} | {area:.4f} |")
    out = os.path.join(FIGURESBA, "stats.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    return out, lines


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RENDERS, exist_ok=True)
    R = view_matrix(ELEV, AZIM)

    # compute shared Y range across all experiments for consistent colouring
    all_y = []
    meshes = {}
    for exp_name, exp_label in EXPERIMENTS:
        v, f = load_mesh(exp_name)
        meshes[exp_name] = (v, f, exp_label)
        if v is not None:
            all_y.append(v[:, 1])
    y_min = np.concatenate(all_y).min()
    y_max = np.concatenate(all_y).max()

    exp_data = []
    for exp_name, exp_label in EXPERIMENTS:
        print(f"\n{'─'*50}\n{exp_name}")
        v, f, _ = meshes[exp_name]
        if v is None:
            print("  [SKIP] no data")
            exp_data.append((exp_name, exp_label, None, None))
            continue
        nv, nf, area = mesh_stats(v, f)
        print(f"  {nv:,} verts · {nf:,} faces · {area:.4f} m²")
        out = render_single(exp_name, v, f, R, y_min, y_max)
        print(f"  → {out}")
        exp_data.append((exp_name, exp_label, v, f))

    print(f"\n{'─'*50}\nComparison figure")
    out = render_comparison(exp_data, R, y_min, y_max)
    print(f"  → {out}")

    print(f"\n{'─'*50}\nComparison with RGB")
    out = render_comparison_with_rgb(exp_data, R, y_min, y_max)
    print(f"  → {out}")

    print(f"\n{'─'*50}\nRGB images")
    copied = copy_rgb_images(exp_data)
    for p in copied:
        print(f"  → {p}")

    print(f"\n{'─'*50}\nComparison with cropped RGB")
    out = render_comparison_with_cropped_rgb(exp_data, R, y_min, y_max)
    print(f"  → {out}")

    print(f"\n{'─'*50}\nStats table")
    out, lines = write_stats(exp_data)
    print(f"  → {out}")
    for l in lines:
        print(l)

    print("\nAll done.")


if __name__ == "__main__":
    main()
