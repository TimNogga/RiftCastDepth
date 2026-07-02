#!/usr/bin/env python3
"""
Depth-coloured mesh renders for the figuresba synthetic ablation study.
No RGB images (synthetic dataset has no texture).

Produces:
  output/figuresba/synthetic/renders/<exp>.png   – one clean render per experiment
  output/figuresba/synthetic/comparison.png      – all 5 side-by-side
  output/figuresba/synthetic/stats.md            – mesh statistics table
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNTH   = os.path.join(ROOT, "output", "figuresba", "synthetic")
RENDERS = os.path.join(SYNTH, "renders")

EXPERIMENTS = [
    ("01_no_depth",            "Baseline\n(no depth)"),
    ("02_tsdf_edge_gradient",  "TSDF +\nedge gradient"),
    ("03_tsdf_no_edge_gradient", "TSDF, no\nedge gradient"),
    ("04_tsdf_post_mc_cutter", "TSDF +\npost-MC cutter"),
    ("05_cutter_no_tsdf",      "Cutter only\n(no TSDF)"),
]

AZIM      = 270
ELEV      = 60
# Front view: D000 is at [0,0,2.5], opening faces +Z → look from +Z toward origin
AZIM_FRONT = 0
ELEV_FRONT = 0
CMAP      = "plasma"
AMBIENT   = 0.25
DIFFUSE   = 0.75
LIGHT_DIR = np.array([0.4, 0.9, 0.3], dtype=np.float64)
LIGHT_DIR /= np.linalg.norm(LIGHT_DIR)
BG        = "#1a1a2e"
DPI       = 150


def load_mesh(exp_name: str):
    d  = os.path.join(SYNTH, exp_name, "frame_00000")
    vp = os.path.join(d, "vertices.bin")
    fp = os.path.join(d, "faces.bin")
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
    fwd  = -np.array([np.cos(e) * np.sin(a), np.sin(e), np.cos(e) * np.cos(a)])
    fwd  /= np.linalg.norm(fwd)
    up    = np.array([0., 1., 0.]) if abs(fwd[1]) < 0.99 else np.array([0., 0., -1.])
    right = np.cross(up, fwd); right /= np.linalg.norm(right)
    return np.stack([right, np.cross(fwd, right), fwd])


def render_ax(ax, verts, faces, R, y_min, y_max):
    vc     = verts - verts.mean(axis=0)
    vv     = (R @ vc.T).T
    proj   = vv[:, :2]
    view_z = vv[:, 2]

    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn /= (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12)

    order  = np.argsort(view_z[faces].mean(axis=1))
    face_y = verts[faces, 1].mean(axis=1)
    y_norm = (face_y - y_min) / max(y_max - y_min, 1e-8)
    shade  = np.clip(AMBIENT + DIFFUSE * np.abs(fn @ LIGHT_DIR), 0., 1.)

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


def render_single(exp_name, verts, faces, R, y_min, y_max):
    fig, ax = plt.subplots(figsize=(5, 7), facecolor=BG)
    render_ax(ax, verts, faces, R, y_min, y_max)

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(y_min, y_max))
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


def render_single_to(exp_name, verts, faces, R, y_min, y_max, out_path):
    fig, ax = plt.subplots(figsize=(5, 5), facecolor=BG)
    render_ax(ax, verts, faces, R, y_min, y_max)

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(y_min, y_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("depth (m)", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)

    plt.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out_path


def render_comparison(exp_data, R, y_min, y_max):
    valid = [(n, l, v, f) for n, l, v, f in exp_data if v is not None]
    n     = len(valid)
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
    out = os.path.join(SYNTH, "comparison.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out


def write_stats(exp_data):
    lines = ["# figuresba synthetic – Mesh Statistics (frame 0)\n",
             "| Experiment | Vertices | Faces | Surface Area (m²) |",
             "|---|---:|---:|---:|"]
    for _, label, verts, faces in exp_data:
        tag = label.replace("\n", " ")
        if verts is None:
            lines.append(f"| **{tag}** | – | – | – |")
        else:
            nv, nf, area = mesh_stats(verts, faces)
            lines.append(f"| **{tag}** | {nv:,} | {nf:,} | {area:.4f} |")
    out = os.path.join(SYNTH, "stats.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    return out, lines


def main():
    os.makedirs(RENDERS, exist_ok=True)
    R = view_matrix(ELEV, AZIM)

    all_y  = []
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

    print(f"\n{'─'*50}\nStats table")
    out, lines = write_stats(exp_data)
    print(f"  → {out}")
    for l in lines:
        print(l)

    print("\nAll done.")


if __name__ == "__main__":
    main()
