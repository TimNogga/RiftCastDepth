#!/usr/bin/env python3
"""Render-test script – sweeps azimuth × elevation to find a good viewpoint."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESH_DIR = os.path.join(ROOT, "output", "figuresba", "02_tsdf_edge", "frame_00000")
OUT_DIR  = os.path.join(ROOT, "output", "figuresba", "render_test")

# ── colormap / light settings ────────────────────────────────────────────────
CMAP      = "plasma"        # height colormap
AMBIENT   = 0.25            # ambient light strength
DIFFUSE   = 0.75            # diffuse light strength
LIGHT_DIR = np.array([0.4, 0.9, 0.3], dtype=np.float64)  # world-space key light
LIGHT_DIR /= np.linalg.norm(LIGHT_DIR)
BG_COLOR  = "#1a1a2e"       # dark background — makes colours pop
FIG_SIZE  = (6, 8)          # portrait, person-shaped
DPI       = 130

# ── azimuth / elevation grid ─────────────────────────────────────────────────
AZIMUTHS   = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
ELEVATIONS = [0, 15, 30, 45, 60, 80]
# also: straight top, straight bottom
EXTRA = [("top", 0, 90), ("bottom", 0, -30)]


def load_mesh():
    verts = np.fromfile(os.path.join(MESH_DIR, "vertices.bin"), dtype=np.float32).reshape(-1, 3).astype(np.float64)
    faces = np.fromfile(os.path.join(MESH_DIR, "faces.bin"),    dtype=np.int64 ).reshape(-1, 3)
    return verts, faces


def view_matrix(elev_deg: float, azim_deg: float):
    """Rotation matrix R such that R @ v gives view-space coords (x=right, y=up, z=toward viewer)."""
    e = np.radians(elev_deg)
    a = np.radians(azim_deg)

    # Camera sits on a unit sphere and looks at the origin
    fwd = -np.array([np.cos(e) * np.sin(a), np.sin(e), np.cos(e) * np.cos(a)])
    fwd /= np.linalg.norm(fwd)

    world_up = np.array([0., 1., 0.])
    if abs(fwd @ world_up) > 0.99:
        world_up = np.array([0., 0., -1.])

    right = np.cross(world_up, fwd); right /= np.linalg.norm(right)
    up    = np.cross(fwd, right)

    # rows = basis vectors → R @ v transforms world→view
    return np.stack([right, up, fwd])   # (3,3)


def compute_face_normals_world(verts, faces):
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    n  = np.cross(v1 - v0, v2 - v0)
    nl = np.linalg.norm(n, axis=1, keepdims=True)
    return n / (nl + 1e-12)


def render(verts, faces, face_normals, elev_deg, azim_deg, label=""):
    R = view_matrix(elev_deg, azim_deg)

    # Centre mesh, project
    centre  = verts.mean(axis=0)
    vc      = verts - centre
    vv      = (R @ vc.T).T          # view-space: x=right, y=up, z=toward-viewer

    proj_2d = vv[:, :2]             # orthographic XY in view space
    # view-z: positive = closer to viewer
    view_z  = vv[:, 2]

    # ── per-face depth (use view-z for painter sort) ───────────────────────
    face_vz   = view_z[faces].mean(axis=1)
    sort_idx  = np.argsort(face_vz)           # paint back-to-front

    # ── height colour (world-Y, independent of view angle) ────────────────
    world_y   = verts[:, 1]
    face_y    = world_y[faces].mean(axis=1)
    y_min, y_max = face_y.min(), face_y.max()
    face_y_n  = (face_y - y_min) / max(y_max - y_min, 1e-8)

    # ── Lambertian shading ─────────────────────────────────────────────────
    fn = face_normals                         # world-space
    dot = fn @ LIGHT_DIR                      # (M,)
    # flip normals that face away from light on back side (double-sided shading)
    dot = np.abs(dot)                         # absolute so back-faces get same light
    shade = np.clip(AMBIENT + DIFFUSE * dot, 0., 1.)

    # ── combine colour + shade ─────────────────────────────────────────────
    cmap    = matplotlib.colormaps[CMAP]
    rgb     = cmap(face_y_n[sort_idx])[:, :3]   # (M, 3) ignoring alpha
    shaded  = np.clip(rgb * shade[sort_idx, None], 0., 1.)
    colors  = np.concatenate([shaded, np.ones((len(shaded), 1))], axis=1)  # RGBA

    polys = proj_2d[faces[sort_idx]]          # (M, 3, 2)

    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    col = PolyCollection(polys, facecolors=colors, edgecolors="none", linewidths=0)
    ax.add_collection(col)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")

    title = label if label else f"azim {azim_deg:+04d}°  elev {elev_deg:+03d}°"
    ax.set_title(title, color="white", fontsize=11, pad=6)

    # small colorbar on the right
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(y_min, y_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, orientation="vertical")
    cbar.set_label("height (m)", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=7)

    plt.tight_layout(pad=0.3)
    return fig


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Loading mesh from {MESH_DIR} …")
    verts, faces = load_mesh()
    face_normals = compute_face_normals_world(verts, faces)
    print(f"  {len(verts):,} verts · {len(faces):,} faces")
    print(f"  Output → {OUT_DIR}\n")

    total = len(AZIMUTHS) * len(ELEVATIONS) + len(EXTRA)
    done  = 0

    for elev in ELEVATIONS:
        for azim in AZIMUTHS:
            fname = f"azim{azim:03d}_elev{elev:02d}.png"
            fig   = render(verts, faces, face_normals, elev, azim)
            fig.savefig(os.path.join(OUT_DIR, fname), dpi=DPI, bbox_inches="tight",
                        facecolor=BG_COLOR)
            plt.close(fig)
            done += 1
            print(f"  [{done:3d}/{total}] {fname}")

    for name, azim, elev in EXTRA:
        fname = f"_{name}.png"
        fig   = render(verts, faces, face_normals, elev, azim, label=name)
        fig.savefig(os.path.join(OUT_DIR, fname), dpi=DPI, bbox_inches="tight",
                    facecolor=BG_COLOR)
        plt.close(fig)
        done += 1
        print(f"  [{done:3d}/{total}] {fname}")

    print(f"\nDone. {total} renders in {OUT_DIR}")


if __name__ == "__main__":
    main()
