#!/usr/bin/env python3
"""
visualize_cutter.py  –  show which depth camera cuts which vertices post-marching-cubes.

Outputs:
  <out>.obj  – vertex-colored OBJ (v x y z r g b) readable in MeshLab / Blender
  <out>.png  – rendered PNG at azim=270° elev=60° (same angle as render_figuresba.py)

Color key (both formats):
  gray    = kept
  red     = cut by D003L only
  blue    = cut by D005Z only
  magenta = cut by both cameras

Usage:
    python scripts/visualize_cutter.py \\
        --config  configs/config_vci.json \\
        --vertices output/comparative_study/exp_03_depth_cutter/frame_00000/vertices.bin \\
        --faces    output/comparative_study/exp_03_depth_cutter/frame_00000/faces.bin \\
        --frame 0 \\
        --out   /tmp/cutter_viz

The script resolves all relative paths from the RIFTCast project root
(the directory containing this scripts/ folder).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

import numpy as np
import torch
from scipy.ndimage import maximum_filter, minimum_filter

ROOT = Path(__file__).resolve().parent.parent

# ── preset parameters (must match GeometryModule.cpp) ────────────────────────
PRESETS = {
    "realdata":      dict(forward_offset=0.24, edge_radius=10, edge_threshold=0.04),
    "real":          dict(forward_offset=0.24, edge_radius=10, edge_threshold=0.04),
    "real_no_edge":  dict(forward_offset=0.24, edge_radius=0,  edge_threshold=0.04),
    "syntheticdata": dict(forward_offset=0.02, edge_radius=2,  edge_threshold=0.06),
    "synthetic":     dict(forward_offset=0.02, edge_radius=2,  edge_threshold=0.06),
    "synth_no_edge": dict(forward_offset=0.02, edge_radius=0,  edge_threshold=0.06),
    "nodepth":       None,
    "none":          None,
}
DEPTH_MIN     = 0.1
DEPTH_MAX     = 3.95
SDF_THRESHOLD = 0.01

# ── visualization colours (RGB 0-1) ──────────────────────────────────────────
CAM_COLORS = [
    (1.00, 0.20, 0.20),   # red   – first depth cam
    (0.20, 0.40, 1.00),   # blue  – second depth cam
    (1.00, 0.75, 0.10),   # amber – third depth cam (if any)
    (0.15, 0.85, 0.35),   # green – fourth depth cam (if any)
]
KEPT_COLOR  = (0.70, 0.70, 0.70)
MULTI_COLOR = (0.90, 0.10, 0.90)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_mesh(v_path, f_path):
    v = np.fromfile(v_path, dtype=np.float32).reshape(-1, 3)
    f = np.fromfile(f_path, dtype=np.int64).reshape(-1, 3)
    return v.astype(np.float64), f


def compute_normals(verts, faces):
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    n = np.zeros_like(verts)
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    return n / (norm + 1e-8)


def load_depth_pth(path: Path, cam_H: int, cam_W: int, flip_images: bool) -> np.ndarray:
    """Load a .pth depth file → float32 array [H, W] in metres."""
    t = torch.load(str(path), map_location="cpu", weights_only=False).float().contiguous()

    # Normalise shape to 2-D [H, W], matching DatasetImporter.cpp logic
    if t.dim() == 3 and t.shape[0] == 3:
        t = t[0]
    elif t.dim() == 1:
        n = t.numel()
        if n == cam_H * cam_W:
            t = t.view(cam_H, cam_W)
        elif n == cam_H * cam_W * 3:
            t = t.view(3, cam_H, cam_W)[0]
        elif n % 3 == 0:
            gray = n // 3
            if gray % 1080 == 0:
                t = t.view(3, 1080, gray // 1080)[0]

    while t.dim() > 2:
        if t.shape[-1] == 1:   t = t.squeeze(-1)
        elif t.shape[0] == 1:  t = t.squeeze(0)
        else: break

    if t.dim() != 2:
        raise ValueError(f"Cannot reduce depth tensor to 2-D, got shape {t.shape}")

    depth = t.numpy().astype(np.float32)
    depth = depth * 4.0          # scale to metres (pth files store depth/4)

    if flip_images:
        depth = depth[::-1].copy()

    return depth


# ── camera / projection helpers ───────────────────────────────────────────────

def build_projection_gl(fx, fy, cx, cy, W, H, near=0.01, far=1000.0) -> np.ndarray:
    """OpenGL projection matching atcg::CameraUtils::convert_from_opencv."""
    P = np.zeros((4, 4), dtype=np.float64)
    P[0, 0] =  2.0 * fx / W
    P[1, 1] =  2.0 * fy / H
    P[0, 2] =  (W - 2.0 * cx) / W
    P[1, 2] =  (-H + 2.0 * cy) / H
    P[2, 2] = -(far + near) / (far - near)
    P[2, 3] = -2.0 * far * near / (far - near)
    P[3, 2] = -1.0
    return P


def build_view(extr_flat, to_world: np.ndarray) -> np.ndarray:
    """Replicates: cv_to_gl * glm::transpose(glm::make_mat4(extr)) * cv_to_gl * to_world."""
    M       = np.array(extr_flat, dtype=np.float64).reshape(4, 4)
    cv2gl   = np.diag([1.0, -1.0, -1.0, 1.0])
    return cv2gl @ M @ cv2gl @ to_world


def camera_world_pos(V: np.ndarray) -> np.ndarray:
    R, t = V[:3, :3], V[:3, 3]
    return -R.T @ t


# ── per-camera cutter simulation ─────────────────────────────────────────────

def simulate_cut(
    verts:   np.ndarray,   # [N, 3] float64
    normals: np.ndarray,   # [N, 3] float64
    VP:      np.ndarray,   # [4, 4] view-projection matrix
    V:       np.ndarray,   # [4, 4] view matrix (for camera position)
    depth:   np.ndarray,   # [H, W] float32, metres, after flip_images
    params:  dict,
) -> np.ndarray:           # [N] bool – True = this camera would remove the vertex
    """
    Python replica of the per-camera cutting loop in GeometryModule.cpp.
    The depth array is in the same state as depths[orig_cam_idx] after loading:
    flip_images already applied.  The cutter flips it one more time (depth[::-1]).
    """
    # Replicate cutter's raw_depth = depths[i].flip({0})
    depth_H, depth_W = depth.shape
    raw_depth = depth[::-1]   # double-flip = original sensor orientation

    N   = len(verts)
    hom = np.concatenate([verts, np.ones((N, 1), dtype=np.float64)], axis=1)  # [N, 4]
    clip = (VP @ hom.T).T   # [N, 4]

    depth_w = clip[:, 3]    # = -z_view (positive = in front)
    dw_safe = depth_w + 1e-12
    ndc_x   = clip[:, 0] / dw_safe
    ndc_y   = clip[:, 1] / dw_safe

    px = ((ndc_x + 1.0) * 0.5 * depth_W).astype(np.int64)
    py = ((1.0 - ndc_y) * 0.5 * depth_H).astype(np.int64)

    in_bounds    = (px >= 0) & (px < depth_W) & (py >= 0) & (py < depth_H)
    valid_depth_w = depth_w > DEPTH_MIN
    valid_proj   = in_bounds & valid_depth_w

    px_c = np.clip(px, 0, depth_W - 1)
    py_c = np.clip(py, 0, depth_H - 1)

    target = raw_depth[py_c, px_c]
    bad_depth = (target < DEPTH_MIN) | (target > DEPTH_MAX)

    # Edge detection
    r = params["edge_radius"]
    if r > 0:
        k        = 2 * r + 1
        max_d    = maximum_filter(raw_depth, size=k, mode="constant", cval=0.0)
        min_d    = minimum_filter(raw_depth, size=k, mode="constant", cval=0.0)
        edge_map = (min_d < DEPTH_MIN) | ((max_d - min_d) > params["edge_threshold"])
        not_edge = ~edge_map[py_c, px_c]
    else:
        not_edge = np.ones(N, dtype=bool)

    # Back-facing test: ray from cam to vertex should oppose vertex normal
    cam_pos = camera_world_pos(V)                       # world-space cam centre
    rays    = verts - cam_pos[None, :]                  # [N, 3]
    rays   /= np.linalg.norm(rays, axis=1, keepdims=True) + 1e-8
    dot     = (normals * rays).sum(axis=1)
    is_facing = dot < -0.1

    # SDF carve condition: vertex is in empty air (closer to camera than sensor surface)
    carve = depth_w < (target + params["forward_offset"] - SDF_THRESHOLD)

    return valid_proj & ~bad_depth & not_edge & is_facing & carve


# ── PNG render (same technique as render_figuresba.py) ───────────────────────

RENDER_AZIM  = 270
RENDER_ELEV  = 60
RENDER_BG    = "#1a1a2e"
LIGHT_DIR    = np.array([0.4, 0.9, 0.3], dtype=np.float64)
LIGHT_DIR   /= np.linalg.norm(LIGHT_DIR)
AMBIENT      = 0.25
DIFFUSE      = 0.75


def _view_matrix(elev_deg: float, azim_deg: float) -> np.ndarray:
    e, a = np.radians(elev_deg), np.radians(azim_deg)
    fwd  = -np.array([np.cos(e)*np.sin(a), np.sin(e), np.cos(e)*np.cos(a)])
    fwd /= np.linalg.norm(fwd)
    up   = np.array([0., 1., 0.]) if abs(fwd[1]) < 0.99 else np.array([0., 0., -1.])
    right = np.cross(up, fwd); right /= np.linalg.norm(right)
    return np.stack([right, np.cross(fwd, right), fwd])   # (3,3)


def render_png(
    path:     Path,
    verts:    np.ndarray,   # [N, 3]
    faces:    np.ndarray,   # [M, 3]
    cut_by:   np.ndarray,   # [N] int bitmask
    cam_vis:  list,         # [(name, color, world_pos), …]
    dpi:      int = 180,
):
    """Render cutter visualization to PNG using the standard figuresba pipeline."""
    R = _view_matrix(RENDER_ELEV, RENDER_AZIM)

    vc      = verts - verts.mean(axis=0)
    vv      = (R @ vc.T).T
    proj    = vv[:, :2]
    view_z  = vv[:, 2]

    # per-face view depth for painter sort
    face_vz = view_z[faces].mean(axis=1)
    order   = np.argsort(face_vz)

    # face normals (world-space) for Lambertian shading
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    shade = np.clip(AMBIENT + DIFFUSE * np.abs(fn @ LIGHT_DIR), 0., 1.)

    # per-face cut color: any vertex cut overrides kept; multi > single
    face_cut = cut_by[faces]                   # [M, 3] bitmask per face vertex
    face_mask = face_cut.any(axis=1)           # [M] bool — at least one vertex cut
    face_bits = face_cut[:, 0] | face_cut[:, 1] | face_cut[:, 2]  # union of vertex bits

    base_rgb = np.full((len(faces), 3), KEPT_COLOR, dtype=np.float64)

    for ci, (_, color, _) in enumerate(cam_vis):
        single = (face_bits == (1 << ci))
        base_rgb[single] = color

    multi_faces = face_mask & ((face_bits & (face_bits - 1)) != 0)
    base_rgb[multi_faces] = MULTI_COLOR

    shaded  = np.clip(base_rgb[order] * shade[order, None], 0., 1.)
    colors  = np.concatenate([shaded, np.ones((len(shaded), 1))], axis=1)

    fig, ax = plt.subplots(figsize=(6, 8), facecolor=RENDER_BG)
    ax.set_facecolor(RENDER_BG)
    col = PolyCollection(proj[faces[order]], facecolors=colors,
                         edgecolors="none", linewidths=0)
    ax.add_collection(col)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")

    # legend patches
    from matplotlib.patches import Patch
    legend_items = [Patch(facecolor=KEPT_COLOR, label="kept")]
    for name, color, _ in cam_vis:
        legend_items.append(Patch(facecolor=color, label=f"cut by {name}"))
    legend_items.append(Patch(facecolor=MULTI_COLOR, label="cut by both"))
    ax.legend(handles=legend_items, loc="lower right",
              fontsize=8, framealpha=0.3,
              labelcolor="white", facecolor="#333355", edgecolor="none")

    ax.set_title(
        f"Post-MC cutter  |  azim={RENDER_AZIM}°  elev={RENDER_ELEV}°",
        color="white", fontsize=10, pad=6,
    )

    plt.tight_layout(pad=0.4)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight", facecolor=RENDER_BG)
    plt.close(fig)
    print(f"[OK] wrote {path}")


# ── OBJ export ────────────────────────────────────────────────────────────────

def write_colored_obj(
    path: Path,
    verts:     np.ndarray,          # [N, 3]
    faces:     np.ndarray,          # [M, 3]
    vert_rgb:  np.ndarray,          # [N, 3]  values 0-1
    cam_data:  list,                # list of (name, color, world_pos)
):
    """Write OBJ with 'v x y z r g b' vertex colours + camera marker pyramids."""
    cam_axis_len = float(np.linalg.norm(verts.max(0) - verts.min(0))) * 0.04

    lines = []
    lines.append("# RIFTCast cutter visualisation\n")
    lines.append(f"# Vertices: {len(verts)}  Faces: {len(faces)}\n")
    lines.append("#\n")
    lines.append("# vertex colour key:\n")
    lines.append("#   gray    = kept\n")
    lines.append("#   magenta = cut by multiple cameras\n")
    for i, (name, color, _) in enumerate(cam_data):
        r, g, b = color
        lines.append(f"#   ({r:.2f},{g:.2f},{b:.2f}) = cut by {name}\n")
    lines.append("\n")

    lines.append("o mesh\n")
    for v, c in zip(verts, vert_rgb):
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}"
                     f" {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")
    vertex_offset = len(verts)

    for f in faces:
        lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")

    # Camera markers: small pyramids pointing in the camera's look direction
    for cam_name, cam_color, world_pos in cam_data:
        cr, cg, cb = cam_color
        tip_idx = vertex_offset
        lines.append(f"\no cam_{cam_name}\n")

        # Four base corners of the pyramid, spread perpendicular to cam-to-scene centre
        scene_centre = verts.mean(0)
        forward = scene_centre - world_pos
        forward /= np.linalg.norm(forward) + 1e-8

        # Build an arbitrary perpendicular basis
        up    = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(forward, up)) > 0.9:
            up = np.array([1.0, 0.0, 0.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up2   = np.cross(right, forward)

        half  = cam_axis_len
        base  = world_pos + forward * 2.5 * half
        tip   = world_pos

        corners = [
            base + right * half + up2 * half,
            base - right * half + up2 * half,
            base - right * half - up2 * half,
            base + right * half - up2 * half,
        ]

        lines.append(f"v {tip[0]:.6f} {tip[1]:.6f} {tip[2]:.6f}"
                     f" {cr:.4f} {cg:.4f} {cb:.4f}\n")
        for c in corners:
            lines.append(f"v {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}"
                         f" {cr:.4f} {cg:.4f} {cb:.4f}\n")

        # Tip index = tip_idx+1 (1-based), base corners = tip_idx+2..tip_idx+5
        t  = tip_idx + 1
        b0, b1, b2, b3 = t + 1, t + 2, t + 3, t + 4
        lines.append(f"f {t} {b0} {b1}\n")
        lines.append(f"f {t} {b1} {b2}\n")
        lines.append(f"f {t} {b2} {b3}\n")
        lines.append(f"f {t} {b3} {b0}\n")
        lines.append(f"f {b0} {b1} {b2} {b3}\n")

        vertex_offset += 5

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.writelines(lines)

    print(f"[OK] wrote {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config",    required=True,
                    help="path to config JSON (e.g. configs/config_vci.json)")
    ap.add_argument("--vertices",  required=True,
                    help="path to vertices.bin  (float32 Nx3)")
    ap.add_argument("--faces",     required=True,
                    help="path to faces.bin     (int64 Mx3)")
    ap.add_argument("--frame",     type=int, default=0,
                    help="frame index for loading depth (default 0)")
    ap.add_argument("--out",       default=None,
                    help="output OBJ path (default: next to vertices.bin)")
    args = ap.parse_args()

    # ── load config ──────────────────────────────────────────────────────────
    cfg_path = resolve(args.config)
    with open(cfg_path) as f:
        cfg = json.load(f)

    dataset_rel  = cfg["dataset"]["path"]
    dataset_path = resolve(dataset_rel)
    flip_images  = cfg["dataset"].get("flip_images", False)
    to_world_raw = cfg["dataset"].get("to_world", list(np.eye(4).flatten()))
    to_world     = np.array(to_world_raw, dtype=np.float64).reshape(4, 4)

    depth_scale  = cfg.get("depth", {}).get("scale", 1000.0)
    mode_raw     = cfg.get("reconstructor", {}).get("depth_fusion_mode", "realdata").lower()
    mode_key     = mode_raw.replace(" ", "").replace("_", "")
    # normalise key: strip underscores for lookup
    preset_lookup = {k.replace("_","").lower(): v for k, v in PRESETS.items()}
    preset       = preset_lookup.get(mode_key.replace("_",""), None)

    if preset is None:
        print(f"[WARN] depth_fusion_mode '{mode_raw}' → no depth; nothing to cut.")
        sys.exit(0)

    cutter_enabled = cfg.get("depth", {}).get("enable_cutter", True)
    if not cutter_enabled:
        print("[WARN] enable_cutter=false in config; no cutting would happen.")

    # ── load calibration ─────────────────────────────────────────────────────
    calib_path = dataset_path / "calibration_dome.json"
    if not calib_path.exists():
        calib_path = resolve("configs/calibration_dome.json")
    with open(calib_path) as f:
        calib = json.load(f)

    depth_cams = [c for c in calib["cameras"]
                  if c["camera_id"].startswith("D") and c.get("camera_type", "") != "mono"]
    if not depth_cams:
        print("[WARN] no depth cameras found in calibration.")
        sys.exit(0)

    print(f"[INFO] depth cameras: {[c['camera_id'] for c in depth_cams]}")

    # ── load mesh ────────────────────────────────────────────────────────────
    verts, faces = load_mesh(resolve(args.vertices), resolve(args.faces))
    normals      = compute_normals(verts, faces)
    N            = len(verts)
    print(f"[INFO] mesh: {N} vertices, {len(faces)} faces")

    # ── frame string for depth path ──────────────────────────────────────────
    frame_str = f"frame_{args.frame:05d}"

    # ── per-camera cut mask ───────────────────────────────────────────────────
    cut_by  = np.zeros(N, dtype=np.int32)   # bitmask: bit i = cut by cam i
    cam_vis = []                             # (name, color, world_pos) for OBJ markers

    for ci, cam_json in enumerate(depth_cams):
        cam_id  = cam_json["camera_id"]
        intr    = cam_json["intrinsics"]
        W, H    = intr["resolution"]         # [width, height] order in JSON
        fx      = float(intr["camera_matrix"][0])
        fy      = float(intr["camera_matrix"][4])
        cx      = float(intr["camera_matrix"][2])
        cy      = float(intr["camera_matrix"][5])

        extr_flat = cam_json["extrinsics"]["view_matrix"]
        V         = build_view(extr_flat, to_world)
        P_proj    = build_projection_gl(fx, fy, cx, cy, W, H)
        VP        = P_proj @ V

        world_pos = camera_world_pos(V)
        color     = CAM_COLORS[ci % len(CAM_COLORS)]
        cam_vis.append((cam_id, color, world_pos))

        # ── load depth ───────────────────────────────────────────────────────
        pth_rgb   = dataset_path / frame_str / "rgb"   / f"{cam_id}.pth"
        pth_depth = dataset_path / frame_str / "depth" / f"{cam_id}.pth"
        pth_path  = pth_rgb if pth_rgb.exists() else pth_depth

        if not pth_path.exists():
            print(f"[SKIP] {cam_id}: no depth file found at {pth_path}")
            continue

        try:
            depth = load_depth_pth(pth_path, H, W, flip_images)
        except Exception as e:
            print(f"[SKIP] {cam_id}: depth load failed – {e}")
            continue

        print(f"[INFO] {cam_id}: depth {depth.shape}, "
              f"range [{depth[depth>0].min():.2f}, {depth.max():.2f}] m")

        # ── simulate cut ─────────────────────────────────────────────────────
        mask = simulate_cut(verts, normals, VP, V, depth, preset)
        n_cut = int(mask.sum())
        print(f"[INFO] {cam_id}: would cut {n_cut} / {N} vertices "
              f"({100*n_cut/N:.1f}%)")

        cut_by[mask] |= (1 << ci)

    # ── assign colours ────────────────────────────────────────────────────────
    vert_rgb = np.full((N, 3), KEPT_COLOR, dtype=np.float64)

    # single-camera cuts
    for ci, (_, color, _) in enumerate(cam_vis):
        single = (cut_by == (1 << ci))
        vert_rgb[single] = color

    # multi-camera cuts
    multi = (cut_by != 0) & ((cut_by & (cut_by - 1)) != 0)
    vert_rgb[multi] = MULTI_COLOR

    n_kept  = int((cut_by == 0).sum())
    n_multi = int(multi.sum())
    print(f"\n[SUMMARY] kept={n_kept}  multi-cut={n_multi}")
    for ci, (name, color, _) in enumerate(cam_vis):
        single = int((cut_by == (1 << ci)).sum())
        print(f"  {name}: {single} vertices ({color[0]:.2f},{color[1]:.2f},{color[2]:.2f})")

    # ── write OBJ + PNG ───────────────────────────────────────────────────────
    if args.out:
        out_stem = Path(args.out).with_suffix("")
    else:
        out_stem = Path(args.vertices).parent / "cutter_viz"

    write_colored_obj(out_stem.with_suffix(".obj"), verts, faces, vert_rgb, cam_vis)
    render_png(out_stem.with_suffix(".png"), verts, faces, cut_by, cam_vis)


if __name__ == "__main__":
    main()
