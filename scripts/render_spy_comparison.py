#!/usr/bin/env python3
"""
render_spy_comparison.py – pixel-aligned front-view renders of the real VCI scene
for the baseline / post-MC cutter / pre-MC TSDF spy comparison figure.

Meshes:
  baseline : runtime2/outputs/vci_real__original_none/frame_00000
  tsdf     : runtime2/outputs/vci_real__mine_tsdf_real/frame_00000
  cutter   : baseline mesh with the post-MC cutter applied (Python replica of
             GeometryModule.cpp via scripts/visualize_cutter.py, preset "real")

All three are rendered with the same camera, colour range, and figure size so
spy-zoom coordinates transfer 1:1 between panels.

Usage:
    python scripts/render_spy_comparison.py                 # final renders
    python scripts/render_spy_comparison.py --scout         # angle grid to pick the view
    python scripts/render_spy_comparison.py --azim 200 --elev 10
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import visualize_cutter as vc  # noqa: E402

BASELINE_DIR = ROOT / "runtime2/outputs/vci_real__original_none/frame_00000"
TSDF_DIR     = ROOT / "runtime2/outputs/vci_real__mine_tsdf_real/frame_00000"
CONFIG       = ROOT / "runtime2/configs/vci_real__mine_tsdf_real.json"
OUT          = ROOT / "output/spy"

CMAP      = "plasma"
AMBIENT   = 0.25
DIFFUSE   = 0.75
LIGHT_DIR = np.array([0.4, 0.9, 0.3]) / np.linalg.norm([0.4, 0.9, 0.3])
BG        = "#1a1a2e"
DPI       = 200


def load_mesh(d: Path):
    v = np.fromfile(d / "vertices.bin", dtype=np.float32).reshape(-1, 3).astype(np.float64)
    f = np.fromfile(d / "faces.bin", dtype=np.int64).reshape(-1, 3)
    return v, f


def view_matrix(elev_deg, azim_deg):
    e, a = np.radians(elev_deg), np.radians(azim_deg)
    fwd = -np.array([np.cos(e) * np.sin(a), np.sin(e), np.cos(e) * np.cos(a)])
    fwd /= np.linalg.norm(fwd)
    up = np.array([0., 1., 0.]) if abs(fwd[1]) < 0.99 else np.array([0., 0., -1.])
    right = np.cross(up, fwd); right /= np.linalg.norm(right)
    return np.stack([right, np.cross(fwd, right), fwd])


def render_ax(ax, verts, faces, R, center, y_min, y_max, lims=None):
    vv = (R @ (verts - center).T).T
    proj, view_z = vv[:, :2], vv[:, 2]

    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn /= (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12)

    order = np.argsort(view_z[faces].mean(axis=1))
    y_norm = (verts[faces, 1].mean(axis=1) - y_min) / max(y_max - y_min, 1e-8)
    shade = np.clip(AMBIENT + DIFFUSE * np.abs(fn @ LIGHT_DIR), 0., 1.)

    rgb = matplotlib.colormaps[CMAP](y_norm[order])[:, :3]
    colors = np.concatenate([np.clip(rgb * shade[order, None], 0., 1.),
                             np.ones((len(order), 1))], axis=1)

    ax.add_collection(PolyCollection(proj[faces[order]], facecolors=colors,
                                     edgecolors="none", linewidths=0))
    if lims is not None:
        ax.set_xlim(lims[0]); ax.set_ylim(lims[1])
    else:
        ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(BG)


def cutter_mesh(verts, faces):
    """Apply the post-MC cutter (preset 'real') to a mesh; returns reduced faces."""
    with open(CONFIG) as f:
        cfg = json.load(f)
    dataset_path = ROOT / cfg["dataset"]["path"]
    flip_images = cfg["dataset"].get("flip_images", False)
    to_world = np.array(cfg["dataset"]["to_world"], dtype=np.float64).reshape(4, 4)
    preset = vc.PRESETS["real"]

    with open(dataset_path / "calibration_dome.json") as f:
        calib = json.load(f)
    depth_cams = [c for c in calib["cameras"]
                  if c["camera_id"].startswith("D") and c.get("camera_type", "") != "mono"]

    normals = vc.compute_normals(verts, faces)
    cut = np.zeros(len(verts), dtype=bool)

    for cam in depth_cams:
        cam_id = cam["camera_id"]
        intr = cam["intrinsics"]
        W, H = intr["resolution"]
        fx, fy = float(intr["camera_matrix"][0]), float(intr["camera_matrix"][4])
        cx, cy = float(intr["camera_matrix"][2]), float(intr["camera_matrix"][5])
        V = vc.build_view(cam["extrinsics"]["view_matrix"], to_world)
        VP = vc.build_projection_gl(fx, fy, cx, cy, W, H) @ V

        pth = dataset_path / "frame_00000" / "rgb" / f"{cam_id}.pth"
        depth = vc.load_depth_pth(pth, H, W, flip_images)
        mask = vc.simulate_cut(verts, normals, VP, V, depth, preset)
        print(f"  {cam_id}: cuts {mask.sum()} / {len(verts)} vertices")
        cut |= mask

    keep = ~cut[faces].any(axis=1)
    print(f"  faces kept: {keep.sum()} / {len(faces)}")
    return faces[keep]


def depth_cam_world_pos(cam_id: str):
    """Return (eye world position, GL view rotation) for a calibrated depth cam."""
    with open(CONFIG) as f:
        cfg = json.load(f)
    dataset_path = ROOT / cfg["dataset"]["path"]
    to_world = np.array(cfg["dataset"]["to_world"], dtype=np.float64).reshape(4, 4)
    with open(dataset_path / "calibration_dome.json") as f:
        calib = json.load(f)
    cam = next(c for c in calib["cameras"] if c["camera_id"] == cam_id)
    V = vc.build_view(cam["extrinsics"]["view_matrix"], to_world)
    return vc.camera_world_pos(V), V


def depth_cam_view(cam_id: str) -> np.ndarray:
    """Render rotation matching a calibrated depth camera's viewing direction."""
    _, V = depth_cam_world_pos(cam_id)
    # GL eye coords: rows = right, up, -forward. The painter sort draws large
    # view_z last, so keep +row2 (toward the camera) as the depth axis: faces
    # nearest the camera are painted on top, giving the camera's own view.
    return np.stack([V[0, :3], V[1, :3], V[2, :3]])


def lookat_R(eye, target, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """GL-style look-at rotation; rows = [right, up, -forward] (camera on top)."""
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    s = np.cross(f, np.asarray(up, float)); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    return np.stack([s, u, -f])


def orbit_eye(eye, target, angle_deg, axis):
    """Rotate eye around target about `axis` (Rodrigues); keeps height along axis."""
    eye = np.asarray(eye, float); target = np.asarray(target, float)
    k = np.asarray(axis, float); k /= np.linalg.norm(k)
    d = eye - target
    th = np.radians(angle_deg)
    c, si = np.cos(th), np.sin(th)
    rot = d * c + np.cross(k, d) * si + k * (k @ d) * (1.0 - c)
    return target + rot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scout", action="store_true", help="render angle grid of cutter mesh")
    ap.add_argument("--azim", type=float, default=180.0)
    ap.add_argument("--elev", type=float, default=10.0)
    ap.add_argument("--cam", default=None,
                    help="render from a depth camera's view (D003L or D005Z)")
    ap.add_argument("--orbit-scout", action="store_true",
                    help="contact sheet orbiting around target at D003L height")
    ap.add_argument("--orbit", type=float, default=None,
                    help="final render: orbit angle (deg) around target at D003L height")
    ap.add_argument("--orbit-base", default="D003L",
                    help="depth cam whose height/distance defines the orbit (default D003L)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    vb, fb = load_mesh(BASELINE_DIR)
    vt, ft = load_mesh(TSDF_DIR)
    print("baseline:", len(vb), "verts;  tsdf:", len(vt), "verts")
    print("applying post-MC cutter to baseline mesh...")
    fc = cutter_mesh(vb, fb)

    y_all = np.concatenate([vb[:, 1], vt[:, 1]])
    y_min, y_max = y_all.min(), y_all.max()
    center = vb.mean(axis=0)  # same center for all panels -> pixel alignment

    if args.scout:
        angles = [(e, a) for e in (5, 20) for a in (0, 45, 90, 135, 180, 225, 270, 315)]
        fig, axes = plt.subplots(2, 8, figsize=(32, 10), facecolor=BG)
        for ax, (e, a) in zip(axes.ravel(), angles):
            render_ax(ax, vb, fc, view_matrix(e, a), center, y_min, y_max)
            ax.set_title(f"elev={e} azim={a}", color="white", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / "_scout_cutter.png", dpi=80, facecolor=BG)
        print("wrote", OUT / "_scout_cutter.png")
        return

    if args.orbit_scout:
        eye0, V0 = depth_cam_world_pos(args.orbit_base)
        up_axis = V0[1, :3]   # camera up in world space = the scene's true vertical
        orbit_angles = [-60, -45, -30, -15, 0, 15, 30, 45, 60]
        fig, axes = plt.subplots(1, len(orbit_angles),
                                 figsize=(4 * len(orbit_angles), 7), facecolor=BG)
        for ax, ang in zip(axes, orbit_angles):
            R = lookat_R(orbit_eye(eye0, center, ang, up_axis), center, up=up_axis)
            render_ax(ax, vb, fc, R, center, y_min, y_max)
            ax.set_title(f"orbit {ang:+d}°", color="white", fontsize=14)
        fig.tight_layout()
        fig.savefig(OUT / "_orbit_scout.png", dpi=90, facecolor=BG)
        print("wrote", OUT / "_orbit_scout.png")
        return

    if args.orbit is not None:
        eye0, V0 = depth_cam_world_pos(args.orbit_base)
        up_axis = V0[1, :3]
        R = lookat_R(orbit_eye(eye0, center, args.orbit, up_axis), center, up=up_axis)
    elif args.cam:
        R = depth_cam_view(args.cam)
    else:
        R = view_matrix(args.elev, args.azim)

    # shared limits from the baseline so all three panels are pixel-aligned
    vv = (R @ (vb - center).T).T
    pad = 0.04 * (vv[:, :2].max(0) - vv[:, :2].min(0))
    lims = ((vv[:, 0].min() - pad[0], vv[:, 0].max() + pad[0]),
            (vv[:, 1].min() - pad[1], vv[:, 1].max() + pad[1]))

    panels = [
        ("01_baseline_front.png", vb, fb),
        ("02_cutter_front.png",   vb, fc),
        ("03_tsdf_front.png",     vt, ft),
    ]
    for name, v, f in panels:
        fig, ax = plt.subplots(figsize=(5, 7), facecolor=BG)
        render_ax(ax, v, f, R, center, y_min, y_max, lims=lims)
        fig.tight_layout(pad=0.1)
        fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        print("wrote", OUT / name)

    # cutter mesh as OBJ for inspection
    with open(OUT / "cutter_only_front.obj", "w") as fh:
        for v in vb:
            fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for f in fc:
            fh.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")
    print("wrote", OUT / "cutter_only_front.obj")


if __name__ == "__main__":
    main()
