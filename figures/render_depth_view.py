#!/usr/bin/env python3
"""Render a depth-shaded view of a reconstruction mesh from a fixed viewpoint."""
import argparse, math
from pathlib import Path
import numpy as np
import trimesh
import pyrender
import matplotlib.cm as cm


def load_mesh(path):
    m = trimesh.load(str(path), process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate([g for g in m.geometry.values()])
    return m


def render_depth(mesh, az, el, dist_scale, yfov, res, center=None, eye=None):
    if center is None:
        center = mesh.bounds.mean(axis=0)
    if eye is None:
        extent = float(np.linalg.norm(mesh.extents))
        dist = extent * dist_scale
        a, e = math.radians(az), math.radians(el)
        eye = center + dist * np.array([
            math.cos(e) * math.sin(a),
            math.sin(e),
            math.cos(e) * math.cos(a),
        ])

    # camera pose (look at center)
    fwd = center - eye; fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(fwd, up)) > 0.95:
        up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    pose = np.eye(4)
    pose[:3, 0] = right; pose[:3, 1] = up; pose[:3, 2] = -fwd; pose[:3, 3] = eye

    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1, 1, 1])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    cam = pyrender.PerspectiveCamera(yfov=yfov)
    scene.add(cam, pose=pose)
    r = pyrender.OffscreenRenderer(res, res)
    depth = r.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
    r.delete()
    return depth


def colormap_depth(depth, cmap, vmin, vmax):
    mask = depth > 0
    out = np.zeros((*depth.shape, 4), dtype=np.uint8)
    if mask.any():
        if vmin is None: vmin = depth[mask].min()
        if vmax is None: vmax = depth[mask].max()
        norm = np.clip((depth - vmin) / max(vmax - vmin, 1e-6), 0, 1)
        rgba = (cm.get_cmap(cmap)(1.0 - norm) * 255).astype(np.uint8)  # near=bright
        out[mask] = rgba[mask]
        out[..., 3] = np.where(mask, 255, 0)
    return out, vmin, vmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--az", type=float, default=20.0)
    ap.add_argument("--el", type=float, default=15.0)
    ap.add_argument("--dist", type=float, default=2.0, help="distance = dist * mesh diagonal")
    ap.add_argument("--yfov", type=float, default=0.6)
    ap.add_argument("--res", type=int, default=1200)
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--center", default=None, help="x,y,z look-at point (else mesh bbox centre)")
    ap.add_argument("--eye", default=None, help="x,y,z camera position (overrides az/el/dist)")
    args = ap.parse_args()

    center = np.array([float(v) for v in args.center.split(",")]) if args.center else None
    eye = np.array([float(v) for v in args.eye.split(",")]) if args.eye else None

    from PIL import Image
    mesh = load_mesh(args.mesh)
    depth = render_depth(mesh, args.az, args.el, args.dist, args.yfov, args.res,
                         center=center, eye=eye)
    rgba, vmin, vmax = colormap_depth(depth, args.cmap, args.vmin, args.vmax)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba).save(str(args.out))
    print(f"{args.out}  depth range [{vmin:.3f},{vmax:.3f}]  hit%={100*(depth>0).mean():.1f}")


if __name__ == "__main__":
    main()
