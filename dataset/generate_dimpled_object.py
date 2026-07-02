#!/usr/bin/env python3
"""Synthetic 'dimpled sphere' dataset for the depth-camera-count ablation."""
import argparse
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
from skimage import measure

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def fibonacci_sphere(samples, radius=1.0):
    pts = []
    phi = math.pi * (math.sqrt(5.0) - 1.0)
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2 if samples > 1 else 1.0
        r = math.sqrt(max(0.0, 1 - y * y))
        th = phi * i
        pts.append((math.cos(th) * r * radius, y * radius, math.sin(th) * r * radius))
    return pts


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def farthest_point_sphere(n, radius=1.0, n_cand=4000, seed_idx=0):
    """Deterministic, well-spread, NESTED camera positions on a sphere.

    Greedy farthest-point sampling over a dense fibonacci candidate cloud. Because the greedy
    order is deterministic, the first N points are always a prefix of the first M (M>N): adding
    a depth camera keeps all existing ones and only appends new well-separated ones. This makes
    the depth-count sweep strictly additive (a naive fibonacci prefix would clump at one pole)."""
    if n <= 0:
        return []
    cand = np.array(fibonacci_sphere(n_cand, radius=1.0), dtype=np.float64)
    chosen = [seed_idx]
    dist = np.linalg.norm(cand - cand[seed_idx], axis=1)
    for _ in range(1, min(n, n_cand)):
        i = int(np.argmax(dist))
        chosen.append(i)
        dist = np.minimum(dist, np.linalg.norm(cand - cand[i], axis=1))
    return [tuple(p) for p in (cand[chosen] * radius)]


def look_at_cv(cam_pos, target):
    z = normalize(target - cam_pos)
    up = np.array([0.0, 1.0, 0.0], np.float32)
    if abs(float(np.dot(z, up))) > 0.97:
        up = np.array([0.0, 0.0, 1.0], np.float32)
    x = normalize(np.cross(z, up))
    y = -normalize(np.cross(x, z))
    R = np.stack([x, y, z], 0).astype(np.float32)
    t = (-R @ cam_pos.reshape(3, 1)).reshape(3).astype(np.float32)
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = R
    view[:3, 3] = t
    return view


def make_pockets(sphere_r, n_pockets, pocket_r, pocket_offset):
    """Pocket spheres centred just outside the surface, carving spherical-cap bowls."""
    centers = []
    for x, y, z in fibonacci_sphere(n_pockets, radius=sphere_r + pocket_offset):
        centers.append(np.array([x, y, z], np.float32))
    return centers, pocket_r


def object_sdf(p, sphere_r, centers, pocket_r):
    """p: (...,3). Object = sphere MINUS pocket spheres. Negative inside."""
    d = np.linalg.norm(p, axis=-1) - sphere_r          
    for c in centers:
        d = np.maximum(d, pocket_r - np.linalg.norm(p - c, axis=-1))  # subtract pocket
    return d


def gt_mesh(sphere_r, centers, pocket_r, grid=256, half=0.75):
    xs = np.linspace(-half, half, grid).astype(np.float32)
    P = np.stack(np.meshgrid(xs, xs, xs, indexing="ij"), -1)
    vol = object_sdf(P, sphere_r, centers, pocket_r)
    verts, faces, _, _ = measure.marching_cubes(vol, level=0.0)
    # map voxel index -> world coords
    verts = verts / (grid - 1) * (2 * half) - half
    return verts.astype(np.float32), faces.astype(np.int64)


def save_obj(path, V, F):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# dimpled sphere GT\n")
        for v in V:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in F:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


def render_mask(cam_pos, ray_w, sphere_r):
    """Silhouette = ray-sphere hit (the smooth hull; pockets don't change the outline)."""
    o = cam_pos.reshape(1, 1, 3)
    b = np.sum(ray_w * o, axis=2)
    c = np.sum(o * o, axis=2) - sphere_r * sphere_r
    disc = b * b - c
    return (disc >= 0.0).astype(np.uint8) * 255


def render_depth(cam_pos, ray_w, ray_cam, sphere_r, centers, pocket_r, bg=5.0, steps=256):
    """March ONLY the rays inside the sphere silhouette, over [t_near, t_far] of the sphere."""
    H, W = ray_w.shape[:2]
    o = cam_pos.astype(np.float32)
    depth = np.full((H, W), bg, np.float32)

    b = (ray_w * o.reshape(1, 1, 3)).sum(2)          # d . o  (d normalized)
    c = float((o * o).sum()) - sphere_r * sphere_r
    disc = b * b - c
    inside = disc >= 0.0
    if not inside.any():
        return depth

    idx = np.where(inside)
    d = ray_w[idx]                                   # (M,3)
    rcz = ray_cam[..., 2][idx]                       # (M,)
    bb = b[idx]
    sq = np.sqrt(disc[idx])
    t_near = -bb - sq
    t_far = -bb + sq
    M = len(t_near)
    dt = (t_far - t_near) / steps                    # per-ray step
    t = t_near.copy()
    hit_t = np.full(M, np.inf, np.float32)
    prev = np.ones(M, np.float32)
    for _ in range(steps):
        p = o.reshape(1, 3) + t[:, None] * d
        sdf = object_sdf(p, sphere_r, centers, pocket_r)
        cross = (prev > 0) & (sdf <= 0) & ~np.isfinite(hit_t)
        hit_t[cross] = t[cross]
        prev = sdf
        t += dt
    dv = np.full(M, bg, np.float32)
    hh = np.isfinite(hit_t)
    dv[hh] = hit_t[hh] * rcz[hh]
    depth[idx] = dv
    return np.clip(depth, 0.0, 65.0)


def save_depth_png_mm(path, depth_m):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16))


def save_mask_bin(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(mask.astype(np.uint8).tobytes())


def save_rgb(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.repeat(mask[:, :, None], 3, axis=2).astype(np.uint8)
    cv2.imwrite(str(path), rgb)


def make_config(dataset_root, config_path):
    cfg = {
        "type": "VCI_REAL", "version": "2.1",
        "dataset": {"frame_count": 1, "start_frame": 0, "path": str(dataset_root),
                    "camera_path": "calibration_dome.json", "flip_images": False, "flip_masks": False,
                    "to_world": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]},
        "depth": {"has_depth": True, "scale": 1000.0, "extension": ".png", "enable_cutter": False},
        "reconstructor": {"gpu": 0, "level": 9, "partial_masks": False, "depth_fusion_mode": "synthetic", "smoothing": False},
        "renderer": {"gpu": 0}, "inpainting": {"enable": False, "path": ""},
        "volume": {"position": [0, 0, 0], "scale": 1.6},
        "server": {"ip": "127.0.0.1", "port": 25565},
    }
    config_path.write_text(json.dumps(cfg, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--config-out", type=Path, required=True)
    ap.add_argument("--gt-out", type=Path, default=None)
    ap.add_argument("--num-rgb", type=int, default=60)
    ap.add_argument("--num-depth", type=int, default=8)
    ap.add_argument("--sphere-radius", type=float, default=0.5)
    ap.add_argument("--cam-radius", type=float, default=2.5)
    ap.add_argument("--num-pockets", type=int, default=12)
    ap.add_argument("--pocket-radius", type=float, default=0.19)
    ap.add_argument("--pocket-offset", type=float, default=0.10)
    # NOTE: DatasetImporter only loads mask .bin files whose byte size is exactly
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fx", type=float, default=1400.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    # DatasetImporter accepts mask .bin only at these byte sizes (else the camera is skipped).
    if args.width * args.height not in (5328 * 4608, 1920 * 1080):
        raise SystemExit(
            f"render resolution {args.width}x{args.height} is not loadable by DatasetImporter "
            f"(must be 5328x4608 or 1920x1080); masks would be silently dropped -> empty hull")

    root = args.dataset_root.resolve()
    if root.exists() and args.force:
        shutil.rmtree(root)
    frame = root / "frame_00000"
    for s in ("rgb", "mask", "depth"):
        (frame / s).mkdir(parents=True, exist_ok=True)

    centers, pr = make_pockets(args.sphere_radius, args.num_pockets, args.pocket_radius, args.pocket_offset)

    if args.gt_out:
        V, F = gt_mesh(args.sphere_radius, centers, pr)
        save_obj(args.gt_out, V, F)
        print(f"GT mesh: {len(V)} verts -> {args.gt_out}")

    W, H = args.width, args.height
    fx = fy = args.fx
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    ray_cam = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], 2)
    ray_cam /= np.linalg.norm(ray_cam, axis=2, keepdims=True)

    cams = []
    for i, (x, y, z) in enumerate(fibonacci_sphere(args.num_rgb, radius=args.cam_radius)):
        cams.append({"id": f"C{i:03d}", "type": "rgb", "pos": np.array([x, y, z], np.float32)})

    for i, (x, y, z) in enumerate(farthest_point_sphere(args.num_depth, radius=args.cam_radius)):
        cams.append({"id": f"D{i:03d}", "type": "depth", "pos": np.array([x, y, z], np.float32)})

    target = np.zeros(3, np.float32)
    calib = []
    for cam in cams:
        view = look_at_cv(cam["pos"], target)
        R = view[:3, :3]
        ray_w = np.einsum("hwc,cd->hwd", ray_cam, R)
        cid = cam["id"]
        if cam["type"] == "rgb":
            mask = render_mask(cam["pos"], ray_w, args.sphere_radius)
            save_mask_bin(frame / "mask" / f"mask_{cid}.bin", mask)
            save_rgb(frame / "rgb" / f"{cid}.jpg", mask)
        else:
            save_rgb(frame / "rgb" / f"{cid}.jpg", np.full((H, W), 255, np.uint8))
            depth = render_depth(cam["pos"], ray_w, ray_cam, args.sphere_radius, centers, pr)
            save_depth_png_mm(frame / "depth" / f"{cid}.png", depth)
        calib.append({
            "camera_id": cid, "camera_type": cam["type"],
            "intrinsics": {"resolution": [W, H], "camera_matrix": [fx, 0, cx, 0, fy, cy, 0, 0, 1]},
            "extrinsics": {"position": [float(c) for c in cam["pos"]], "view_matrix": view.reshape(-1).astype(float).tolist()},
        })

    (root / "calibration_dome.json").write_text(json.dumps({"cameras": calib}, indent=2))
    make_config(root, args.config_out.resolve())
    print(f"dataset: {args.num_rgb} RGB + {args.num_depth} depth, {args.num_pockets} pockets -> {root}")


if __name__ == "__main__":
    main()
