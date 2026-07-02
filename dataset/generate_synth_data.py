#!/usr/bin/env python3
# Source: https://stackoverflow.com/a/26127012 (Fnord, CC BY-SA 4.0)
import argparse
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
except ImportError:
    torch = None


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def fibonacci_sphere(samples: int, radius: float = 1.0) -> list:
    points = []
    phi = math.pi * (math.sqrt(5.0) - 1.0)
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2
        r = math.sqrt(max(0.0, 1 - y * y))
        theta = phi * i
        x = math.cos(theta) * r
        z = math.sin(theta) * r
        points.append((x * radius, y * radius, z * radius))
    return points


def fibonacci_hemisphere_posz(samples: int, radius: float = 1.0) -> list:
    """Evenly distributes points on the +Z hemisphere (z > 0), facing into the open face."""
    points = []
    phi = math.pi * (math.sqrt(5.0) - 1.0)
    for i in range(samples):
        # z goes from 1 down to ~0, always positive — full +Z hemisphere coverage
        z = 1.0 - (i / float(samples))
        r = math.sqrt(max(0.0, 1.0 - z * z))
        theta = phi * i
        x = math.cos(theta) * r
        y = math.sin(theta) * r
        points.append((x * radius, y * radius, z * radius))
    return points


def frontal_cone_depth_positions(samples: int, radius: float = 1.0, half_angle_deg: float = 20.0) -> list:
    """Depth cameras inside a NARROW cone around the +Z opening axis. Frontal views see the
    cavity and back wall head-on and never graze the side walls, so they open the aperture
    (deep air through the opening) without fabricating false-air/solid on the side walls."""
    import numpy as _np
    pts = []
    ha = math.radians(half_angle_deg)
    phi = math.pi * (math.sqrt(5.0) - 1.0)
    for i in range(samples):
        # spread within the cone: tilt grows mildly, azimuth by golden angle
        frac = 0.0 if samples == 1 else i / float(samples - 1)
        tilt = ha * math.sqrt(frac)            # 0 at center, up to half_angle
        az = phi * i
        z = math.cos(tilt)
        r = math.sin(tilt)
        x = math.cos(az) * r
        y = math.sin(az) * r
        pts.append((x * radius, y * radius, z * radius))
    return pts


def wall_facing_depth_positions(radius: float = 1.0) -> list:
    """Deliberate depth-camera positions so EACH inner wall of the open (+Z) box is seen
    roughly head-on. A camera offset to the opposite side of the opening looks straight at a
    wall's inner face through the aperture, giving it reliable surface evidence (no grazing).
    Returns positions on a sphere of the given radius (all with z>0, outside the +Z opening)."""
    import numpy as _np

    def norm(v):
        v = _np.asarray(v, dtype=float)
        return (v / _np.linalg.norm(v) * radius).tolist()

    pts = [
        norm([0.0, 0.0, 1.0]),    # back wall (-Z), head-on down the axis
        norm([0.0, -1.2, 1.0]),   # sees +Y inner wall (top)    — offset -Y
        norm([0.0, 1.2, 1.0]),    # sees -Y inner wall (bottom) — offset +Y
        norm([-1.2, 0.0, 1.0]),   # sees +X inner wall (right)  — offset -X
        norm([1.2, 0.0, 1.0]),    # sees -X inner wall (left)   — offset +X
        # diagonal fills for the wall regions near the aperture corners
        norm([-1.0, -1.0, 1.1]),
        norm([1.0, -1.0, 1.1]),
        norm([-1.0, 1.0, 1.1]),
        norm([1.0, 1.0, 1.1]),
        norm([0.0, 0.0, 1.0]),    # extra axial (deduped below if identical)
    ]
    # de-duplicate near-identical axial entries
    uniq = []
    for p in pts:
        if not any(_np.allclose(p, q, atol=1e-6) for q in uniq):
            uniq.append(p)
    return uniq


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-8:
        return v
    return v / n


def look_at_cv(camera_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    z_cam_world = normalize(target - camera_pos)

    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(z_cam_world, world_up))) > 0.97:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    x_cam_world = normalize(np.cross(z_cam_world, world_up))
    y_cam_world_up = np.cross(x_cam_world, z_cam_world)
    y_cam_world = -normalize(y_cam_world_up)

    R = np.stack([x_cam_world, y_cam_world, z_cam_world], axis=0).astype(np.float32)
    t = (-R @ camera_pos.reshape(3, 1)).reshape(3).astype(np.float32)

    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = R
    view[:3, 3] = t
    return view 


def ray_aabb_intersection(ray_o: np.ndarray, ray_d: np.ndarray, bmin: np.ndarray, bmax: np.ndarray):
    safe_d = np.where(np.abs(ray_d) < 1e-8, 1e-8, ray_d)
    inv_d = 1.0 / safe_d

    t0 = (bmin.reshape(1, 1, 3) - ray_o.reshape(1, 1, 3)) * inv_d
    t1 = (bmax.reshape(1, 1, 3) - ray_o.reshape(1, 1, 3)) * inv_d

    t_near = np.maximum.reduce(np.minimum(t0, t1), axis=2)
    t_far = np.minimum.reduce(np.maximum(t0, t1), axis=2)

    hit = t_far >= np.maximum(t_near, 0.0)
    t_hit = np.where(hit, np.where(t_near > 0.0, t_near, t_far), np.inf)
    return hit, t_hit


def generate_open_box_walls(xmin, xmax, ymin, ymax, zmin, zmax, thickness):
    walls = []

    walls.append((
        np.array([xmin, ymin, zmin], dtype=np.float32),
        np.array([xmin + thickness, ymax, zmax], dtype=np.float32),
    ))

    walls.append((
        np.array([xmax - thickness, ymin, zmin], dtype=np.float32),
        np.array([xmax, ymax, zmax], dtype=np.float32),
    ))

    walls.append((
        np.array([xmin, ymin, zmin], dtype=np.float32),
        np.array([xmax, ymin + thickness, zmax], dtype=np.float32),
    ))

    walls.append((
        np.array([xmin, ymax - thickness, zmin], dtype=np.float32),
        np.array([xmax, ymax, zmax], dtype=np.float32),
    ))

    walls.append((
        np.array([xmin, ymin, zmin], dtype=np.float32),
        np.array([xmax, ymax, zmin + thickness], dtype=np.float32),
    ))

    return walls


def render_closed_cube_mask(ray_o: np.ndarray, ray_w: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
    hit, _ = ray_aabb_intersection(ray_o, ray_w, bmin, bmax)
    return (hit.astype(np.uint8) * 255)


def render_open_box_depth(ray_o: np.ndarray, ray_w: np.ndarray, ray_cam: np.ndarray, walls) -> np.ndarray:
    t_best = np.full(ray_w.shape[:2], np.inf, dtype=np.float32)

    for bmin, bmax in walls:
        hit, t = ray_aabb_intersection(ray_o, ray_w, bmin, bmax)
        update = hit & (t < t_best)
        t_best = np.where(update, t, t_best)

    hit_any = np.isfinite(t_best)

    # depth = np.zeros_like(t_best, dtype=np.float32)
    # Use 5.0 for background so depth actively carves empty space instead of providing no data!
    depth = np.full_like(t_best, 5.0, dtype=np.float32)
    depth[hit_any] = t_best[hit_any] * ray_cam[..., 2][hit_any]
    depth = np.clip(depth, 0.0, 65.0)
    return depth


def save_mask_bin(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(mask.astype(np.uint8).tobytes())


def save_depth_png_mm(path: Path, depth_m: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    depth_mm = np.clip(depth_m * 1000.0, 0.0, 65535.0).astype(np.uint16)
    cv2.imwrite(str(path), depth_mm)


def save_depth_pth_mm(path: Path, depth_m: np.ndarray) -> None:
    if torch is None:
        raise RuntimeError("Writing .pth depth requires PyTorch. Install torch or use --depth-format png.")

    path.parent.mkdir(parents=True, exist_ok=True)
    depth_mm = np.clip(depth_m * 1000.0, 0.0, 65535.0).astype(np.int32)
    depth_tensor = torch.from_numpy(depth_mm).contiguous()
    torch.save(depth_tensor, str(path))


def _tensor_to_depth_meters(t) -> np.ndarray:
    if t.dim() == 3 and t.size(0) == 1:
        t = t.squeeze(0)
    if t.dim() == 3 and t.size(-1) == 1:
        t = t.squeeze(-1)
    if t.dim() != 2:
        raise RuntimeError(f"Unsupported depth tensor shape for preview: {tuple(t.size())}")
    return t.detach().cpu().numpy().astype(np.float32) / 1000.0


def load_depth_pth_m(path: Path) -> np.ndarray:
    if torch is None:
        raise RuntimeError("Loading .pth depth for preview requires PyTorch.")

    payload = torch.load(str(path), map_location="cpu")
    if torch.is_tensor(payload):
        return _tensor_to_depth_meters(payload)

    if isinstance(payload, dict):
        for value in payload.values():
            if torch.is_tensor(value):
                return _tensor_to_depth_meters(value)

    raise RuntimeError(f"Could not find depth tensor in {path}")


def colorize_depth(depth_m: np.ndarray) -> np.ndarray:
    valid = depth_m > 0.0
    vis = np.zeros_like(depth_m, dtype=np.uint8)

    if np.any(valid):
        dmin = float(depth_m[valid].min())
        dmax = float(depth_m[valid].max())
        if dmax > dmin:
            scaled = (depth_m - dmin) / (dmax - dmin)
            vis[valid] = np.clip(scaled[valid] * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            vis[valid] = 255

    color = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def save_depth_preview_jpg(path: Path, depth_m: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), colorize_depth(depth_m))


def save_rgb(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    rgb[..., 0] = mask
    rgb[..., 1] = mask
    rgb[..., 2] = mask
    cv2.imwrite(str(path), rgb)


def make_config(dataset_root: Path, config_path: Path, depth_extension: str) -> None:
    config = {
        "type": "VCI_REAL",
        "version": "2.1",
        "dataset": {
            "frame_count": 1,
            "start_frame": 0,
            "path": str(dataset_root),
            "camera_path": "calibration_dome.json",
            "flip_images": False,
            "flip_masks": False,
            "to_world": [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        },
        "depth": {
            "has_depth": True,
            "scale": 1000.0,
            "extension": depth_extension,
        },
        "reconstructor": {
            "gpu": 0,
            "level": 9,
            "partial_masks": False,
            "depth_fusion_mode": "synthetic",
            "smoothing": False,
        },
        "renderer": {
            "gpu": 0,
        },
        "inpainting": {
            "enable": False,
            "path": "",
        },
        "volume": {
            "position": [0.0, 0.0, 0.0],
            "scale": 1.6,
        },
        "server": {
            "ip": "127.0.0.1",
            "port": 25565,
        },
    }

    config_path.write_text(json.dumps(config, indent=4))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an importer-compatible synthetic multi-camera open-box dataset.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "data" / "multiple_depth_cams",
        help="Output dataset root folder",
    )
    parser.add_argument(
        "--config-out",
        type=Path,
        default=REPO_ROOT / "config_synthetic_open_cube_15cams.json",
        help="Output config json path",
    )
    parser.add_argument(
        "--num-cameras",
        type=int,
        default=15,
        help="Number of RGB camera positions on the full fibonacci sphere",
    )
    parser.add_argument(
        "--num-depth-cameras",
        type=int,
        default=12,
        help="Number of depth camera positions on the +Z fibonacci hemisphere (all face the open side)",
    )
    parser.add_argument(
        "--sphere-radius",
        type=float,
        default=2.5,
        help="Radius of the fibonacci sphere on which cameras are placed",
    )
    parser.add_argument(
        "--depth-layout",
        choices=("hemisphere", "wall-facing", "frontal"),
        default="hemisphere",
        help="Depth camera placement: 'hemisphere' (fibonacci +Z), 'wall-facing' "
             "(each inner wall head-on), or 'frontal' (narrow cone around the opening axis, "
             "avoids grazing the side walls)",
    )
    parser.add_argument("--frontal-half-angle", type=float, default=20.0,
                        help="Half-angle (deg) of the frontal depth-camera cone")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fx", type=float, default=1400.0)
    parser.add_argument("--fy", type=float, default=1400.0)
    parser.add_argument(
        "--depth-format",
        choices=("png", "pth", "both"),
        default="both",
        help="Depth storage format. 'both' writes .png and .pth, importer will prefer .pth automatically.",
    )
    parser.add_argument("--force", action="store_true", help="Delete existing dataset root before generation")

    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    config_out = args.config_out.resolve()

    if dataset_root.exists() and args.force:
        shutil.rmtree(dataset_root)

    frame_dir = dataset_root / "frame_00000"
    (frame_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (frame_dir / "mask").mkdir(parents=True, exist_ok=True)
    (frame_dir / "depth").mkdir(parents=True, exist_ok=True)
    (frame_dir / "depth_preview").mkdir(parents=True, exist_ok=True)

    W, H = args.width, args.height
    fx, fy = float(args.fx), float(args.fy)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0

    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    x = (u - cx) / fx
    y = (v - cy) / fy
    z = np.ones_like(x)
    ray_cam = np.stack([x, y, z], axis=2)
    ray_cam /= np.linalg.norm(ray_cam, axis=2, keepdims=True)

    cube_bmin = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
    cube_bmax = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    # Hollow open box (open face on +Z) for depth — creates the concavity
    walls = generate_open_box_walls(
        xmin=-0.5, xmax=0.5,
        ymin=-0.5, ymax=0.5,
        zmin=-0.5, zmax=0.5,
        thickness=0.08,
    )

    # RGB cameras: full fibonacci sphere
    cameras = []
    for i, (x, y, z) in enumerate(fibonacci_sphere(args.num_cameras, radius=args.sphere_radius)):
        cameras.append({"id": f"C{i:03d}", "type": "rgb", "pos": np.array([x, y, z], dtype=np.float32)})

    # Depth cameras: either the +Z fibonacci hemisphere, or a deliberate wall-facing set
    # where each inner wall is observed head-on (avoids grazing false-air that holes walls).
    if args.depth_layout == "wall-facing":
        depth_pts = wall_facing_depth_positions(radius=args.sphere_radius)
    elif args.depth_layout == "frontal":
        depth_pts = frontal_cone_depth_positions(args.num_depth_cameras, radius=args.sphere_radius,
                                                 half_angle_deg=args.frontal_half_angle)
    else:
        depth_pts = fibonacci_hemisphere_posz(args.num_depth_cameras, radius=args.sphere_radius)
    for i, (x, y, z) in enumerate(depth_pts):
        cameras.append({"id": f"D{i:03d}", "type": "depth", "pos": np.array([x, y, z], dtype=np.float32)})

    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    calib_cameras = []

    for cam in cameras:
        cam_id = cam["id"]
        cam_type = cam["type"]
        cam_pos = cam["pos"]

        view = look_at_cv(cam_pos, target)
        R = view[:3, :3]

        ray_w = np.einsum("hwc,cd->hwd", ray_cam, R)

        if cam_type == "rgb":
            mask = render_closed_cube_mask(cam_pos, ray_w, cube_bmin, cube_bmax)
            save_mask_bin(frame_dir / "mask" / f"mask_{cam_id}.bin", mask)
            save_rgb(frame_dir / "rgb" / f"{cam_id}.jpg", mask)
        else:
            placeholder = np.full((H, W), 255, dtype=np.uint8)
            save_rgb(frame_dir / "rgb" / f"{cam_id}.jpg", placeholder)

            depth_m = render_open_box_depth(cam_pos, ray_w, ray_cam, walls)

            if args.depth_format in ("png", "both"):
                save_depth_png_mm(frame_dir / "depth" / f"{cam_id}.png", depth_m)

            if args.depth_format in ("pth", "both"):
                pth_path = frame_dir / "depth" / f"{cam_id}.pth"
                save_depth_pth_mm(pth_path, depth_m)

                rgb_pth_path = frame_dir / "rgb" / f"{cam_id}.pth"
                save_depth_pth_mm(rgb_pth_path, depth_m)

                depth_from_pth_m = load_depth_pth_m(pth_path)
                save_depth_preview_jpg(frame_dir / "depth_preview" / f"{cam_id}_from_pth_preview.jpg", depth_from_pth_m)
            else:
                save_depth_preview_jpg(frame_dir / "depth_preview" / f"{cam_id}_preview.jpg", depth_m)

        calib_cameras.append(
            {
                "camera_id": cam_id,
                "camera_type": cam_type,
                "intrinsics": {
                    "resolution": [W, H],
                    "camera_matrix": [
                        fx, 0.0, cx,
                        0.0, fy, cy,
                        0.0, 0.0, 1.0,
                    ],
                },
                "extrinsics": {
                    "position": [float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])],
                    "view_matrix": view.reshape(-1).astype(float).tolist(),
                },
            }
        )

    calib = {"cameras": calib_cameras}
    (dataset_root / "calibration_dome.json").write_text(json.dumps(calib, indent=2))

    config_depth_extension = ".pth" if args.depth_format == "pth" else ".png"
    make_config(dataset_root, config_out, depth_extension=config_depth_extension)

    print(f"Synthetic dataset created ({args.num_cameras} RGB cameras on full sphere + {args.num_depth_cameras} depth cameras on +Z hemisphere):")
    print(dataset_root)
    print("Config written:")
    print(config_out)
    print("Depth format:")
    print(args.depth_format)
    print("Notice: Depth .pth files were additionally saved into the /rgb folder as requested.")


if __name__ == "__main__":
    main()