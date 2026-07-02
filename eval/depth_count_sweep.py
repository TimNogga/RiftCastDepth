#!/usr/bin/env python3
"""
Depth-camera count sweep on the synthetic open-box dataset.

For a fixed set of RGB cameras (visual hull) we progressively add depth cameras
(1 … N_MAX), reconstruct the mesh with the exporter for each count, and measure
the Chamfer distance to the parametric ground-truth open box.

The open box is concave (one open face) so the RGB visual hull can never carve
the interior cavity — only depth fusion can. This sweep therefore shows how the
geometric error drops as more depth sensors observe the cavity.

Pipeline per camera count N:
    1. write a calibration containing all RGB cams + the first N depth cams
    2. run RIFTCast_exporter (TSDF, synthetic depth fusion) -> vertices.bin
    3. sample the mesh + GT box and compute bidirectional Chamfer distance

Run from repo root:
    .venv/bin/python3 eval/depth_count_sweep.py

Stages can be run independently:
    --skip-gen      reuse an existing data/depth_sweep dataset
    --skip-runs     reuse existing reconstructions, only (re)compute metrics+plot
    --counts 1 5 10 20   only evaluate these depth-camera counts
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# reuse the validated rendering / IO helpers from the dataset generator
import generate_synth_data as gsd  # noqa: E402

# reuse the validated mesh-sampling / chamfer helpers
sys.path.insert(0, str(SCRIPT_DIR))
import chamfer_fscore as cf  # noqa: E402


# ── configuration ────────────────────────────────────────────────────────────

DATASET_ROOT = REPO_ROOT / "data" / "depth_sweep"
OUT_DIR = REPO_ROOT / "output" / "evaluation" / "depth_count_sweep"
GT_OBJ = REPO_ROOT / "scripts" / "ground_truth_open_box.obj"

N_MAX = 20                     # maximum number of depth cameras
RADIUS_DEPTH = 2.5             # depth-camera distance from origin [m]
W, H = 1920, 1080
FX = FY = 1400.0

N_SAMPLES = 200_000            # surface points sampled per mesh for chamfer
F_THRESHOLDS = [0.005, 0.01, 0.02, 0.05]

# The exporter binary was built against the `riftcast` conda env's libtorch
# (torch 2.10.0+cu126). That env's torch/lib must be on LD_LIBRARY_PATH or the
# binary hits an undefined-symbol error against any newer torch on the system.
TORCH_LIB = "/home/timnogga/anaconda3/envs/riftcast/lib/python3.12/site-packages/torch/lib"
EXPORTER_PREFIX = [
    "xvfb-run", "-a", "-e", "/dev/stdout", "-s", "-screen 0 1280x720x24",
    "env", f"LD_LIBRARY_PATH={TORCH_LIB}",
    "__NV_PRIME_RENDER_OFFLOAD=1", "__GLX_VENDOR_LIBRARY_NAME=nvidia",
    str(REPO_ROOT / "bin" / "RIFTCast_exporter"),
]

# Fixed RGB cameras — full outer coverage so the visual hull is well constrained.
RGB_CAMERAS = [
    ("C000", [0.0, 0.0, -2.8]),
    ("C001", [-2.8, 0.0, 0.0]),
    ("C002", [2.8, 0.0, 0.0]),
    ("C003", [0.0, 2.8, 0.0]),
    ("C004", [0.0, -2.8, 0.0]),
    ("C005", [0.3, 0.3, 2.8]),
    ("C006", [2.0, 2.0, 2.0]),
    ("C007", [-2.0, 2.0, 2.0]),
    ("C008", [2.0, -2.0, 2.0]),
    ("C009", [-2.0, -2.0, 2.0]),
    ("C010", [2.0, 2.0, -2.0]),
    ("C011", [-2.0, 2.0, -2.0]),
    ("C012", [2.0, -2.0, -2.0]),
    ("C013", [-2.0, -2.0, -2.0]),
]


# ── depth-camera placement ───────────────────────────────────────────────────

def hemisphere_depth_positions(n: int, radius: float) -> list[np.ndarray]:
    """
    n depth-camera positions on the +Z hemisphere (the open face of the box),
    spread with a Fibonacci spiral, then ordered by angular proximity to the
    opening axis (+Z).

    Ordering matters: when only the first k cameras are used, we want the k most
    informative ones (looking straight into the cavity) first, so the sweep
    reflects a sensible greedy placement rather than a random subset.
    """
    pts = []
    golden = np.pi * (3.0 - np.sqrt(5.0))
    # sample over the +Z hemisphere: z in (0, 1]
    for i in range(n):
        z = (i + 0.5) / n              # (0,1] -> +Z hemisphere only
        r = np.sqrt(max(0.0, 1.0 - z * z))
        theta = golden * i
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        pts.append(np.array([x, y, z], dtype=np.float32))
    # order by proximity to +Z axis (most opening-facing first)
    pts.sort(key=lambda p: -float(p[2]))
    return [p * radius for p in pts]


# ── calibration / config writers ─────────────────────────────────────────────

def camera_entry(cam_id: str, cam_type: str, pos: np.ndarray) -> dict:
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    view = gsd.look_at_cv(pos.astype(np.float32), np.zeros(3, np.float32))
    return {
        "camera_id": cam_id,
        "camera_type": cam_type,
        "intrinsics": {
            "resolution": [W, H],
            "camera_matrix": [FX, 0.0, cx, 0.0, FY, cy, 0.0, 0.0, 1.0],
        },
        "extrinsics": {
            "position": [float(pos[0]), float(pos[1]), float(pos[2])],
            "view_matrix": view.reshape(-1).astype(float).tolist(),
        },
    }


def write_calibration(path: Path, depth_positions: list[np.ndarray], n_depth: int):
    cams = [camera_entry(cid, "rgb", np.array(p, np.float32)) for cid, p in RGB_CAMERAS]
    for i in range(n_depth):
        cams.append(camera_entry(f"D{i:03d}", "depth", depth_positions[i]))
    path.write_text(json.dumps({"cameras": cams}, indent=2))


def write_config(path: Path, dataset_root: Path):
    config = {
        "type": "VCI_REAL",
        "version": "2.1",
        "dataset": {
            "frame_count": 1,
            "start_frame": 0,
            "path": str(dataset_root),
            "camera_path": "calibration_full.json",
            "flip_images": False,
            "flip_masks": False,
            "to_world": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                         0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        },
        "depth": {"has_depth": True, "scale": 1000.0, "extension": ".png", "enable_cutter": False},
        "reconstructor": {"gpu": 0, "level": 9, "partial_masks": False,
                          "depth_fusion_mode": "synthetic", "smoothing": False},
        "renderer": {"gpu": 0},
        "inpainting": {"enable": False, "path": ""},
        "volume": {"position": [0.0, 0.0, 0.0], "scale": 1.6},
        "server": {"ip": "127.0.0.1", "port": 25565},
    }
    path.write_text(json.dumps(config, indent=4))


# ── dataset generation ───────────────────────────────────────────────────────

def generate_dataset(depth_positions: list[np.ndarray]):
    """Render the open-box dataset once: RGB masks + 20 depth maps."""
    frame_dir = DATASET_ROOT / "frame_00000"
    for sub in ("rgb", "mask", "depth", "depth_preview"):
        (frame_dir / sub).mkdir(parents=True, exist_ok=True)

    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    ray_cam = np.stack([(u - cx) / FX, (v - cy) / FY, np.ones_like(u)], axis=2)
    ray_cam /= np.linalg.norm(ray_cam, axis=2, keepdims=True)

    cube_bmin = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
    cube_bmax = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    walls = gsd.generate_open_box_walls(-0.5, 0.5, -0.5, 0.5, -0.5, 0.5, thickness=0.08)
    target = np.zeros(3, np.float32)

    print(f"Generating dataset → {DATASET_ROOT}")
    # RGB cameras: silhouette mask + dummy rgb
    for cid, pos in RGB_CAMERAS:
        pos = np.array(pos, np.float32)
        R = gsd.look_at_cv(pos, target)[:3, :3]
        ray_w = np.einsum("hwc,cd->hwd", ray_cam, R)
        mask = gsd.render_closed_cube_mask(pos, ray_w, cube_bmin, cube_bmax)
        gsd.save_mask_bin(frame_dir / "mask" / f"mask_{cid}.bin", mask)
        gsd.save_rgb(frame_dir / "rgb" / f"{cid}.jpg", mask)
        print(f"  rgb   {cid}")

    # Depth cameras: perspective depth of the open box (background carves empty space)
    for i, pos in enumerate(depth_positions):
        cid = f"D{i:03d}"
        R = gsd.look_at_cv(pos.astype(np.float32), target)[:3, :3]
        ray_w = np.einsum("hwc,cd->hwd", ray_cam, R)
        depth_m = gsd.render_open_box_depth(pos.astype(np.float32), ray_w, ray_cam, walls)
        gsd.save_rgb(frame_dir / "rgb" / f"{cid}.jpg", np.full((H, W), 255, np.uint8))
        # 16-bit PNG depth in mm. We deliberately avoid .pth: the exporter loads
        # torch pickles with its build-time torch (2.10) and chokes on tensors
        # saved by a newer torch. PNG depth (uint16 mm /scale -> metres) is
        # version-independent and the importer falls back to it when no .pth exists.
        gsd.save_depth_png_mm(frame_dir / "depth" / f"{cid}.png", depth_m)
        gsd.save_depth_preview_jpg(frame_dir / "depth_preview" / f"{cid}.jpg", depth_m)
        print(f"  depth {cid}  pos=({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f})")

    # full calibration (all cameras) referenced by the base config
    write_calibration(DATASET_ROOT / "calibration_full.json", depth_positions, N_MAX)
    write_config(DATASET_ROOT / "config_depth_sweep.json", DATASET_ROOT)
    print("Dataset generation done.\n")


# ── exporter ─────────────────────────────────────────────────────────────────

def run_exporter(config: Path, calib_abs: Path, out_dir: Path) -> bool:
    cmd = EXPORTER_PREFIX + [
        "--dataset", str(config),
        "--calibration", str(calib_abs),
        "--output", str(out_dir),
        "--inpaint", "false",
        "--frame", "0",
    ]
    log_path = out_dir / "exporter.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=log,
                              stderr=subprocess.STDOUT)
    ok = (out_dir / "frame_00000" / "vertices.bin").exists()
    if proc.returncode != 0 or not ok:
        print(f"    [warn] exporter rc={proc.returncode}, vertices={'yes' if ok else 'NO'} "
              f"(see {log_path})")
    return ok


# ── chamfer ──────────────────────────────────────────────────────────────────

def chamfer_for_run(run_dir: Path, pts_gt: np.ndarray) -> dict | None:
    vp = run_dir / "frame_00000" / "vertices.bin"
    fp = run_dir / "frame_00000" / "faces.bin"
    if not vp.exists() or not fp.exists():
        return None
    v = np.fromfile(vp, dtype=np.float32).reshape(-1, 3)
    f = np.fromfile(fp, dtype=np.int64).reshape(-1, 3)
    if len(v) == 0 or len(f) == 0:
        return None
    pts_pred = cf.sample_surface(v, f, N_SAMPLES)
    m = cf.chamfer_and_fscore(pts_pred, pts_gt, F_THRESHOLDS)
    m["vertices"] = int(len(v))
    m["faces"] = int(len(f))
    return m


# ── plot ─────────────────────────────────────────────────────────────────────

def make_plot(results: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = sorted(results.keys())
    cd_cm = [results[n]["chamfer_mean_cm"] for n in counts]
    f1cm = [results[n]["fscores"][0.01]["fscore"] for n in counts]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color_cd = "#e74c3c"
    ax1.plot(counts, cd_cm, "-o", color=color_cd, linewidth=2.2,
             markersize=7, label="Chamfer distance")
    ax1.set_xlabel("Number of depth cameras", fontsize=12)
    ax1.set_ylabel("Chamfer distance to GT [cm]  ↓", color=color_cd, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=color_cd)
    ax1.set_xticks(counts)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color_f = "#2980b9"
    ax2.plot(counts, f1cm, "--s", color=color_f, linewidth=1.8,
             markersize=6, label="F-Score @1cm")
    ax2.set_ylabel("F-Score @1cm  ↑", color=color_f, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=color_f)
    ax2.set_ylim(0, 1.02)

    best = counts[int(np.argmin(cd_cm))]
    ax1.set_title("Reconstruction error vs. number of depth cameras\n"
                  "(synthetic open box, 14 RGB cameras fixed)", fontweight="bold")
    for n, y in zip(counts, cd_cm):
        ax1.annotate(f"{y:.2f}", (n, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=7, color=color_cd)

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right")
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "chamfer_vs_depth_cameras.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_png}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gen", action="store_true", help="reuse existing dataset")
    ap.add_argument("--skip-runs", action="store_true", help="reuse existing reconstructions")
    ap.add_argument("--counts", type=int, nargs="+", default=None,
                    help="specific depth-camera counts to evaluate (default 1..N_MAX)")
    args = ap.parse_args()

    np.random.seed(42)
    counts = args.counts if args.counts else list(range(1, N_MAX + 1))
    depth_positions = hemisphere_depth_positions(N_MAX, RADIUS_DEPTH)

    if not args.skip_gen:
        generate_dataset(depth_positions)

    config = DATASET_ROOT / "config_depth_sweep.json"
    calib_dir = OUT_DIR / "calibrations"
    calib_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_runs:
        print("Running reconstructions …")
        for n in counts:
            calib = calib_dir / f"calib_{n:02d}depth.json"
            write_calibration(calib, depth_positions, n)
            run_dir = OUT_DIR / f"run_{n:02d}depth"
            t0 = time.time()
            ok = run_exporter(config, calib, run_dir)
            print(f"  [{n:2d} depth cams] {'ok' if ok else 'FAILED'}  ({time.time()-t0:.1f}s)")
        print()

    # ground truth
    print("Loading + sampling GT box …")
    gt_v, gt_f = cf.load_obj(GT_OBJ)
    pts_gt = cf.sample_surface(gt_v, gt_f, N_SAMPLES)

    print("Computing Chamfer distances …")
    results: dict[int, dict] = {}
    for n in counts:
        run_dir = OUT_DIR / f"run_{n:02d}depth"
        m = chamfer_for_run(run_dir, pts_gt)
        if m is None:
            print(f"  [{n:2d} depth cams] no mesh — skipped")
            continue
        results[n] = m
        print(f"  [{n:2d} depth cams] CD={m['chamfer_mean_cm']:.3f} cm  "
              f"F@1cm={m['fscores'][0.01]['fscore']:.3f}  "
              f"verts={m['vertices']:,}")

    if not results:
        print("No results — nothing to plot.")
        return

    # serialise (float threshold keys -> strings)
    def serialise(r):
        out = {k: v for k, v in r.items() if k != "fscores"}
        out["fscores"] = {f"{int(t*1000)}mm": v for t, v in r["fscores"].items()}
        return out

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "depth_count_sweep.json", "w") as f:
        json.dump({str(n): serialise(results[n]) for n in sorted(results)}, f, indent=2)
    print(f"  → {OUT_DIR / 'depth_count_sweep.json'}")

    make_plot(results)
    print(f"\nDone. Output → {OUT_DIR}")


if __name__ == "__main__":
    main()
