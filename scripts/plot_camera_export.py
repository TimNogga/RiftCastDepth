#!/usr/bin/env python3
"""Plot camera poses from RIFTCast camera_export.json.

Usage:
  python scripts/plot_camera_export.py /path/to/camera_export.json
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Plot cameras from camera_export.json")
    parser.add_argument("json_path", type=Path, help="Path to camera_export.json")
    parser.add_argument(
        "--axis-scale",
        type=float,
        default=0.08,
        help="Length of per-camera orientation axes",
    )
    parser.add_argument(
        "--label",
        action="store_true",
        help="Draw camera id labels next to centers",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Output image path (default: <json_path stem>_plot.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also open an interactive plot window (if display is available)",
    )
    return parser.parse_args()


def _as_4x4_row_major(flat16):
    arr = np.asarray(flat16, dtype=np.float64)
    if arr.size != 16:
        raise ValueError(f"Expected 16 values for matrix, got {arr.size}")
    return arr.reshape(4, 4)


def load_cameras(path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cams = data.get("cameras", [])
    out = []
    for cam in cams:
        cid = cam.get("camera_id_string", str(cam.get("camera_id_numeric", "unknown")))
        ex = cam.get("extrinsics", {})

        view_flat = ex.get("view_matrix_gl_to_world")
        if view_flat is None:
            view_flat = ex.get("view_matrix_opencv")
        if view_flat is None:
            continue

        view = _as_4x4_row_major(view_flat)

        cam_to_world = np.linalg.inv(view)
        center = cam_to_world[:3, 3]
        rot = cam_to_world[:3, :3]

        out.append(
            {
                "id": cid,
                "center": center,
                "right": rot[:, 0],
                "up": rot[:, 1],
                "forward": rot[:, 2],
            }
        )

    return out


def set_axes_equal(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    center = np.mean(limits, axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([center[0] - radius, center[0] + radius])
    ax.set_ylim3d([center[1] - radius, center[1] + radius])
    ax.set_zlim3d([center[2] - radius, center[2] + radius])


def main():
    args = parse_args()
    cameras = load_cameras(args.json_path)

    if not cameras:
        raise SystemExit("No cameras found in JSON file")

    centers = np.stack([c["center"] for c in cameras], axis=0)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=40, c="k", label="camera center")

    for c in cameras:
        p = c["center"]
        s = args.axis_scale
        ax.quiver(p[0], p[1], p[2], c["right"][0], c["right"][1], c["right"][2], length=s, color="r")
        ax.quiver(p[0], p[1], p[2], c["up"][0], c["up"][1], c["up"][2], length=s, color="g")
        ax.quiver(
            p[0], p[1], p[2], c["forward"][0], c["forward"][1], c["forward"][2], length=s, color="b"
        )

        if args.label:
            ax.text(p[0], p[1], p[2], c["id"], fontsize=8)

    ax.set_title(f"Camera Poses ({len(cameras)} cameras)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    set_axes_equal(ax)
    plt.tight_layout()

    output_path = args.save
    if output_path is None:
        output_path = args.json_path.with_name(f"{args.json_path.stem}_plot.png")

    plt.savefig(output_path, dpi=180)
    print(f"Saved plot to: {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
