#!/usr/bin/env python3
"""
Visualize camera positions, orientations and targets from camera_debug.json.
4-panel plot: 3D + Top-Down (XZ) + Front (XY) + Side (ZY).
Each camera shows:
  - A dot at its position
  - A bold arrow for its viewing direction (from the GL view matrix)
  - A line to its target point (from the extrinsics decomposition)
Computes the 'center of attention' = point closest to all viewing rays.
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load_cameras(path):
    with open(path) as f:
        return json.load(f)


def gl_col_major_to_mat4(flat):
    """GLM stores column-major: flat[0..3] = col0.
    reshape(4,4) puts each column as a row → .T gives the standard matrix."""
    return np.array(flat, dtype=np.float64).reshape(4, 4).T


def get_forward(view_matrix_gl_flat):
    """Extract world-space forward direction from GL view matrix.
    OpenGL camera looks along -Z in camera space.
    V = [R|t; 0 1], forward = -R^T * [0,0,1] = -R[2,:] = -(row 2 of R)."""
    V = gl_col_major_to_mat4(view_matrix_gl_flat)
    return -V[2, :3]


def compute_center_of_attention(positions, directions):
    """Find the 3D point closest to all viewing rays (least-squares)."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for p, d in zip(positions, directions):
        I_ddT = np.eye(3) - np.outer(d, d)
        A += I_ddT
        b += I_ddT @ p
    return np.linalg.solve(A, b)


def draw_volume_rect_2d(ax, vol_pos, scale, ix, iy, label=None):
    s2 = scale / 2
    xs = [vol_pos[ix] - s2, vol_pos[ix] + s2, vol_pos[ix] + s2, vol_pos[ix] - s2, vol_pos[ix] - s2]
    ys = [vol_pos[iy] - s2, vol_pos[iy] - s2, vol_pos[iy] + s2, vol_pos[iy] + s2, vol_pos[iy] - s2]
    ax.plot(xs, ys, color="orange", linewidth=1.5, alpha=0.6, label=label)


def main():
    path = sys.argv[1] if len(sys.argv) >= 2 else "export_depth_ON/camera_debug.json"
    data = load_cameras(path)
    cameras = data["cameras"]
    vol_pos = np.array(data["volume"]["position"])
    vol_scale = data["volume"]["scale"]

    # Old hardcoded visual hull volume (for comparison)
    vh_old_lower = np.array([-2, 0, -2])
    vh_old_scale = 4.0

    # New VH volume (derived from model matrix = volume center - scale to center + scale)
    vh_new_lower = vol_pos - vol_scale
    vh_new_scale = vol_scale * 2

    # Extract per-camera data
    cam_list = []
    for cam in cameras:
        pos = np.array(cam["extrinsics"]["position"])
        target = np.array(cam["extrinsics"]["target"])
        forward = get_forward(cam["extrinsics"]["view_matrix_gl"])
        forward_n = forward / np.linalg.norm(forward)
        cam_list.append(dict(
            pos=pos, target=target, forward=forward_n,
            name=cam["name"], id=cam["id"],
        ))

    positions = np.array([c["pos"] for c in cam_list])
    forwards = np.array([c["forward"] for c in cam_list])
    targets = np.array([c["target"] for c in cam_list])

    # Center of attention (where all viewing rays converge)
    coa = compute_center_of_attention(positions, forwards)

    scene_extent = (positions.max(axis=0) - positions.min(axis=0)).max()
    arrow_len = scene_extent * 0.15

    print(f"Volume center (config):    {vol_pos}")
    print(f"Center of attention (rays): {coa.round(3)}")
    print(f"Camera centroid:            {positions.mean(axis=0).round(3)}")
    print(f"Targets centroid:           {targets.mean(axis=0).round(3)}")

    # ---- Figure: 2x2 layout ----
    fig = plt.figure(figsize=(22, 20))

    # ===== Panel 1: 3D =====
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")

    for c in cam_list:
        p, fwd, t = c["pos"], c["forward"], c["target"]

        # Camera dot
        ax3d.scatter(*p, c="dodgerblue", s=40, zorder=5, edgecolors="black", linewidths=0.3)

        # Viewing direction arrow (bold blue)
        ax3d.quiver(*p, *(fwd * arrow_len), color="dodgerblue",
                    arrow_length_ratio=0.18, linewidth=2.0)

        # Line from camera to its target (red dashed)
        ax3d.plot([p[0], t[0]], [p[1], t[1]], [p[2], t[2]],
                  color="red", linewidth=0.8, alpha=0.5, linestyle="--")

        # Label
        ax3d.text(p[0], p[1], p[2], f" {c['name']}", fontsize=5)

    # Volume center (config)
    ax3d.scatter(*vol_pos, c="orange", s=250, marker="*", zorder=10, label=f"Volume center {tuple(vol_pos)}")

    # Volume box
    s2 = vol_scale / 2
    corners = np.array([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],
                        [1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]], dtype=float) * s2 + vol_pos
    for i, j in [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]:
        ax3d.plot(*zip(corners[i], corners[j]), color="orange", linewidth=0.8, alpha=0.5)

    # Center of attention
    ax3d.scatter(*coa, c="lime", s=250, marker="D", zorder=10, edgecolors="black",
                 label=f"Center of attention {tuple(coa.round(2))}")

    # Target points cluster
    ax3d.scatter(targets[:, 0], targets[:, 1], targets[:, 2],
                 c="red", s=10, alpha=0.4, label="Camera targets")

    all_pts = np.vstack([positions, vol_pos.reshape(1, 3), coa.reshape(1, 3), targets])
    center = all_pts.mean(axis=0)
    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 * 1.15
    ax3d.set_xlim(center[0] - max_range, center[0] + max_range)
    ax3d.set_ylim(center[1] - max_range, center[1] + max_range)
    ax3d.set_zlim(center[2] - max_range, center[2] + max_range)
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    ax3d.set_title("3D View")
    ax3d.legend(fontsize=7, loc="upper left")

    # ===== 2D panels =====
    views_2d = [
        ("Top-Down (XZ)", 0, 2, "X", "Z", (2, 2, 2)),
        ("Front (XY)",    0, 1, "X", "Y", (2, 2, 3)),
        ("Side (ZY)",     2, 1, "Z", "Y", (2, 2, 4)),
    ]

    for title, ix, iy, xlabel, ylabel, subplot_pos in views_2d:
        ax = fig.add_subplot(*subplot_pos)

        for c in cam_list:
            p, fwd, t = c["pos"], c["forward"], c["target"]

            # Camera dot
            ax.plot(p[ix], p[iy], "o", color="dodgerblue", markersize=6,
                    markeredgecolor="black", markeredgewidth=0.4, zorder=5)

            # Viewing direction arrow (blue)
            ax.annotate("", xy=(p[ix] + fwd[ix] * arrow_len, p[iy] + fwd[iy] * arrow_len),
                        xytext=(p[ix], p[iy]),
                        arrowprops=dict(arrowstyle="-|>", color="dodgerblue", lw=2.0))

            # Line to target (red dashed)
            ax.plot([p[ix], t[ix]], [p[iy], t[iy]],
                    color="red", linewidth=0.6, alpha=0.4, linestyle="--")

            # Label
            ax.text(p[ix], p[iy] + scene_extent * 0.012, c["name"],
                    fontsize=4.5, ha="center", color="gray")

        # Targets
        ax.scatter(targets[:, ix], targets[:, iy], c="red", s=8, alpha=0.3, zorder=3, label="Targets")

        # Volume center
        ax.plot(vol_pos[ix], vol_pos[iy], "*", color="orange", markersize=18, zorder=10,
                markeredgecolor="black", markeredgewidth=0.5, label="Volume center (config)")
        draw_volume_rect_2d(ax, vol_pos, vol_scale * 2, ix, iy, label="Config volume (new VH)")

        # Old hardcoded VH volume (red dashed)
        old_center = vh_old_lower + vh_old_scale / 2
        s2o = vh_old_scale / 2
        xs = [old_center[ix]-s2o, old_center[ix]+s2o, old_center[ix]+s2o, old_center[ix]-s2o, old_center[ix]-s2o]
        ys = [old_center[iy]-s2o, old_center[iy]-s2o, old_center[iy]+s2o, old_center[iy]+s2o, old_center[iy]-s2o]
        ax.plot(xs, ys, color="red", linewidth=1.5, alpha=0.6, linestyle="--", label="Old VH volume (hardcoded)")

        # Center of attention
        ax.plot(coa[ix], coa[iy], "D", color="lime", markersize=12, zorder=10,
                markeredgecolor="black", markeredgewidth=0.8, label="Center of attention")

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)

    fig.suptitle(
        f"Camera Debug — {len(cameras)} cameras\n"
        f"Blue dot + arrow = camera position + viewing direction  |  "
        f"Red dashes = line to target point\n"
        f"Orange star = config volume center ({vol_pos[0]:.1f}, {vol_pos[1]:.1f}, {vol_pos[2]:.1f})  |  "
        f"Green diamond = center of attention ({coa[0]:.2f}, {coa[1]:.2f}, {coa[2]:.2f})",
        fontsize=11, fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = path.replace(".json", "_plot.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
