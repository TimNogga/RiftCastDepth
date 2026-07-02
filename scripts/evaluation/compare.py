#!/usr/bin/env python3
"""
Comparative Analysis – Depth-Augmented RIFTCast Experiments
=============================================================
Generates mesh-quality and rendered-view comparison plots for the three
main experiments:
  exp_01_no_depth        – visual hull, no depth
  exp_02_depth_tsdf      – visual hull + TSDF depth fusion (cutter off)
  exp_03_depth_cutter    – visual hull + TSDF + post-MC concavity cutter

Run from repo root:
    python3 scripts/evaluation/compare.py
Outputs go to output/evaluation/plots/.
"""

from pathlib import Path
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDY_DIR = REPO_ROOT / "output" / "comparative_study"
PLOTS_DIR = REPO_ROOT / "output" / "evaluation" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 10,
    "figure.facecolor": "white",
})

EXPERIMENTS = [
    {"id": "exp_01_no_depth",         "label": "No Depth\n(Baseline)", "short": "no_depth",    "color": "#e74c3c"},
    {"id": "exp_02_depth_tsdf",        "label": "Depth TSDF",           "short": "depth_tsdf",  "color": "#3498db"},
    {"id": "exp_03_depth_cutter_020m", "label": "Depth + Cutter",       "short": "depth_cutter","color": "#2ecc71"},
]


# ── helpers ────────────────────────────────────────────────────────────────

def save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path.name}")


def rot90ccw(img: np.ndarray) -> np.ndarray:
    return np.rot90(img, k=1)


def load_mesh(exp_dir: Path):
    vp = exp_dir / "frame_00000/vertices.bin"
    fp = exp_dir / "frame_00000/faces.bin"
    if not vp.exists():
        return None, None
    return (np.fromfile(vp, dtype=np.float32).reshape(-1, 3),
            np.fromfile(fp, dtype=np.int64).reshape(-1, 3))


def mesh_stats(v, f):
    if v is None:
        return {}
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    all_e = np.concatenate([
        np.linalg.norm(v1 - v0, axis=1),
        np.linalg.norm(v2 - v1, axis=1),
        np.linalg.norm(v0 - v2, axis=1),
    ])
    return {
        "vertices":       len(v),
        "faces":          len(f),
        "surface_area_m2": float(areas.sum()),
        "mean_edge_mm":   float(all_e.mean() * 1000),
        "degenerate":     int((areas < 1e-10).sum()),
    }


def load_rendered_rgb(exp_dir: Path, cam_idx: int = None) -> np.ndarray | None:
    if not HAS_PIL:
        return None
    rgb_dir = exp_dir / "frame_00000/rgb"
    if not rgb_dir.exists():
        return None
    files = sorted(rgb_dir.glob("*.png"))
    if not files:
        return None
    pick = files[cam_idx % len(files)] if cam_idx is not None else files[len(files) // 2]
    return rot90ccw(np.array(PILImage.open(pick)))


def load_error_map(exp_dir: Path, cam_idx: int = None) -> np.ndarray | None:
    if not HAS_PIL:
        return None
    err_dir = exp_dir / "frame_00000/error"
    if not err_dir.exists():
        return None
    files = sorted(err_dir.glob("*.png"))
    if not files:
        return None
    pick = files[cam_idx % len(files)] if cam_idx is not None else files[len(files) // 2]
    return rot90ccw(np.array(PILImage.open(pick).convert("L")))


def collect_error_fractions(exp_dir: Path) -> list[float]:
    if not HAS_PIL:
        return []
    err_dir = exp_dir / "frame_00000/error"
    if not err_dir.exists():
        return []
    return [float((np.array(PILImage.open(p).convert("L")) > 128).mean())
            for p in sorted(err_dir.glob("*.png"))]


# ── load ───────────────────────────────────────────────────────────────────

def load_experiments() -> dict:
    results = {}
    for exp in EXPERIMENTS:
        d = STUDY_DIR / exp["id"]
        if not d.exists():
            print(f"  SKIP {exp['id']} (not found)")
            continue
        v, f = load_mesh(d)
        s = mesh_stats(v, f)
        err = collect_error_fractions(d)
        s["mean_error_pct"] = float(np.mean(err) * 100) if err else 0.0
        results[exp["id"]] = {**exp, "stats": s, "dir": d, "vertices": v, "faces": f}
    print(f"  Loaded {len(results)} experiments")
    return results


# ── plots ──────────────────────────────────────────────────────────────────

def plot_mesh_sizes(results: dict):
    print("[1] Mesh size comparison …")
    avail  = [e for e in EXPERIMENTS if e["id"] in results]
    labels = [e["label"] for e in avail]
    colors = [e["color"] for e in avail]
    verts  = [results[e["id"]]["stats"].get("vertices", 0) for e in avail]
    faces  = [results[e["id"]]["stats"].get("faces",    0) for e in avail]
    x = np.arange(len(avail))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Mesh Complexity", fontweight="bold")
    for ax, vals, title, unit in zip(axes,
                                     [verts, faces],
                                     ["Vertex Count", "Face Count"],
                                     ["vertices", "faces"]):
        bars = ax.bar(x, vals, color=colors, width=0.55, edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(f"Number of {unit}"); ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{v:,}", ha="center", va="bottom", fontsize=8)
    save_fig(fig, PLOTS_DIR / "01_mesh_size_comparison.png")


def plot_rendered_gallery(results: dict):
    print("[2] Rendered views gallery …")
    if not HAS_PIL:
        print("  SKIP – PIL not available"); return
    avail = [e for e in EXPERIMENTS if e["id"] in results]
    cam_indices = [3, 8, 14, 20]
    fig, axes = plt.subplots(len(avail), len(cam_indices),
                             figsize=(5 * len(cam_indices), 5 * len(avail)))
    fig.suptitle("Rendered Views per Experiment", fontweight="bold")
    for row, exp in enumerate(avail):
        for col, ci in enumerate(cam_indices):
            ax = axes[row][col]
            img = load_rendered_rgb(results[exp["id"]]["dir"], ci)
            if img is not None:
                ax.imshow(img); ax.axis("off")
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")
            if col == 0:
                ax.set_ylabel(exp["label"].replace("\n", " "), color=exp["color"],
                              fontsize=10, fontweight="bold")
            if row == 0:
                ax.set_title(f"Camera {col + 1}", fontsize=9)
    save_fig(fig, PLOTS_DIR / "02_rendered_gallery.png")


def plot_error_comparison(results: dict):
    print("[3] Error map comparison …")
    if not HAS_PIL:
        print("  SKIP – PIL not available"); return
    avail = [e for e in EXPERIMENTS if e["id"] in results]
    cam_idx = 5
    fig, axes = plt.subplots(1, len(avail), figsize=(5 * len(avail), 6))
    fig.suptitle("Error Maps (per-pixel rendering error)", fontweight="bold")
    for ax, exp in zip(axes, avail):
        err = load_error_map(results[exp["id"]]["dir"], cam_idx)
        mean_err = results[exp["id"]]["stats"].get("mean_error_pct", 0)
        if err is not None:
            ax.imshow(err, cmap="hot")
        ax.axis("off")
        ax.set_title(f"{exp['label'].replace(chr(10), ' ')}\nerr={mean_err:.2f}%",
                     color=exp["color"], fontsize=10, fontweight="bold")
    save_fig(fig, PLOTS_DIR / "03_error_comparison.png")


def plot_main_comparison(results: dict):
    print("[4] Main thesis figure: baseline vs depth …")
    if not HAS_PIL:
        print("  SKIP – PIL not available"); return
    key = ["exp_01_no_depth", "exp_02_depth_tsdf"]
    if not all(k in results for k in key):
        print("  SKIP – need exp_01 and exp_02"); return
    cam_indices = [3, 8, 14, 20]
    fig, axes = plt.subplots(2, len(cam_indices), figsize=(5 * len(cam_indices), 10))
    fig.suptitle("No Depth (top)  vs  Depth TSDF (bottom)", fontweight="bold", fontsize=13)
    for row, exp_id in enumerate(key):
        exp = next(e for e in EXPERIMENTS if e["id"] == exp_id)
        for col, ci in enumerate(cam_indices):
            ax = axes[row][col]
            img = load_rendered_rgb(results[exp_id]["dir"], ci)
            if img is not None:
                ax.imshow(img)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(exp["label"].replace("\n", " "), color=exp["color"],
                              fontsize=11, fontweight="bold")
    save_fig(fig, PLOTS_DIR / "04_MAIN_nodepth_vs_depth.png")


def plot_mesh_projections(results: dict):
    print("[5] Mesh orthographic projections …")
    avail = [e for e in EXPERIMENTS if e["id"] in results]
    proj_configs = [
        (0, 1, "Front (X–Y)", "X [m]", "Y [m]"),
        (0, 2, "Top  (X–Z)",  "X [m]", "Z [m]"),
        (2, 1, "Side (Z–Y)",  "Z [m]", "Y [m]"),
    ]
    fig, axes = plt.subplots(3, len(avail), figsize=(5 * len(avail), 15))
    fig.suptitle("Mesh Orthographic Projections", fontweight="bold")
    for col, exp in enumerate(avail):
        v = results[exp["id"]]["vertices"]
        for row, (xi, yi, title, xl, yl) in enumerate(proj_configs):
            ax = axes[row][col]
            if v is not None:
                ax.scatter(v[:, xi], v[:, yi], s=0.05, c=exp["color"], alpha=0.3, rasterized=True)
            ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
            ax.set_xlabel(xl); ax.set_ylabel(yl)
            if row == 0:
                n = results[exp["id"]]["stats"].get("vertices", "?")
                ax.set_title(f"{exp['label'].replace(chr(10), ' ')}\n{n:,} verts", fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{title}\n{yl}")
    save_fig(fig, PLOTS_DIR / "05_mesh_projections.png")


def write_summary(results: dict):
    summary = {
        exp_id: {
            "label": data["label"].replace("\n", " "),
            **data["stats"],
        }
        for exp_id, data in results.items()
    }
    out = PLOTS_DIR / "experiment_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  → experiment_summary.json")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("Comparative Study – Depth-Augmented RIFTCast")
    print("=" * 55)
    results = load_experiments()
    if not results:
        print("No experiment results found under", STUDY_DIR)
        sys.exit(1)
    plot_mesh_sizes(results)
    plot_rendered_gallery(results)
    plot_error_comparison(results)
    plot_main_comparison(results)
    plot_mesh_projections(results)
    write_summary(results)
    print(f"\nDone. Plots → {PLOTS_DIR}")


if __name__ == "__main__":
    main()
