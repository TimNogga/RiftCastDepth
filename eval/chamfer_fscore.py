#!/usr/bin/env python3
"""Chamfer Distance and F-Score evaluation for the synthetic open-cube dataset."""

from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
GT_OBJ    = REPO_ROOT / "scripts" / "ground_truth_open_box.obj"
OUT_DIR   = REPO_ROOT / "output" / "evaluation" / "synthetic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    "no_depth":    REPO_ROOT / "runtime/outputs/runtime_depth_fusion_mode_none_no_depth",
    "depth_real":  REPO_ROOT / "runtime/outputs/runtime_depth_fusion_mode_real_data",
    "depth_synth": REPO_ROOT / "runtime/outputs/runtime_depth_fusion_mode_synthetic_data",
}
RUN_LABELS = {
    "no_depth":    "No Depth (Baseline)",
    "depth_real":  "Depth TSDF (real preset)",
    "depth_synth": "Depth TSDF (synth preset)",
}
RUN_COLORS = {"no_depth": "#e74c3c", "depth_real": "#3498db", "depth_synth": "#2ecc71"}

N_SAMPLES   = 200_000          # surface points sampled per mesh
F_THRESHOLDS = [0.005, 0.01, 0.02, 0.05]  # metres: 5mm, 1cm, 2cm, 5cm


# ── mesh I/O ───────────────────────────────────────────────────────────────

def load_obj(path: Path):
    """Load an OBJ with quad or triangle faces → (V, F_tri) numpy arrays."""
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append(list(map(float, line.split()[1:4])))
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
                if len(idx) == 3:
                    faces.append(idx)
                elif len(idx) == 4:          # split quad → 2 triangles
                    faces.append([idx[0], idx[1], idx[2]])
                    faces.append([idx[0], idx[2], idx[3]])
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int64)


def load_bin_mesh(run_dir: Path):
    """Load vertices.bin + faces.bin from an exporter output directory."""
    vp = run_dir / "frame_00000/vertices.bin"
    fp = run_dir / "frame_00000/faces.bin"
    v = np.fromfile(vp, dtype=np.float32).reshape(-1, 3)
    f = np.fromfile(fp, dtype=np.int64).reshape(-1, 3)
    return v, f


# ── surface sampling ───────────────────────────────────────────────────────

def sample_surface(verts: np.ndarray, faces: np.ndarray, n: int) -> np.ndarray:
    """
    Area-weighted random sampling of n points on the triangle mesh surface.
    Returns (n, 3) float32 array.
    """
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross  = np.cross(v1 - v0, v2 - v0)
    areas  = 0.5 * np.linalg.norm(cross, axis=1)
    total  = areas.sum()
    if total == 0:
        raise ValueError("Mesh has zero total area – cannot sample.")

    probs    = areas / total
    tri_idx  = np.random.choice(len(faces), size=n, p=probs)
    r1 = np.random.rand(n, 1)
    r2 = np.random.rand(n, 1)
    # Barycentric sampling (Osada et al.)
    mask = (r1 + r2) > 1.0
    r1[mask] = 1.0 - r1[mask]
    r2[mask] = 1.0 - r2[mask]
    r3 = 1.0 - r1 - r2

    pts = (r1 * verts[faces[tri_idx, 0]] +
           r2 * verts[faces[tri_idx, 1]] +
           r3 * verts[faces[tri_idx, 2]])
    return pts.astype(np.float32)


# ── metrics ────────────────────────────────────────────────────────────────

def chamfer_and_fscore(pts_pred: np.ndarray, pts_gt: np.ndarray,
                       thresholds: list[float]) -> dict:
    """
    Compute bidirectional Chamfer distance and F-Score at given thresholds.

    Chamfer Distance (one-sided):
        d_pred→gt = mean of min distances from each pred point to GT
        d_gt→pred = mean of min distances from each GT point to pred
        CD = (d_pred→gt + d_gt→pred) / 2

    F-Score at threshold t:
        Precision = fraction of pred points with a GT point within t
        Recall    = fraction of GT points with a pred point within t
        F = 2*P*R / (P+R)
    """
    tree_gt   = cKDTree(pts_gt)
    tree_pred = cKDTree(pts_pred)

    dist_pred_to_gt, _ = tree_gt.query(pts_pred,   workers=-1)
    dist_gt_to_pred, _ = tree_pred.query(pts_gt,   workers=-1)

    d_fwd  = float(dist_pred_to_gt.mean())   # pred → gt
    d_bwd  = float(dist_gt_to_pred.mean())   # gt → pred
    cd     = (d_fwd + d_bwd) / 2.0

    fscores = {}
    for t in thresholds:
        precision = float((dist_pred_to_gt < t).mean())
        recall    = float((dist_gt_to_pred < t).mean())
        denom = precision + recall
        fs = (2 * precision * recall / denom) if denom > 0 else 0.0
        fscores[t] = {"precision": precision, "recall": recall, "fscore": fs}

    return {
        "chamfer_mean_m":  cd,
        "chamfer_fwd_m":   d_fwd,
        "chamfer_bwd_m":   d_bwd,
        "chamfer_mean_cm": cd * 100,
        "fscores":         fscores,
    }


# ── plots ──────────────────────────────────────────────────────────────────

def plot_chamfer(results: dict):
    names  = list(results.keys())
    labels = [RUN_LABELS[n] for n in names]
    colors = [RUN_COLORS[n] for n in names]
    cd_cm  = [results[n]["chamfer_mean_cm"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(names)), cd_cm, color=colors, width=0.5, edgecolor="white")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Chamfer Distance [cm] ↓")
    ax.set_title("Chamfer Distance to GT (lower = better)", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, v in zip(bars, cd_cm):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01,
                f"{v:.3f} cm", ha="center", va="bottom", fontsize=9, fontweight="bold")
    best = int(np.argmin(cd_cm))
    bars[best].set_edgecolor("#2c3e50"); bars[best].set_linewidth(2.5)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "01_chamfer_distance.png", bbox_inches="tight")
    plt.close(fig)
    print("  → 01_chamfer_distance.png")


def plot_fscore(results: dict):
    names      = list(results.keys())
    thresholds = F_THRESHOLDS
    x = np.arange(len(thresholds))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, name in enumerate(names):
        fs_vals = [results[name]["fscores"][t]["fscore"] for t in thresholds]
        bars = ax.bar(x + i*width, fs_vals, width, label=RUN_LABELS[name],
                      color=RUN_COLORS[name], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{int(t*1000)} mm" for t in thresholds], fontsize=10)
    ax.set_ylabel("F-Score ↑")
    ax.set_title("F-Score at Distance Thresholds (higher = better)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "02_fscore.png", bbox_inches="tight")
    plt.close(fig)
    print("  → 02_fscore.png")


def plot_precision_recall(results: dict):
    names      = list(results.keys())
    thresholds = F_THRESHOLDS

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Precision & Recall at Distance Thresholds", fontweight="bold")
    x = np.arange(len(thresholds))
    width = 0.25

    for col, metric in enumerate(["precision", "recall"]):
        ax = axes[col]
        for i, name in enumerate(names):
            vals = [results[name]["fscores"][t][metric] for t in thresholds]
            ax.bar(x + i*width, vals, width, label=RUN_LABELS[name],
                   color=RUN_COLORS[name], alpha=0.85)
        ax.set_xticks(x + width)
        ax.set_xticklabels([f"{int(t*1000)} mm" for t in thresholds])
        ax.set_ylabel(metric.capitalize() + " ↑")
        ax.set_title(metric.capitalize(), fontweight="bold")
        ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "03_precision_recall.png", bbox_inches="tight")
    plt.close(fig)
    print("  → 03_precision_recall.png")


def plot_summary_table(results: dict):
    names = list(results.keys())
    rows = []
    for n in names:
        r = results[n]
        fs_1cm = r["fscores"][0.01]
        rows.append([
            RUN_LABELS[n],
            f"{r['chamfer_mean_cm']:.3f}",
            f"{r['chamfer_fwd_m']*100:.3f}",
            f"{r['chamfer_bwd_m']*100:.3f}",
            f"{fs_1cm['precision']:.3f}",
            f"{fs_1cm['recall']:.3f}",
            f"{fs_1cm['fscore']:.3f}",
        ])
    col_labels = ["Configuration", "CD [cm] ↓", "CD pred→GT", "CD GT→pred",
                  "Prec @1cm ↑", "Rec @1cm ↑", "F @1cm ↑"]
    fig, ax = plt.subplots(figsize=(16, 3))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 2.2)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 1:
            cell.set_facecolor("#f0f4f8")
    ax.set_title("Chamfer Distance & F-Score Summary (synthetic open-cube dataset)",
                 fontweight="bold", pad=15)
    fig.savefig(OUT_DIR / "04_geometry_table.png", bbox_inches="tight")
    plt.close(fig)
    print("  → 04_geometry_table.png")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)

    print("=" * 55)
    print("Chamfer Distance & F-Score – Synthetic Dataset")
    print("=" * 55)

    # Load and sample GT mesh
    print(f"\nLoading GT mesh: {GT_OBJ.name}")
    gt_v, gt_f = load_obj(GT_OBJ)
    print(f"  GT: {len(gt_v)} vertices, {len(gt_f)} triangles")
    pts_gt = sample_surface(gt_v, gt_f, N_SAMPLES)
    print(f"  Sampled {N_SAMPLES:,} GT surface points")

    results = {}
    for run_name, run_dir in RUNS.items():
        if not run_dir.exists():
            print(f"\n  SKIP {run_name} – not found"); continue
        print(f"\n[{run_name}]")
        pred_v, pred_f = load_bin_mesh(run_dir)
        print(f"  Pred: {len(pred_v):,} vertices, {len(pred_f):,} faces")
        pts_pred = sample_surface(pred_v, pred_f, N_SAMPLES)
        print(f"  Sampled {N_SAMPLES:,} prediction surface points")
        print("  Computing distances …")
        metrics = chamfer_and_fscore(pts_pred, pts_gt, F_THRESHOLDS)
        results[run_name] = metrics
        print(f"  Chamfer Distance: {metrics['chamfer_mean_cm']:.4f} cm")
        print(f"  F-Score @1cm:     {metrics['fscores'][0.01]['fscore']:.4f}")
        for t in F_THRESHOLDS:
            fs = metrics["fscores"][t]
            print(f"    {int(t*1000):2d}mm  P={fs['precision']:.3f}  R={fs['recall']:.3f}  F={fs['fscore']:.3f}")

    if not results:
        print("No results."); return

    print("\nGenerating plots …")
    plot_chamfer(results)
    plot_fscore(results)
    plot_precision_recall(results)
    plot_summary_table(results)

    # serialise (convert float keys to strings for JSON)
    def serialise(r):
        out = {k: v for k, v in r.items() if k != "fscores"}
        out["fscores"] = {f"{int(t*1000)}mm": v for t, v in r["fscores"].items()}
        return out

    with open(OUT_DIR / "geometry_metrics.json", "w") as f:
        json.dump({n: serialise(results[n]) for n in results}, f, indent=2)
    print("  → geometry_metrics.json")
    print(f"\nDone. Output → {OUT_DIR}")


if __name__ == "__main__":
    main()
