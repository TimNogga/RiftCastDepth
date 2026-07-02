#!/usr/bin/env python3
"""Generate PSNR/SSIM/LPIPS comparison plots from per-experiment metrics JSONs."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT   = Path(__file__).resolve().parents[2]
METRICS_DIR = REPO_ROOT / "output" / "evaluation" / "metrics"
PLOTS_DIR   = REPO_ROOT / "output" / "evaluation" / "plots"

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 10,
    "figure.facecolor": "white",
})

EXPERIMENT_ORDER = [
    ("exp_01_no_depth",          "No Depth\n(Baseline)", "#e74c3c"),
    ("exp_02_depth_tsdf",        "Depth TSDF",           "#3498db"),
    ("exp_03_depth_cutter_020m", "Depth + Cutter",       "#2ecc71"),
]


def load_metrics():
    data = {}
    for exp_id, label, color in EXPERIMENT_ORDER:
        p = METRICS_DIR / f"{exp_id}_metrics.json"
        if not p.exists():
            print(f"  SKIP {exp_id} (no metrics json)")
            continue
        with open(p) as f:
            d = json.load(f)
        data[exp_id] = {"label": label, "color": color, "mean": d["mean"], "cameras": d["cameras"]}
    return data


def plot_bar_comparison(data: dict):
    exps = list(data.keys())
    labels = [data[e]["label"] for e in exps]
    colors = [data[e]["color"] for e in exps]

    psnrs  = [data[e]["mean"]["psnr"]  for e in exps]
    ssims  = [data[e]["mean"]["ssim"]  for e in exps]
    lpipss = [data[e]["mean"]["lpips"] or 0.0 for e in exps]

    x = np.arange(len(exps))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Image Quality Metrics – Mean over 8 Hold-out Cameras", fontweight="bold")

    for ax, vals, title, better in zip(axes,
                                       [psnrs, ssims, lpipss],
                                       ["PSNR (dB) ↑", "SSIM ↑", "LPIPS ↓"],
                                       [True, True, False]):
        bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        # Annotate values
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001 * max(vals),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        # Highlight best
        best_idx = (np.argmax(vals) if better else np.argmin(vals))
        bars[best_idx].set_edgecolor("#2c3e50")
        bars[best_idx].set_linewidth(2.5)

    plt.tight_layout()
    out = PLOTS_DIR / "11_psnr_ssim_lpips.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}")


def plot_per_camera_breakdown(data: dict):
    """PSNR per camera for the 3 main experiments."""
    key_exps = ["exp_01_no_depth", "exp_02_depth_tsdf", "exp_03_depth_cutter_020m"]
    avail = [e for e in key_exps if e in data]
    if len(avail) < 2:
        return

    cam_names = [c["cam_name"] for c in data[avail[0]]["cameras"]]
    x = np.arange(len(cam_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, exp_id in enumerate(avail):
        psnrs = [c["psnr"] for c in data[exp_id]["cameras"]]
        ax.bar(x + i * width, psnrs, width, label=data[exp_id]["label"].replace("\n", " "),
               color=data[exp_id]["color"], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(cam_names, rotation=45, ha="right")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Per-Camera PSNR: Baseline vs Depth TSDF vs Depth+Cutter", fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "12_per_camera_psnr.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}")


def plot_metrics_table(data: dict):
    rows = []
    for exp_id in data:
        m = data[exp_id]["mean"]
        rows.append([
            data[exp_id]["label"].replace("\n", " "),
            f"{m['psnr']:.2f}",
            f"{m['ssim']:.4f}",
            f"{m['lpips']:.4f}" if m["lpips"] else "N/A",
        ])

    col_labels = ["Experiment", "PSNR (dB) ↑", "SSIM ↑", "LPIPS ↓"]
    fig, ax = plt.subplots(figsize=(12, max(3, len(rows) * 0.6 + 1.5)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.0)

    header_color = "#2c3e50"
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 1:
            cell.set_facecolor("#f0f4f8")

    ax.set_title("Image Quality Metrics Summary (mean over 8 cameras)", fontweight="bold", pad=15)
    out = PLOTS_DIR / "13_metrics_table.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}")

    # Also save as JSON summary
    summary = {exp_id: {"label": data[exp_id]["label"].replace("\n", " "), **data[exp_id]["mean"]}
               for exp_id in data}
    with open(PLOTS_DIR / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  → metrics_summary.json")


def main():
    print("=" * 50)
    print("Metrics Plot Generation")
    print("=" * 50)
    data = load_metrics()
    if not data:
        print("No metrics JSON found in", METRICS_DIR)
        return
    print(f"  Loaded {len(data)} experiments")
    plot_bar_comparison(data)
    plot_per_camera_breakdown(data)
    plot_metrics_table(data)
    print("Done.")


if __name__ == "__main__":
    main()
