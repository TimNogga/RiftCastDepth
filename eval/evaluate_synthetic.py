#!/usr/bin/env python3
"""
Synthetic dataset evaluation – compares reconstructed renders against GT images.

Three runs evaluated (synthetic_open_cube_8cams, frame 0):
  no_depth   – visual hull, no depth
  real_data  – TSDF, real-data preset
  synth_data – TSDF, synthetic-data preset

Run from repo root:
    python3 eval/evaluate_synthetic.py

Outputs go to output/evaluation/synthetic/.
"""

from pathlib import Path
import json
import math
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import lpips as lpips_lib
    import torch
    _LPIPS_NET = None
    def _get_lpips():
        global _LPIPS_NET
        if _LPIPS_NET is None:
            _LPIPS_NET = lpips_lib.LPIPS(net="alex")
            if torch.cuda.is_available():
                _LPIPS_NET = _LPIPS_NET.cuda()
        return _LPIPS_NET
    def _lpips(a, b):
        net = _get_lpips()
        def t(x):
            x = torch.from_numpy(x).float() / 127.5 - 1.0
            x = x.permute(2, 0, 1).unsqueeze(0)
            return x.cuda() if torch.cuda.is_available() else x
        with torch.no_grad():
            return float(net(t(a), t(b)).item())
    HAS_LPIPS = True
except ModuleNotFoundError:
    HAS_LPIPS = False
    def _lpips(a, b): return float("nan")

try:
    from skimage.metrics import structural_similarity as _sk_ssim
    def _ssim(a, b): return float(_sk_ssim(a, b, channel_axis=2, data_range=255))
except ModuleNotFoundError:
    def _ssim(a, b):
        c1, c2 = (0.01*255)**2, (0.03*255)**2
        a, b = a.astype(np.float64), b.astype(np.float64)
        k = 11
        mu_a = cv2.GaussianBlur(a,(k,k),1.5); mu_b = cv2.GaussianBlur(b,(k,k),1.5)
        sig_a2 = cv2.GaussianBlur(a*a,(k,k),1.5)-mu_a**2
        sig_b2 = cv2.GaussianBlur(b*b,(k,k),1.5)-mu_b**2
        sig_ab = cv2.GaussianBlur(a*b,(k,k),1.5)-mu_a*mu_b
        return float(((2*mu_a*mu_b+c1)*(2*sig_ab+c2)/((mu_a**2+mu_b**2+c1)*(sig_a2+sig_b2+c2))).mean())

REPO_ROOT  = Path(__file__).resolve().parents[2]
DATASET    = REPO_ROOT / "data" / "synthetic_open_cube_8cams" / "frame_00000"
OUT_DIR    = REPO_ROOT / "output" / "evaluation" / "synthetic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "figure.facecolor": "white"})

RUNS = {
    "no_depth":   REPO_ROOT / "runtime/outputs/runtime_depth_fusion_mode_none_no_depth",
    "depth_real": REPO_ROOT / "runtime/outputs/runtime_depth_fusion_mode_real_data",
    "depth_synth":REPO_ROOT / "runtime/outputs/runtime_depth_fusion_mode_synthetic_data",
}
RUN_LABELS = {
    "no_depth":    "No Depth\n(Baseline)",
    "depth_real":  "Depth TSDF\n(real preset)",
    "depth_synth": "Depth TSDF\n(synth preset)",
}
RUN_COLORS = {"no_depth": "#e74c3c", "depth_real": "#3498db", "depth_synth": "#2ecc71"}

# Camera ID → GT name mapping (colour cameras only)
CAM_MAP = {0: "C000", 1: "C001", 2: "C002", 3: "C003", 4: "C004", 5: "C005"}
MASK_H, MASK_W = 1080, 1920


# ── helpers ────────────────────────────────────────────────────────────────

def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(cam_name: str) -> np.ndarray | None:
    p = DATASET / "mask" / f"mask_{cam_name}.bin"
    if not p.exists():
        return None
    raw = np.fromfile(p, dtype=np.uint8)
    if raw.size == MASK_H * MASK_W:
        return np.flipud(raw.reshape(MASK_H, MASK_W)) > 127
    return None


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    return 20 * math.log10(255.0 / math.sqrt(mse)) if mse > 0 else float("inf")


def load_mesh_stats(run_dir: Path) -> dict:
    vp = run_dir / "frame_00000/vertices.bin"
    fp = run_dir / "frame_00000/faces.bin"
    if not vp.exists():
        return {}
    v = np.fromfile(vp, dtype=np.float32).reshape(-1, 3)
    f = np.fromfile(fp, dtype=np.int64).reshape(-1, 3)
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1-v0, v2-v0), axis=1)
    edges = np.concatenate([np.linalg.norm(v1-v0, axis=1),
                            np.linalg.norm(v2-v1, axis=1),
                            np.linalg.norm(v0-v2, axis=1)])
    return {
        "vertices":        int(len(v)),
        "faces":           int(len(f)),
        "surface_area_m2": float(areas.sum()),
        "mean_edge_mm":    float(edges.mean() * 1000),
        "degenerate":      int((areas < 1e-10).sum()),
    }


# ── evaluation ─────────────────────────────────────────────────────────────

def evaluate_run(run_name: str, run_dir: Path) -> dict:
    rgb_dir = run_dir / "frame_00000" / "rgb"
    cam_results = []

    for cam_id, cam_name in CAM_MAP.items():
        pred_path = rgb_dir / f"rgb_{cam_id}.png"
        gt_path   = DATASET / "rgb" / f"{cam_name}.jpg"
        if not pred_path.exists() or not gt_path.exists():
            print(f"    skip {cam_name} (missing file)")
            continue

        pred = load_image(pred_path)
        gt   = load_image(gt_path)
        if pred.shape[:2] != gt.shape[:2]:
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_AREA)

        mask = load_mask(cam_name)
        if mask is not None:
            pred_m = np.zeros_like(pred); gt_m = np.zeros_like(gt)
            pred_m[mask] = pred[mask];    gt_m[mask]  = gt[mask]
        else:
            pred_m, gt_m = pred, gt

        r = {
            "cam_id":   cam_id,
            "cam_name": cam_name,
            "psnr":     psnr(pred_m, gt_m),
            "ssim":     _ssim(pred_m, gt_m),
            "lpips":    _lpips(pred_m, gt_m) if HAS_LPIPS else float("nan"),
        }
        cam_results.append(r)
        print(f"    {cam_name}  PSNR={r['psnr']:6.2f}  SSIM={r['ssim']:.4f}  LPIPS={r['lpips']:.4f}")

    psnrs  = [r["psnr"]  for r in cam_results]
    ssims  = [r["ssim"]  for r in cam_results]
    lpipss = [r["lpips"] for r in cam_results if not math.isnan(r["lpips"])]
    mean = {
        "psnr":  float(np.mean(psnrs))  if psnrs  else None,
        "ssim":  float(np.mean(ssims))  if ssims  else None,
        "lpips": float(np.mean(lpipss)) if lpipss else None,
    }
    print(f"    MEAN  PSNR={mean['psnr']:6.2f}  SSIM={mean['ssim']:.4f}  LPIPS={mean['lpips']:.4f}")

    mesh = load_mesh_stats(run_dir)
    return {"run": run_name, "cameras": cam_results, "mean": mean, "mesh": mesh}


# ── plots ──────────────────────────────────────────────────────────────────

def plot_bar_metrics(all_results: dict):
    names  = list(all_results.keys())
    labels = [RUN_LABELS[n].replace("\n", " ") for n in names]
    colors = [RUN_COLORS[n] for n in names]
    psnrs  = [all_results[n]["mean"]["psnr"]  for n in names]
    ssims  = [all_results[n]["mean"]["ssim"]  for n in names]
    lpipss = [all_results[n]["mean"]["lpips"] or 0 for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Synthetic Dataset – Image Quality vs GT (mean over 6 cameras)", fontweight="bold")
    x = np.arange(len(names))
    for ax, vals, title, better in zip(axes,
                                       [psnrs, ssims, lpipss],
                                       ["PSNR (dB) ↑", "SSIM ↑", "LPIPS ↓"],
                                       [True, True, False]):
        bars = ax.bar(x, vals, color=colors, width=0.5, edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax.set_title(title, fontweight="bold"); ax.grid(True, axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        best = np.argmax(vals) if better else np.argmin(vals)
        bars[best].set_edgecolor("#2c3e50"); bars[best].set_linewidth(2.5)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "01_metrics_bar.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  → 01_metrics_bar.png")


def plot_per_camera(all_results: dict):
    names = list(all_results.keys())
    cam_names = [r["cam_name"] for r in all_results[names[0]]["cameras"]]
    x = np.arange(len(cam_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, name in enumerate(names):
        psnrs = [r["psnr"] for r in all_results[name]["cameras"]]
        ax.bar(x + i*width, psnrs, width, label=RUN_LABELS[name].replace("\n", " "),
               color=RUN_COLORS[name], alpha=0.85)
    ax.set_xticks(x + width); ax.set_xticklabels(cam_names)
    ax.set_ylabel("PSNR (dB)"); ax.set_title("Per-Camera PSNR vs GT", fontweight="bold")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "02_per_camera_psnr.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  → 02_per_camera_psnr.png")


def plot_mesh_comparison(all_results: dict):
    names  = list(all_results.keys())
    labels = [RUN_LABELS[n].replace("\n", " ") for n in names]
    colors = [RUN_COLORS[n] for n in names]
    verts  = [all_results[n]["mesh"].get("vertices", 0) for n in names]
    areas  = [all_results[n]["mesh"].get("surface_area_m2", 0) for n in names]
    x = np.arange(len(names))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Synthetic Dataset – Mesh Complexity", fontweight="bold")
    for ax, vals, title, unit in zip(axes, [verts, areas], ["Vertex Count", "Surface Area [m²]"], ["", " m²"]):
        bars = ax.bar(x, vals, color=colors, width=0.5, edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax.set_title(title, fontweight="bold"); ax.grid(True, axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            label = f"{v:,}{unit}" if unit == "" else f"{v:.3f}{unit}"
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.005,
                    label, ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "03_mesh_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  → 03_mesh_comparison.png")


def plot_summary_table(all_results: dict):
    names = list(all_results.keys())
    rows = []
    for n in names:
        m = all_results[n]["mean"]
        mesh = all_results[n]["mesh"]
        rows.append([
            RUN_LABELS[n].replace("\n", " "),
            f"{mesh.get('vertices', '?'):,}",
            f"{mesh.get('surface_area_m2', 0):.3f}",
            f"{m['psnr']:.2f}",
            f"{m['ssim']:.4f}",
            f"{m['lpips']:.4f}" if m["lpips"] else "N/A",
        ])
    col_labels = ["Configuration", "Vertices", "Surface [m²]", "PSNR ↑", "SSIM ↑", "LPIPS ↓"]
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 2.2)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 1:
            cell.set_facecolor("#f0f4f8")
    ax.set_title("Synthetic Dataset Evaluation Summary", fontweight="bold", pad=15)
    fig.savefig(OUT_DIR / "04_summary_table.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  → 04_summary_table.png")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("Synthetic Dataset Evaluation")
    print("=" * 55)

    all_results = {}
    for run_name, run_dir in RUNS.items():
        if not run_dir.exists():
            print(f"  SKIP {run_name} – output dir not found")
            continue
        print(f"\n[{run_name}]")
        all_results[run_name] = evaluate_run(run_name, run_dir)

    if not all_results:
        print("No results computed."); return

    print("\nGenerating plots …")
    plot_bar_metrics(all_results)
    plot_per_camera(all_results)
    plot_mesh_comparison(all_results)
    plot_summary_table(all_results)

    summary = {
        n: {"label": RUN_LABELS[n].replace("\n", " "),
            "mean": all_results[n]["mean"],
            "mesh": all_results[n]["mesh"]}
        for n in all_results
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  → results.json")
    print(f"\nDone. Output → {OUT_DIR}")


if __name__ == "__main__":
    main()
