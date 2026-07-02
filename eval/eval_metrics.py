#!/usr/bin/env python3
"""Depth-camera-count evaluation for the dimpled-sphere dataset.

Computes, per depth-camera count N, a set of meaningful reconstruction metrics against the GT mesh
and writes a CSV + plots. Metrics:
  - chamfer_global_mm          : symmetric Chamfer over the whole surface
  - chamfer_pocket_mm          : symmetric Chamfer restricted to the concavity (pocket) region
  - completeness_pocket_mm     : mean GT->recon distance in pocket region  (does depth FILL the pockets)
  - accuracy_pocket_mm         : mean recon->GT distance in pocket region
  - fscore_global / fscore_pocket @ tau : precision/recall F-score at a distance threshold
  - pocket_depth_mm            : how deep the carve reaches in the pocket directions (recovery of concavity)

Region split uses the angle to the 12 known pocket axes (<18 deg = pocket, >32 deg = smooth).
The recon frame can be sign-mirrored vs GT (cv_to_gl); we resolve the best axis sign-flip first so
region assignment and metrics are correct.
"""
import argparse, csv, itertools, math
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPHERE_R = 0.5
N_POCKETS = 12


def fib(n, r=1.0):
    out = []; phi = math.pi * (math.sqrt(5) - 1)
    for i in range(n):
        y = 1 - (i / (n - 1)) * 2 if n > 1 else 1.0
        rr = math.sqrt(max(0.0, 1 - y * y)); th = phi * i
        out.append((math.cos(th) * rr * r, y * r, math.sin(th) * rr * r))
    return np.array(out)


POCKET_AXES = fib(N_POCKETS, 1.0); POCKET_AXES /= np.linalg.norm(POCKET_AXES, axis=1, keepdims=True)


def sample_mesh(path, n=200000, seed=0):
    m = trimesh.load(str(path), process=False)
    p, _ = trimesh.sample.sample_surface(m, n, seed=seed)
    return np.asarray(p)


def best_flip(recon, gt):
    """Resolve recon<->GT mirror: pick axis sign-flip minimising symmetric Chamfer."""
    gt_tree = cKDTree(gt); best = None
    for s in itertools.product((1, -1), repeat=3):
        S = np.array(s, float); rp = recon * S
        d1, _ = gt_tree.query(rp); d2, _ = cKDTree(rp).query(gt)
        c = d1.mean() + d2.mean()
        if best is None or c < best[0]:
            best = (c, S)
    return best[1]


def region_mask(pts):
    d = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip(d @ POCKET_AXES.T, -1, 1))).min(1)
    return ang < 18, ang > 32  # pocket, smooth


def metrics_for(recon_pts, gt_pts, gt_tree, gp_pocket, gp_smooth, taus=(0.002, 0.005)):
    rtree = cKDTree(recon_pts)
    d_rg, _ = gt_tree.query(recon_pts)   # recon->GT  (accuracy)
    d_gr, _ = rtree.query(gt_pts)        # GT->recon  (completeness)
    rp_pocket, _ = region_mask(recon_pts)
    out = {
        "chamfer_global_mm": (d_rg.mean() + d_gr.mean()) * 1000,
        "chamfer_pocket_mm": (d_rg[rp_pocket].mean() + d_gr[gp_pocket].mean()) * 1000,
        "completeness_pocket_mm": d_gr[gp_pocket].mean() * 1000,
        "accuracy_pocket_mm": d_rg[rp_pocket].mean() * 1000,
        "completeness_smooth_mm": d_gr[gp_smooth].mean() * 1000,
        "accuracy_smooth_mm": d_rg[gp_smooth].mean() * 1000,
    }
    for tau in taus:
        prec = (d_rg < tau).mean(); rec = (d_gr < tau).mean()
        f = 2 * prec * rec / (prec + rec + 1e-12)
        prec_p = (d_rg[rp_pocket] < tau).mean(); rec_p = (d_gr[gp_pocket] < tau).mean()
        fp = 2 * prec_p * rec_p / (prec_p + rec_p + 1e-12)
        mm = int(tau * 1000)
        out[f"fscore_global_{mm}mm"] = f
        out[f"fscore_pocket_{mm}mm"] = fp
    # pocket carve depth: 5th-percentile recon radius in pocket directions (sphere=0.5 -> deeper=lower)
    rr = np.linalg.norm(recon_pts[rp_pocket], axis=1)
    out["pocket_min_radius"] = float(np.percentile(rr, 5)) if rr.size else float("nan")
    out["pocket_carve_depth_mm"] = (SPHERE_R - out["pocket_min_radius"]) * 1000 if rr.size else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meshes-dir", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--ns", type=int, nargs="+", required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--plots-dir", type=Path, required=True)
    args = ap.parse_args()

    gt_pts = sample_mesh(args.gt)
    gt_tree = cKDTree(gt_pts)
    gp_pocket, gp_smooth = region_mask(gt_pts)

    rows = []
    for N in args.ns:
        mp = args.meshes_dir / f"recon_d{N}.obj"
        if not mp.exists():
            print(f"  N={N}: missing {mp}, skipping"); continue
        rp = sample_mesh(mp)
        rp = rp * best_flip(rp, gt_pts)            # align to GT frame
        m = metrics_for(rp, gt_pts, gt_tree, gp_pocket, gp_smooth)
        m["num_depth"] = N
        rows.append(m)
        print(f"  N={N:2d}  chamfer={m['chamfer_global_mm']:.2f}mm  "
              f"cmp.pocket={m['completeness_pocket_mm']:.2f}  carve_depth={m['pocket_carve_depth_mm']:.1f}mm  "
              f"F@5mm={m['fscore_global_5mm']:.3f}")

    rows.sort(key=lambda r: r["num_depth"])
    cols = ["num_depth", "chamfer_global_mm", "chamfer_pocket_mm", "completeness_pocket_mm",
            "accuracy_pocket_mm", "completeness_smooth_mm", "accuracy_smooth_mm",
            "fscore_global_2mm", "fscore_pocket_2mm", "fscore_global_5mm", "fscore_pocket_5mm",
            "pocket_carve_depth_mm", "pocket_min_radius"]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow({k: r.get(k) for k in cols})
    print(f"wrote {args.out_csv}")

    Ns = [r["num_depth"] for r in rows]
    def series(k): return [r[k] for r in rows]
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    def lineplot(fname, specs, ylabel, title, lower_better=True):
        plt.figure(figsize=(7, 4.5))
        for key, lab, style in specs:
            plt.plot(Ns, series(key), style, lw=2, ms=7, label=lab)
        plt.xlabel("number of depth cameras"); plt.ylabel(ylabel); plt.title(title)
        plt.xticks(Ns); plt.grid(alpha=.3); plt.legend()
        if lower_better: plt.gca().margins(y=0.15)
        plt.tight_layout(); plt.savefig(args.plots_dir / fname, dpi=140); plt.close()
        print(f"  plot: {args.plots_dir / fname}")

    lineplot("chamfer_vs_cameras.png",
             [("chamfer_global_mm", "global", "o-"), ("chamfer_pocket_mm", "pocket region", "s--")],
             "Chamfer distance [mm]", "Chamfer vs depth-camera count")
    lineplot("pocket_recovery_vs_cameras.png",
             [("completeness_pocket_mm", "completeness (GT→recon)", "o-"),
              ("accuracy_pocket_mm", "accuracy (recon→GT)", "s--")],
             "pocket-region error [mm]", "Concavity recovery vs depth-camera count")
    lineplot("fscore_vs_cameras.png",
             [("fscore_global_5mm", "global @5mm", "o-"), ("fscore_pocket_5mm", "pocket @5mm", "s--"),
              ("fscore_global_2mm", "global @2mm", "o:"), ("fscore_pocket_2mm", "pocket @2mm", "s:")],
             "F-score", "F-score vs depth-camera count (higher = better)", lower_better=False)
    lineplot("pocket_carve_depth_vs_cameras.png",
             [("pocket_carve_depth_mm", "carve depth into pocket", "o-")],
             "pocket carve depth [mm]", "Concavity carve depth vs depth-camera count", lower_better=False)

    # combined 2x2 overview
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0,0].plot(Ns, series("chamfer_global_mm"), "o-", label="global")
    ax[0,0].plot(Ns, series("chamfer_pocket_mm"), "s--", label="pocket"); ax[0,0].set_title("Chamfer [mm]"); ax[0,0].legend()
    ax[0,1].plot(Ns, series("completeness_pocket_mm"), "o-", label="completeness")
    ax[0,1].plot(Ns, series("accuracy_pocket_mm"), "s--", label="accuracy"); ax[0,1].set_title("Pocket region error [mm]"); ax[0,1].legend()
    ax[1,0].plot(Ns, series("fscore_pocket_5mm"), "o-", label="pocket@5mm")
    ax[1,0].plot(Ns, series("fscore_global_5mm"), "s--", label="global@5mm"); ax[1,0].set_title("F-score"); ax[1,0].legend()
    ax[1,1].plot(Ns, series("pocket_carve_depth_mm"), "o-"); ax[1,1].set_title("Pocket carve depth [mm]")
    for a in ax.flat: a.set_xlabel("depth cameras"); a.set_xticks(Ns); a.grid(alpha=.3)
    fig.suptitle("Dimpled-sphere depth ablation — metrics vs depth-camera count", fontsize=14)
    fig.tight_layout(); fig.savefig(args.plots_dir / "combined_metrics.png", dpi=140); plt.close()
    print(f"  plot: {args.plots_dir / 'combined_metrics.png'}")


if __name__ == "__main__":
    main()
