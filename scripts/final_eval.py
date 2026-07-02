#!/usr/bin/env python3
"""Final thesis evaluation for the dimpled-sphere depth-camera ablation.

Consolidates the whole story into one CSV + one 2-panel figure:
  (left)  global symmetric Chamfer vs #depth cameras, with the RGB-only (N=0) baseline.
  (right) region-split COMPLETENESS (GT->recon) for POCKET vs SMOOTH regions vs #depth cameras.

Conclusion this supports: depth carving monotonically improves the concave POCKET regions it is
designed for, while the SMOOTH convex surface is over-carved by oblique cameras under a single
global threshold -> global Chamfer is non-monotonic and the wrong headline metric.

Reads the latest carve build's recons (16_grazing_deepair) + RGB-only baseline (17).
"""
import itertools, math
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "output/carve_experiments/16_grazing_deepair"
RGB = ROOT / "output/carve_experiments/17_rgb_only_baseline"
OUT = ROOT / "output/FINAL_evaluation"
OUT.mkdir(parents=True, exist_ok=True)
SAMPLES = 200000
POCKETS = 12


def fibonacci_dirs(n):
    phi = math.pi * (math.sqrt(5.0) - 1.0)
    out = []
    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2 if n > 1 else 1.0
        r = math.sqrt(max(0.0, 1 - y * y))
        th = phi * i
        out.append((math.cos(th) * r, y, math.sin(th) * r))
    v = np.array(out, np.float64)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def load_recon(folder):
    V = np.fromfile(folder / "vertices.bin", dtype=np.float32).reshape(-1, 3)
    F = np.fromfile(folder / "faces.bin", dtype=np.int64).reshape(-1, 3)
    return trimesh.Trimesh(vertices=V, faces=F, process=False)


def sample(mesh, n, seed=0):
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(pts)


def min_axis_angle_deg(pts, axes):
    d = pts / np.clip(np.linalg.norm(pts, axis=1, keepdims=True), 1e-9, None)
    cos = d @ axes.T
    return np.degrees(np.arccos(np.clip(cos.max(axis=1), -1, 1)))


AXES = fibonacci_dirs(POCKETS)
GT = trimesh.load(str(EXP / "ground_truth.obj"), process=False)
GP = sample(GT, SAMPLES, 0)
GP_ang = min_axis_angle_deg(GP, AXES)
gt_pocket = GP_ang < 18.0
gt_smooth = GP_ang > 32.0


def eval_recon(folder):
    rp = sample(load_recon(folder), SAMPLES, 0)
    # resolve recon<->GT mirror flip by min symmetric chamfer
    best = None
    for s in itertools.product((1, -1), repeat=3):
        S = np.array(s, float)
        c = cKDTree(rp * S).query(GP)[0].mean() + cKDTree(GP).query(rp * S)[0].mean()
        if best is None or c < best[0]:
            best = (c, S)
    chamfer_mm = best[0] * 1000.0
    rpf = rp * best[1]
    tg, tr = cKDTree(GP), cKDTree(rpf)
    acc_mm = tg.query(rpf)[0] * 1000.0           # recon -> GT (accuracy)
    comp_mm = tr.query(GP)[0] * 1000.0           # GT -> recon (completeness)
    rp_ang = min_axis_angle_deg(rpf, AXES)
    rp_pocket, rp_smooth = rp_ang < 18.0, rp_ang > 32.0
    return dict(
        chamfer=chamfer_mm,
        pocket_comp=comp_mm[gt_pocket].mean(),
        smooth_comp=comp_mm[gt_smooth].mean(),
        pocket_acc=acc_mm[rp_pocket].mean(),
        smooth_acc=acc_mm[rp_smooth].mean(),
    )


rows = []
# N=0 RGB-only baseline
r0 = eval_recon(RGB / "recon/frame_00000")
rows.append((0, r0))
for N in (1, 2, 4, 8, 12):
    rows.append((N, eval_recon(EXP / f"recon_d{N}/frame_00000")))

# write CSV
csv = OUT / "final_metrics.csv"
with open(csv, "w") as f:
    f.write("num_depth,chamfer_mm,pocket_completeness_mm,smooth_completeness_mm,"
            "pocket_accuracy_mm,smooth_accuracy_mm\n")
    for N, r in rows:
        f.write(f"{N},{r['chamfer']:.3f},{r['pocket_comp']:.3f},{r['smooth_comp']:.3f},"
                f"{r['pocket_acc']:.3f},{r['smooth_acc']:.3f}\n")

Ns = [N for N, _ in rows]
cham = [r["chamfer"] for _, r in rows]
pc = [r["pocket_comp"] for _, r in rows]
sc = [r["smooth_comp"] for _, r in rows]
pa = [r["pocket_acc"] for _, r in rows]
sa = [r["smooth_acc"] for _, r in rows]

# plot
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
ax[0].axhline(cham[0], ls="--", c="gray", label=f"RGB-only (N=0): {cham[0]:.1f} mm")
ax[0].plot(Ns[1:], cham[1:], "o-", c="C0", label="RGB + depth")
ax[0].set_xlabel("# depth cameras"); ax[0].set_ylabel("global Chamfer (mm)")
ax[0].set_title("Global Chamfer vs depth-camera count")
ax[0].legend(); ax[0].grid(alpha=0.3)

ax[1].plot(Ns, pc, "o-", c="C3", label="pocket (concave) completeness")
ax[1].plot(Ns, sc, "s-", c="C2", label="smooth (convex) completeness")
ax[1].plot(Ns, pa, "o--", c="C3", alpha=0.5, label="pocket accuracy")
ax[1].plot(Ns, sa, "s--", c="C2", alpha=0.5, label="smooth accuracy")
ax[1].set_xlabel("# depth cameras"); ax[1].set_ylabel("region error (mm)")
ax[1].set_title("Region-split error vs depth-camera count")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "final_ablation.png", dpi=130)

print("== FINAL METRICS ==")
print(f"{'N':>3} {'chamfer':>8} {'pock_comp':>9} {'smooth_comp':>11} {'pock_acc':>8} {'smooth_acc':>10}")
for N, r in rows:
    print(f"{N:>3} {r['chamfer']:>8.2f} {r['pocket_comp']:>9.2f} {r['smooth_comp']:>11.2f} "
          f"{r['pocket_acc']:>8.2f} {r['smooth_acc']:>10.2f}")
print(f"\nwrote {csv}\nwrote {OUT/'final_ablation.png'}")
