#!/usr/bin/env python3
"""Region-split reconstruction diagnostic for the dimpled sphere.

Splits error into POCKET regions (near one of the 12 fibonacci pocket axes, cap half-angle ~17 deg)
vs SMOOTH sphere, and reports BOTH directions separately:
  - accuracy     (recon -> GT): recon points far from the true surface  => wrong-place geometry
  - completeness (GT   -> recon): true-surface points far from the recon => missing surface
Over-carving the convex surface shows up as SMOOTH-region error; pockets-not-deep-enough shows up as
POCKET-region error. Auto-resolves the recon<->GT axis-flip the same way eval/chamfer.py does.
"""
import argparse, itertools, math
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree


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


def load_recon(folder: Path):
    V = np.fromfile(folder / "vertices.bin", dtype=np.float32).reshape(-1, 3)
    F = np.fromfile(folder / "faces.bin", dtype=np.int64).reshape(-1, 3)
    return trimesh.Trimesh(vertices=V, faces=F, process=False)


def sample(mesh, n, seed=0):
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(pts)


def min_axis_angle_deg(pts, axes):
    """For each point, the smallest angle (deg) between its direction and any pocket axis."""
    d = pts / np.clip(np.linalg.norm(pts, axis=1, keepdims=True), 1e-9, None)
    cos = d @ axes.T                      # (P, 12)
    return np.degrees(np.arccos(np.clip(cos.max(axis=1), -1, 1)))


def stats(dist_mm, ang, pocket_max=18.0, smooth_min=32.0):
    pk = ang < pocket_max
    sm = ang > smooth_min
    def m(mask):
        return (dist_mm[mask].mean() if mask.any() else float("nan"), int(mask.sum()))
    return {"pocket": m(pk), "smooth": m(sm), "all": (dist_mm.mean(), len(dist_mm))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=200000)
    ap.add_argument("--pockets", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    axes = fibonacci_dirs(args.pockets)
    recon = load_recon(args.recon)
    gt = trimesh.load(str(args.gt), process=False)
    rp = sample(recon, args.samples, args.seed)
    gp = sample(gt, args.samples, args.seed)

    # resolve the axis sign-flip (recon frame mirrored vs GT) by min symmetric chamfer
    best = None
    for s in itertools.product((1, -1), repeat=3):
        S = np.array(s, float)
        c = cKDTree(rp * S).query(gp)[0].mean() + cKDTree(gp).query(rp * S)[0].mean()
        if best is None or c < best[0]:
            best = (c, S)
    S = best[1]
    rpf = rp * S

    tg, tr = cKDTree(gp), cKDTree(rpf)
    acc_mm = tg.query(rpf)[0] * 1000.0          # recon -> GT
    comp_mm = tr.query(gp)[0] * 1000.0          # GT -> recon
    acc = stats(acc_mm, min_axis_angle_deg(rpf, axes))
    comp = stats(comp_mm, min_axis_angle_deg(gp, axes))

    print(f"flip={tuple(int(x) for x in S)}  pocket<18deg  smooth>32deg")
    print("                         pocket            smooth            all")
    def row(name, d):
        (pm, pn), (sm, sn), (am, an) = d["pocket"], d["smooth"], d["all"]
        print(f"  {name:22s} {pm:6.2f}mm n={pn:<6d} {sm:6.2f}mm n={sn:<6d} {am:6.2f}mm")
    row("accuracy  recon->GT", acc)
    row("completeness GT->recon", comp)


if __name__ == "__main__":
    main()
