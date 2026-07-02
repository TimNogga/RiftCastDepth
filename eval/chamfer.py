#!/usr/bin/env python3
"""Symmetric Chamfer distance between a RIFTCast reconstruction and the ground-truth mesh."""
import argparse
import itertools
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def load_recon(folder: Path) -> trimesh.Trimesh:
    V = np.fromfile(folder / "vertices.bin", dtype=np.float32).reshape(-1, 3)
    F = np.fromfile(folder / "faces.bin", dtype=np.int64).reshape(-1, 3)
    return trimesh.Trimesh(vertices=V, faces=F, process=False)


def sample(mesh: trimesh.Trimesh, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(pts)


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Chamfer = mean(d(a->b)) + mean(d(b->a)), in metres."""
    ta, tb = cKDTree(a), cKDTree(b)
    d_ab, _ = tb.query(a)
    d_ba, _ = ta.query(b)
    return float(d_ab.mean() + d_ba.mean())


def best_aligned_chamfer(recon_pts: np.ndarray, gt_pts: np.ndarray):
    """Try the axis sign-flips (recon frame is mirrored vs GT) and return the min Chamfer."""
    best = None
    for sx, sy, sz in itertools.product((1, -1), repeat=3):
        S = np.array([sx, sy, sz], dtype=float)
        c = chamfer(recon_pts * S, gt_pts)
        if best is None or c < best[0]:
            best = (c, (sx, sy, sz))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", type=Path, required=True, help="frame_00000 folder with vertices.bin/faces.bin")
    ap.add_argument("--gt", type=Path, required=True, help="ground-truth .obj")
    ap.add_argument("--samples", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    recon = load_recon(args.recon)
    gt = trimesh.load(str(args.gt), process=False)

    rp = sample(recon, args.samples, args.seed)
    gp = sample(gt, args.samples, args.seed)

    c, flip = best_aligned_chamfer(rp, gp)
    print(f"chamfer={c*1000:.3f} mm   (flip={flip})")
    return c


if __name__ == "__main__":
    main()
