#!/usr/bin/env python3
"""Offscreen multi-view renders of the GT and reconstructions for the thesis evaluation figure."""
import argparse, itertools, math, os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
from pathlib import Path
import numpy as np
import trimesh
import pyrender
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = 480


def best_flip(recon_v, gt_v):
    gv = gt_v[np.random.default_rng(0).choice(len(gt_v), min(40000, len(gt_v)), replace=False)]
    rv = recon_v[np.random.default_rng(1).choice(len(recon_v), min(40000, len(recon_v)), replace=False)]
    gt_tree = cKDTree(gv); best = None
    for s in itertools.product((1, -1), repeat=3):
        S = np.array(s, float); rp = rv * S
        d1, _ = gt_tree.query(rp); d2, _ = cKDTree(rp).query(gv)
        c = d1.mean() + d2.mean()
        if best is None or c < best[0]:
            best = (c, S)
    return best[1]


def look_at(pos, target=np.zeros(3), up=np.array([0., 1., 0.])):
    z = pos - target; z = z / np.linalg.norm(z)
    if abs(np.dot(z, up)) > 0.95: up = np.array([0., 0., 1.])
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    M = np.eye(4); M[:3, 0] = x; M[:3, 1] = y; M[:3, 2] = z; M[:3, 3] = pos
    return M


def render_views(mesh, azimuths, elev_deg=18.0, dist=1.75):
    tm = mesh.copy(); tm.vertices -= tm.bounding_box.centroid
    imgs = []
    r = pyrender.OffscreenRenderer(RES, RES)
    for az in azimuths:
        a = math.radians(az); e = math.radians(elev_deg)
        pos = dist * np.array([math.cos(e) * math.cos(a), math.sin(e), math.cos(e) * math.sin(a)])
        pose = look_at(pos)
        scene = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[0.45, 0.45, 0.45])
        mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.80, 0.82, 0.86, 1.0],
                                                 metallicFactor=0.0, roughnessFactor=0.9)
        scene.add(pyrender.Mesh.from_trimesh(tm, material=mat, smooth=False))
        cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.2)
        scene.add(cam, pose=pose)
        # KEY light: offset ~40 deg azimuth and raised elevation from the view, so the sphere shows a
        # shading gradient and the concave pockets read as shaded dimples.
        ak = a - math.radians(42); ek = e + math.radians(38)
        key_pos = dist * np.array([math.cos(ek) * math.cos(ak), math.sin(ek), math.cos(ek) * math.sin(ak)])
        scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=4.0), pose=look_at(key_pos))
        color, _ = r.render(scene)
        imgs.append(color)
    r.delete()
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meshes-dir", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--ns", type=int, nargs="+", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--views", type=int, default=5)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    azimuths = list(np.linspace(0, 360, args.views, endpoint=False))

    gt = trimesh.load(str(args.gt), process=False)
    gt_v = np.asarray(gt.vertices)

    def save_strip(imgs, path, label=None):
        n = len(imgs)
        fig, ax = plt.subplots(1, n, figsize=(2.4 * n, 2.6))
        if n == 1: ax = [ax]
        for a, im, az in zip(ax, imgs, azimuths):
            a.imshow(im); a.set_xticks([]); a.set_yticks([]); a.set_title(f"{int(az)}°", fontsize=9)
        if label: fig.suptitle(label, fontsize=12)
        fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close()
        print(f"  {path}")

    rows = [("Ground truth", gt)]
    gt_imgs = render_views(gt, azimuths)
    save_strip(gt_imgs, args.out_dir / "gt_multiview.png", "Ground truth")
    all_imgs = [gt_imgs]

    for N in args.ns:
        mp = args.meshes_dir / f"recon_d{N}.obj"
        if not mp.exists():
            print(f"  N={N}: missing {mp}"); continue
        m = trimesh.load(str(mp), process=False)
        S = best_flip(np.asarray(m.vertices), gt_v)
        m.vertices = np.asarray(m.vertices) * S
        imgs = render_views(m, azimuths)
        save_strip(imgs, args.out_dir / f"recon_d{N}_multiview.png", f"{N} depth camera(s)")
        rows.append((f"{N} depth cam" + ("s" if N != 1 else ""), m)); all_imgs.append(imgs)

    # comparison grid: rows = meshes, cols = views
    nrows, ncols = len(all_imgs), len(azimuths)
    fig, ax = plt.subplots(nrows, ncols, figsize=(2.3 * ncols, 2.4 * nrows))
    if nrows == 1: ax = ax[None, :]
    for i, (lab, _) in enumerate(rows):
        for j, az in enumerate(azimuths):
            ax[i, j].imshow(all_imgs[i][j]); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if i == 0: ax[i, j].set_title(f"{int(az)}°", fontsize=10)
            if j == 0: ax[i, j].set_ylabel(lab, fontsize=11)
    fig.suptitle("Dimpled-sphere: ground truth vs reconstruction (depth-camera count)", fontsize=14)
    fig.tight_layout(); fig.savefig(args.out_dir / "comparison_grid.png", dpi=140, bbox_inches="tight"); plt.close()
    print(f"  {args.out_dir / 'comparison_grid.png'}")


if __name__ == "__main__":
    main()
