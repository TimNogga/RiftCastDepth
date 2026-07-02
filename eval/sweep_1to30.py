#!/usr/bin/env python3
"""Depth-camera ablation 1..30 on the dimpled sphere, WITHOUT re-rendering per N.

The depth cameras are placed by greedy farthest-point sampling, whose first-N prefix is nested
(prefix of N+1 contains prefix of N). So we render ONE dataset with 30 depth cameras, then for each
N=1..30 we hand the exporter a calibration file listing the 60 RGB cameras + the first N depth
cameras. Metrics: global symmetric Chamfer + region-split pocket/smooth completeness vs GT.

Uses the CURRENT bin/RIFTCast_exporter (= carve build 16_grazing_deepair). Outputs under
output/sweep_1to30/.
"""
import json, subprocess, itertools, math, os, sys, shutil
from pathlib import Path
_OUT_NAME = os.environ.get("SWEEP_OUT", "sweep_1to30")
import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "data/_dimpled_d30"
CFG_GEN = ROOT / "configs/_dimpled_d30.json"
OUT = ROOT / "output" / _OUT_NAME
GT = ROOT / "output/sweep_1to30/ground_truth.obj"  # shared GT (same object across builds)
NMAX = 30
LEVEL = 8
SAMPLES = 150000
POCKETS = 12
OUT.mkdir(parents=True, exist_ok=True)

EXPORT_ENV = {
    **os.environ,
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "__NV_PRIME_RENDER_OFFLOAD": "1",
    "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
}


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


def render_once():
    if (DS / "frame_00000" / "calibration_dome.json").exists() or (DS / "calibration_dome.json").exists():
        print(f"[render] reusing existing dataset {DS}")
        return
    print(f"[render] generating {DS} (60 RGB + {NMAX} depth) ...")
    subprocess.run([
        sys.executable, str(ROOT / "dataset/generate_dimpled_object.py"),
        "--num-rgb", "60", "--num-depth", str(NMAX), "--num-pockets", str(POCKETS),
        "--sphere-radius", "0.5", "--cam-radius", "2.5",
        "--dataset-root", str(DS), "--config-out", str(CFG_GEN),
        "--gt-out", str(GT), "--force",
    ], check=True)


def base_config():
    cfg = json.loads(CFG_GEN.read_text())
    cfg["reconstructor"]["level"] = LEVEL
    return cfg


def write_subset_calib(n):
    """Write calibration_dome_dN.json = all RGB + first N depth cameras."""
    full = json.loads((DS / "calibration_dome.json").read_text())["cameras"]
    rgb = [c for c in full if c["camera_type"] == "rgb"]
    dep = [c for c in full if c["camera_type"] == "depth"]
    dep.sort(key=lambda c: int(c["camera_id"][1:]))  # D000, D001, ... nested FPS order
    sub = rgb + dep[:n]
    p = DS / f"calibration_dome_d{n}.json"
    p.write_text(json.dumps({"cameras": sub}, indent=2))
    return p


def run_exporter(n):
    calib = write_subset_calib(n)
    cfg = base_config()
    cfg["dataset"]["camera_path"] = calib.name
    cfgp = ROOT / f"configs/_sweep30_d{n}.json"
    cfgp.write_text(json.dumps(cfg, indent=2))
    outdir = OUT / f"recon_d{n}"
    log = OUT / f"recon_d{n}.log"
    with open(log, "w") as lf:
        # Headless over SSH: the exporter needs a GL context, so wrap in a virtual X server.
        r = subprocess.run([
            "xvfb-run", "-a", "-s", "-screen 0 1280x720x24",
            str(ROOT / "bin/RIFTCast_exporter"),
            "--dataset", str(cfgp), "--output", str(outdir), "--frame", "0",
        ], env=EXPORT_ENV, stdout=lf, stderr=subprocess.STDOUT)
    frame = outdir / "frame_00000"
    ok = (frame / "vertices.bin").exists() and (frame / "faces.bin").exists()
    return frame if (r.returncode == 0 and ok) else None


# ---- metrics ----
AXES = fibonacci_dirs(POCKETS)


def load_recon(frame):
    V = np.fromfile(frame / "vertices.bin", dtype=np.float32).reshape(-1, 3)
    F = np.fromfile(frame / "faces.bin", dtype=np.int64).reshape(-1, 3)
    return trimesh.Trimesh(vertices=V, faces=F, process=False)


def min_axis_angle_deg(pts):
    d = pts / np.clip(np.linalg.norm(pts, axis=1, keepdims=True), 1e-9, None)
    return np.degrees(np.arccos(np.clip((d @ AXES.T).max(axis=1), -1, 1)))


def metrics(frame, gp, gt_pocket, gt_smooth):
    rp, _ = trimesh.sample.sample_surface(load_recon(frame), SAMPLES, seed=0)
    rp = np.asarray(rp)
    best = None
    for s in itertools.product((1, -1), repeat=3):
        S = np.array(s, float)
        c = cKDTree(rp * S).query(gp)[0].mean() + cKDTree(gp).query(rp * S)[0].mean()
        if best is None or c < best[0]:
            best = (c, S)
    rpf = rp * best[1]
    comp_mm = cKDTree(rpf).query(gp)[0] * 1000.0
    acc_mm = cKDTree(gp).query(rpf)[0] * 1000.0
    rp_ang = min_axis_angle_deg(rpf)
    return dict(
        chamfer=best[0] * 1000.0,
        pocket_comp=comp_mm[gt_pocket].mean(),
        smooth_comp=comp_mm[gt_smooth].mean(),
        smooth_acc=acc_mm[rp_ang > 32.0].mean(),
        nverts=len(load_recon(frame).vertices),
    )


def main():
    render_once()
    gt = trimesh.load(str(GT), process=False)
    gp, _ = trimesh.sample.sample_surface(gt, SAMPLES, seed=0)
    gp = np.asarray(gp)
    gang = min_axis_angle_deg(gp)
    gt_pocket, gt_smooth = gang < 18.0, gang > 32.0

    csvp = OUT / "sweep_1to30.csv"
    rows = []
    if os.environ.get("DEPTH_LIST"):
        depth_list = [int(x) for x in os.environ["DEPTH_LIST"].split(",")]
    else:
        depth_list = list(range(1, NMAX + 1))
    csvp.write_text("num_depth,chamfer_mm,pocket_completeness_mm,smooth_completeness_mm,smooth_accuracy_mm,nverts\n")
    for n in depth_list:
        frame = run_exporter(n)
        if frame is None:
            print(f"N={n:2d}  EXPORT FAILED (see log)")
            continue
        # write obj (standing rule)
        subprocess.run([sys.executable, str(ROOT / "eval/recon_to_obj.py"),
                        "--recon", str(frame), "--out", str(OUT / f"recon_d{n}.obj")],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        m = metrics(frame, gp, gt_pocket, gt_smooth)
        rows.append((n, m))
        with open(csvp, "a") as f:
            f.write(f"{n},{m['chamfer']:.3f},{m['pocket_comp']:.3f},{m['smooth_comp']:.3f},"
                    f"{m['smooth_acc']:.3f},{m['nverts']}\n")
        print(f"N={n:2d}  chamfer={m['chamfer']:6.2f}  pocket_comp={m['pocket_comp']:6.2f}  "
              f"smooth_acc={m['smooth_acc']:6.2f}  verts={m['nverts']}")
        # strip heavy raw export dir to save disk, keep vertices/faces
        for sub in ("rgb", "mask", "depth", "error", "primitives"):
            shutil.rmtree(frame / sub, ignore_errors=True)

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ns = [n for n, _ in rows]
    cham = [m["chamfer"] for _, m in rows]
    pc = [m["pocket_comp"] for _, m in rows]
    sa = [m["smooth_acc"] for _, m in rows]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].plot(Ns, cham, "o-", c="C0", ms=4)
    ax[0].set_xlabel("# depth cameras"); ax[0].set_ylabel("global Chamfer (mm)")
    ax[0].set_title("Global Chamfer vs depth-camera count (1–30)")
    ax[0].grid(alpha=0.3)
    ax[1].plot(Ns, pc, "o-", c="C3", ms=4, label="pocket (concave) completeness")
    ax[1].plot(Ns, sa, "s-", c="C2", ms=4, label="smooth (convex) accuracy")
    ax[1].set_xlabel("# depth cameras"); ax[1].set_ylabel("region error (mm)")
    ax[1].set_title("Region-split error vs depth-camera count (1–30)")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "sweep_1to30.png", dpi=130)
    print(f"\nwrote {csvp}\nwrote {OUT/'sweep_1to30.png'}")


if __name__ == "__main__":
    main()
