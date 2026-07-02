#!/usr/bin/env python3
"""Box-fill presentation slides: sweep the concavity carve offset so the open"""
import subprocess, os, sys
from pathlib import Path

ROOT = Path("/home/timnogga/bachelorthesis/biwi_kinect_head_pose/RIFTCast")
CFG = ROOT / "configs/real_02_tsdf_edge_gradient.json"
OUT = ROOT / "output/box_fill/slides"
EXPORTER = ROOT / "bin/RIFTCast_exporter"

# fill up: open -> filled
OFFSETS = [0.24, 0.21, 0.18, 0.15, 0.12, 0.09, 0.05, 0.00]

# fixed depth view into the box opening (-y), with locked colour scale
CENTER = "1.125,1.467,2.365"
EYE = "1.30,0.92,2.50"
YFOV = "0.62"
RES = "1000"
CMAP = "turbo"
VMIN, VMAX = "0.42", "1.15"

OUT.mkdir(parents=True, exist_ok=True)


def run(cmd, **kw):
    return subprocess.run(cmd, **kw)


def main():
    for i, off in enumerate(OFFSETS, 1):
        tag = f"{off:.2f}".replace(".", "")
        recon_dir = OUT / f"recon_off{tag}"
        obj = OUT / f"box_off{tag}.obj"
        png = OUT / f"slide_{i:02d}_off{tag}.png"
        print(f"\n=== slide {i}/{len(OFFSETS)}  offset={off:.2f} ===")

        env = dict(os.environ, RIFT_CARVE_OFFSET=f"{off:.4f}")
        log = OUT / f"recon_off{tag}.log"
        with open(log, "w") as lf:
            r = run([str(EXPORTER), "--dataset", str(CFG),
                     "--output", str(recon_dir), "--frame", "0", "--inpaint", "false"],
                    env=env, stdout=lf, stderr=subprocess.STDOUT)
        frame = recon_dir / "frame_00000"
        if r.returncode != 0 or not (frame / "vertices.bin").exists():
            print(f"  !! exporter failed (see {log})"); continue

        run([sys.executable, str(ROOT / "eval/recon_to_obj.py"),
             "--recon", str(frame), "--out", str(obj)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # exporter uses the host's NVIDIA GL (above); pyrender needs a virtual
        # X server, so wrap ONLY the render step in xvfb.
        run(["xvfb-run", "-a", "-s", "-screen 0 1280x1280x24",
             sys.executable, str(ROOT / "figures/render_depth_view.py"),
             "--mesh", str(obj), "--out", str(png),
             "--center", CENTER, "--eye", EYE, "--yfov", YFOV, "--res", RES,
             "--cmap", CMAP, "--vmin", VMIN, "--vmax", VMAX])
        print(f"  -> {png.name}")

    print("\nslides in", OUT)


if __name__ == "__main__":
    main()
