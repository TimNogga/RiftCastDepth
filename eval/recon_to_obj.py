#!/usr/bin/env python3
"""Write an .obj from a RIFTCast recon frame folder (vertices.bin + faces.bin)."""
import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", type=Path, required=True, help="frame_00000 folder with vertices.bin/faces.bin")
    ap.add_argument("--out", type=Path, default=None, help="output .obj (default: <recon>/recon.obj)")
    args = ap.parse_args()

    V = np.fromfile(args.recon / "vertices.bin", dtype=np.float32).reshape(-1, 3)
    F = np.fromfile(args.recon / "faces.bin", dtype=np.int64).reshape(-1, 3)
    out = args.out or (args.recon / "recon.obj")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("# RIFTCast reconstruction\n")
        for v in V:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in F:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
    print(f"wrote {len(V)} verts, {len(F)} faces -> {out}")


if __name__ == "__main__":
    main()
