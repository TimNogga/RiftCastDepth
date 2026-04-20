#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def convert_bin_to_obj(vertices_path: Path, faces_path: Path, output_path: Path) -> None:
    vertices = np.fromfile(vertices_path, dtype=np.float32).reshape(-1, 3)
    faces = np.fromfile(faces_path, dtype=np.int64).reshape(-1, 3)

    with output_path.open("w") as f:
        f.write("# RIFTCast runtime mesh\n")
        f.write(f"# Vertices: {len(vertices)} | Faces: {len(faces)}\n\n")

        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def find_bin_pairs(search_root: Path):
    pairs = []
    for vertices_path in sorted(search_root.rglob("vertices.bin")):
        faces_path = vertices_path.with_name("faces.bin")
        if faces_path.exists():
            pairs.append((vertices_path, faces_path))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert all runtime vertices.bin/faces.bin files to OBJ meshes."
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Root folder to scan for vertices.bin/faces.bin pairs (default: runtime/outputs)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="voxel_based.obj",
        help="Output OBJ filename written next to each bin pair",
    )

    args = parser.parse_args()
    search_root = args.search_root.resolve()

    if not search_root.exists():
        print(f"Search root does not exist: {search_root}")
        return

    pairs = find_bin_pairs(search_root)
    if not pairs:
        print(f"No vertices.bin/faces.bin pairs found under: {search_root}")
        return

    converted = 0
    for vertices_path, faces_path in pairs:
        out_path = vertices_path.with_name(args.output_name)
        convert_bin_to_obj(vertices_path, faces_path, out_path)
        print(f"Wrote {out_path}")
        converted += 1

    print(f"Converted {converted} mesh folder(s).")


if __name__ == "__main__":
    main()