#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np

def generate_open_box_walls(xmin, xmax, ymin, ymax, zmin, zmax, thickness):
    """
    Exact same bounding box logic used in the synthetic dataset generator.
    Keeps mathematical alignment 100% perfect.
    """
    walls = []

    # Left wall
    walls.append((
        np.array([xmin, ymin, zmin], dtype=np.float32),
        np.array([xmin + thickness, ymax, zmax], dtype=np.float32),
    ))

    # Right wall
    walls.append((
        np.array([xmax - thickness, ymin, zmin], dtype=np.float32),
        np.array([xmax, ymax, zmax], dtype=np.float32),
    ))

    # Bottom wall
    walls.append((
        np.array([xmin, ymin, zmin], dtype=np.float32),
        np.array([xmax, ymin + thickness, zmax], dtype=np.float32),
    ))

    # Top wall
    walls.append((
        np.array([xmin, ymax - thickness, zmin], dtype=np.float32),
        np.array([xmax, ymax, zmax], dtype=np.float32),
    ))

    # Back wall
    walls.append((
        np.array([xmin, ymin, zmin], dtype=np.float32),
        np.array([xmax, ymax, zmin + thickness], dtype=np.float32),
    ))

    return walls

def append_box_to_obj(f, bmin, bmax, start_idx):
    """
    Takes a bounding box and writes its 8 vertices and 6 faces (quads) 
    to the open file handle, maintaining proper counter-clockwise winding.
    """
    # Define the 8 corners of the box
    vertices = [
        (bmin[0], bmin[1], bmin[2]), # 0: left bottom back
        (bmax[0], bmin[1], bmin[2]), # 1: right bottom back
        (bmax[0], bmax[1], bmin[2]), # 2: right top back
        (bmin[0], bmax[1], bmin[2]), # 3: left top back
        (bmin[0], bmin[1], bmax[2]), # 4: left bottom front
        (bmax[0], bmin[1], bmax[2]), # 5: right bottom front
        (bmax[0], bmax[1], bmax[2]), # 6: right top front
        (bmin[0], bmax[1], bmax[2]), # 7: left top front
    ]
    
    # Write vertices
    for v in vertices:
        f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

    # Define the 6 faces using 0-based indexing (counter-clockwise pointing OUT)
    faces = [
        (0, 3, 2, 1), # Back (-Z)
        (4, 5, 6, 7), # Front (+Z)
        (0, 1, 5, 4), # Bottom (-Y)
        (3, 7, 6, 2), # Top (+Y)
        (0, 4, 7, 3), # Left (-X)
        (1, 2, 6, 5), # Right (+X)
    ]
    
    # Write faces (OBJ uses 1-based indexing)
    for face in faces:
        idx = [i + start_idx for i in face]
        f.write(f"f {idx[0]} {idx[1]} {idx[2]} {idx[3]}\n")

    # Return the updated starting index for the next box
    return start_idx + 8

def main():
    parser = argparse.ArgumentParser(description="Generate Ground Truth OBJ for the synthetic open box.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ground_truth_open_box.obj"),
        help="Path to save the output .obj file"
    )
    args = parser.parse_args()

    # Define the exact same dimensions used in your dataset script
    walls = generate_open_box_walls(
        xmin=-0.5,
        xmax=0.5,
        ymin=-0.5,
        ymax=0.5,
        zmin=-0.5,
        zmax=0.5,
        thickness=0.08,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w") as f:
        f.write("# Ground Truth Open Box (5 thick walls)\n")
        f.write("# Matches synthetic dataset boundaries exactly.\n")
        
        vertex_start_idx = 1
        for i, (bmin, bmax) in enumerate(walls):
            f.write(f"\n# Wall {i+1}\n")
            vertex_start_idx = append_box_to_obj(f, bmin, bmax, vertex_start_idx)

    print(f"Ground truth mesh successfully written to: {args.output.resolve()}")

if __name__ == "__main__":
    main()