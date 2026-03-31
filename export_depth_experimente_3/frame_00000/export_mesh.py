import numpy as np
import os

def convert_bin_to_obj(vertices_path, faces_path, output_path="baseline_mesh.obj"):
    """
    Reads raw PyTorch tensor binary dumps and converts them to a 3D OBJ file.
    """
    if not os.path.exists(vertices_path) or not os.path.exists(faces_path):
        print("Error: Could not find the .bin files. Check your paths.")
        return

    print(f"Reading {vertices_path}...")
    vertices = np.fromfile(vertices_path, dtype=np.float32).reshape(-1, 3)
    print(f" -> Found {len(vertices)} vertices.")

    print(f"Reading {faces_path}...")
    faces = np.fromfile(faces_path, dtype=np.int64).reshape(-1, 3)
    print(f" -> Found {len(faces)} faces.")

    print(f"Exporting to {output_path}...")
    with open(output_path, 'w') as f:
        f.write("# RIFTCast Baseline Mesh\n")
        f.write(f"# Vertices: {len(vertices)} | Faces: {len(faces)}\n\n")
        
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

vert_file = "vertices.bin" 
face_file = "faces.bin"

convert_bin_to_obj(vert_file, face_file, "person_baseline.obj")