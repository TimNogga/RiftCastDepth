import os
import json
import torch
import numpy as np

def get_camera_carve_mask(vertices, vert_normals, camera_name, depth_path, camera_data, angle_threshold=-0.4):
    """Evaluates a single camera and returns a boolean mask of vertices that should be carved."""
    # Load depth map
    depth_tensor = torch.load(depth_path, map_location='cpu')
    depth_map = depth_tensor.reshape(3, 1080, 1920)[0].numpy() * 4.0 
    
    # Extract camera parameters
    cam_info = next(cam for cam in camera_data["cameras"] if cam["name"] == camera_name)
    projection_matrix = np.array(cam_info["view_projection"]).reshape(4, 4).T
    camera_pos = np.array(cam_info["extrinsics"]["position"])

    # Vector from camera to vertex
    rays = vertices - camera_pos
    rays /= np.linalg.norm(rays, axis=1, keepdims=True) + 1e-8
    facing_dot = np.sum(vert_normals * rays, axis=1)

    # Transform to clip space
    homogeneous_vertices = np.hstack((vertices, np.ones((len(vertices), 1))))
    clip_space = (projection_matrix @ homogeneous_vertices.T).T 
    depth_w = clip_space[:, 3]
    
    # Convert to normalized device coordinates and then to pixels
    ndc_x = clip_space[:, 0] / depth_w 
    ndc_y = clip_space[:, 1] / depth_w 
    pixel_x = ((ndc_x + 1.0) * 960).astype(int) 
    pixel_y = ((1.0 - ndc_y) * 540).astype(int) 
    
    # Identify points valid for this specific camera's view
    valid_mask = (pixel_x >= 0) & (pixel_x < 1920) & (pixel_y >= 0) & (pixel_y < 1080) & (depth_w > 0.1) 
    target_depths = depth_map[pixel_y[valid_mask], pixel_x[valid_mask]] 
    invalid_depth_mask = (target_depths < 0.1) | (target_depths > 3.9) 

    # Apply the normal shield
    is_facing = facing_dot[valid_mask] < angle_threshold
    
    # Identify carved vertices
    carved_mask_subset = (~invalid_depth_mask) & is_facing & ((depth_w[valid_mask] - 0.15) < (target_depths - 0.02))
    
    # Map back to full vertex array
    should_remove = np.zeros(len(vertices), dtype=bool)
    should_remove[valid_mask] = carved_mask_subset

    # --- TELEMETRY PRINT BLOCK ---
    print(f"\n[PYTHON BASELINE - {camera_name}]")
    print(f"  -> Camera Pos: {camera_pos}")
    print(f"  -> P Matrix Row 0: {projection_matrix[0]}")
    print(f"  -> Vertex [0] World Pos: {vertices[0]}")
    print(f"  -> Vertex [0] Pixel (X,Y): {pixel_x[0]}, {pixel_y[0]}")
    print(f"  -> Vertex [0] depth_w: {depth_w[0]}")
    try:
        print(f"  -> Vertex [0] target_depth: {depth_map[pixel_y[0], pixel_x[0]]}")
    except IndexError:
        print(f"  -> Vertex [0] target_depth: OUT OF BOUNDS")
    print(f"  -> Vertex [0] facing_dot: {facing_dot[0]}")
    # -----------------------------
    
    return should_remove

def export_multi_cam_mesh(vertices_path, faces_path, cam1_name, cam1_depth, cam2_name, cam2_depth, json_path, output_dir="outputs/multi_cam"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load geometry
    vertices = np.fromfile(vertices_path, dtype=np.float32).reshape(-1, 3) 
    faces = np.fromfile(faces_path, dtype=np.int64).reshape(-1, 3) 
    
    with open(json_path, 'r') as file:
        camera_data = json.load(file)

    # Calculate normals ONCE for the entire mesh
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    vert_normals = np.zeros_like(vertices)
    np.add.at(vert_normals, faces[:, 0], face_normals)
    np.add.at(vert_normals, faces[:, 1], face_normals)
    np.add.at(vert_normals, faces[:, 2], face_normals)
    vert_normals /= np.linalg.norm(vert_normals, axis=1, keepdims=True) + 1e-8

    angle_threshold = -0.4

    # Process Camera 1
    print(f"Processing {cam1_name}...")
    carve_mask_1 = get_camera_carve_mask(vertices, vert_normals, cam1_name, cam1_depth, camera_data, angle_threshold)
    
    # Process Camera 2
    print(f"Processing {cam2_name}...")
    carve_mask_2 = get_camera_carve_mask(vertices, vert_normals, cam2_name, cam2_depth, camera_data, angle_threshold)

    # --- COMBINE MASKS & ANALYZE ---
    total_carve_mask = carve_mask_1 | carve_mask_2
    keep_mask = ~total_carve_mask

    cam1_only = carve_mask_1 & ~carve_mask_2
    cam2_only = carve_mask_2 & ~carve_mask_1
    both_cams = carve_mask_1 & carve_mask_2

    print(f"\n--- Carving Stats ---")
    print(f"Total vertices carved: {total_carve_mask.sum()}")
    print(f"Carved by {cam1_name} ONLY: {cam1_only.sum()}")
    print(f"Carved by {cam2_name} ONLY: {cam2_only.sum()}  <-- (This is Camera 2's unique effect!)")
    print(f"Carved by BOTH: {both_cams.sum()}")

    # --- 1. Save Diagnostic PLY ---
    colors = np.full((len(vertices), 3), [128, 128, 128], dtype=np.uint8)
    
    # Green: Carved by Cam 1
    colors[cam1_only] = [0, 255, 0]
    # Yellow: Carved by Both
    colors[both_cams] = [255, 255, 0]
    # Magenta: Carved by Cam 2 exclusively (Highlights the added value of Cam 2)
    colors[cam2_only] = [255, 0, 255]
    
    ply_path = os.path.join(output_dir, f"Diagnostic_DualCam.ply")
    with open(ply_path, 'w') as file:
        file.write(f"ply\nformat ascii 1.0\nelement vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face {len(faces)}\nproperty list uchar int vertex_index\nend_header\n")
        np.savetxt(file, np.hstack((vertices, colors)), fmt="%.6f %.6f %.6f %d %d %d")
        np.savetxt(file, np.hstack((np.full((len(faces), 1), 3), faces)), fmt="%d %d %d %d")

    # --- 2. Save Clean OBJ ---
    new_vertices = vertices[keep_mask]
    old_to_new = np.full(len(vertices), -1, dtype=np.int64)
    old_to_new[keep_mask] = np.arange(len(new_vertices))
    
    # Only keep faces where all three vertices were kept
    new_faces = old_to_new[faces]
    valid_faces = new_faces[np.all(new_faces != -1, axis=1)]

    obj_path = os.path.join(output_dir, f"Clean_Mesh_DualCam.obj")
    with open(obj_path, 'w') as f:
        for v in new_vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in valid_faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    print(f"\nResults saved to {output_dir}")

# --- EXECUTION ---
base_path = "/home/timnogga/bachelorthesis/biwi_kinect_head_pose/RIFTCast"

export_multi_cam_mesh(
    vertices_path=f"{base_path}/export_depth_eimer/frame_00000/vertices.bin", 
    faces_path=f"{base_path}/export_depth_eimer/frame_00000/faces.bin", 
    
    cam1_name="D003L",
    cam1_depth=f"{base_path}/data/2026_03_20_orbbec_002_standard/frame_00000/rgb/D003L.pth", 
    
    cam2_name="D005Z",
    cam2_depth=f"{base_path}/data/2026_03_20_orbbec_002_standard/frame_00000/rgb/D005Z.pth", 
    
    json_path=f"{base_path}/export_depth_eimer/camera_debug.json"
)