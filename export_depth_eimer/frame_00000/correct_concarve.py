import os
import json
import torch
import numpy as np

def export_final_mesh(vertices_path, faces_path, depth_path, json_path, output_dir="outputs/works"):
    os.makedirs(output_dir, exist_ok=True)
    
    vertices = np.fromfile(vertices_path, dtype=np.float32).reshape(-1, 3) # reshapes the binary data into XYZ
    faces = np.fromfile(faces_path, dtype=np.int64).reshape(-1, 3) # reshapes the binary data into triangle indices
    
    depth_tensor = torch.load(depth_path, map_location='cpu') # loads the depth tensor
    depth_map = depth_tensor.reshape(3, 1080, 1920)[0].numpy() * 4.0 # gets depth map and applys scale cause it normalized 
    
    with open(json_path, 'r') as file:
        camera_data = json.load(file)
        
    d003l_camera = next(cam for cam in camera_data["cameras"] if cam["name"] == "D003L")
    projection_matrix = np.array(d003l_camera["view_projection"]).reshape(4, 4).T  # we have it in column-major order, so we transpose it to row-major for numpy
    camera_pos = np.array(d003l_camera["extrinsics"]["position"])

    # Calculate normals to identify which way vertices are facing
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    vert_normals = np.zeros_like(vertices)
    np.add.at(vert_normals, faces[:, 0], face_normals)
    np.add.at(vert_normals, faces[:, 1], face_normals)
    np.add.at(vert_normals, faces[:, 2], face_normals)
    vert_normals /= np.linalg.norm(vert_normals, axis=1, keepdims=True) + 1e-8

    # Vector from camera to vertex
    rays = vertices - camera_pos
    rays /= np.linalg.norm(rays, axis=1, keepdims=True) + 1e-8
    facing_dot = np.sum(vert_normals * rays, axis=1)

    homogeneous_vertices = np.hstack((vertices, np.ones((len(vertices), 1)))) # add a 1 to get xyz1 so we can multiply with the projection matrix
    clip_space = (projection_matrix @ homogeneous_vertices.T).T # transform to clip space
    depth_w = clip_space[:, 3]
    
    ndc_x = clip_space[:, 0] / depth_w #convert to normalized device coordinates 
    ndc_y = clip_space[:, 1] / depth_w 
    
    pixel_x = ((ndc_x + 1.0) * 960).astype(int) # convert to [0,2] range and then to pixel coordinates (1920/2 = 960)
    pixel_y = ((1.0 - ndc_y) * 540).astype(int) #same for this
    
    colors = np.full((len(vertices), 3), [128, 128, 128], dtype=np.uint8)
    
    valid_mask = (pixel_x >= 0) & (pixel_x < 1920) & (pixel_y >= 0) & (pixel_y < 1080) & (depth_w > 0.1) #we only care about points visible to the camera
    target_depths = depth_map[pixel_y[valid_mask], pixel_x[valid_mask]] # get the corresponding depth values from the depth map for the valid points 
    invalid_depth_mask = (target_depths < 0.1) | (target_depths > 3.9) 

    # Lock to requested -0.4 angle threshold
    angle_threshold = -0.4
    is_facing = facing_dot[valid_mask] < angle_threshold
    
    # Identify vertices to be carved (deleted)
    carved_mask_subset = (~invalid_depth_mask) & is_facing & ((depth_w[valid_mask] - 0.15) < (target_depths - 0.02))
    
    # Create a full-length mask for all vertices
    should_remove = np.zeros(len(vertices), dtype=bool)
    should_remove[valid_mask] = carved_mask_subset
    keep_mask = ~should_remove

    # --- 1. Save Diagnostic PLY ---
    valid_colors = np.full((valid_mask.sum(), 3), [0, 0, 255], dtype=np.uint8)
    valid_colors[invalid_depth_mask] = [255, 0, 0]
    valid_colors[carved_mask_subset] = [0, 255, 0]
    colors[valid_mask] = valid_colors
    
    ply_path = os.path.join(output_dir, f"Diagnostic_Angle_{angle_threshold}.ply")
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

    obj_path = os.path.join(output_dir, f"Clean_Mesh_Angle_{angle_threshold}.obj")
    with open(obj_path, 'w') as f:
        for v in new_vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in valid_faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    print(f"Results saved to {output_dir}")

base_path = "/home/timnogga/bachelorthesis/biwi_kinect_head_pose/RIFTCast"

export_final_mesh(
    f"{base_path}/export_depth_eimer/frame_00000/vertices.bin", 
    f"{base_path}/export_depth_eimer/frame_00000/faces.bin", 
    f"{base_path}/data/2026_03_20_orbbec_002_standard/frame_00000/rgb/D003L.pth", 
    f"{base_path}/export_depth_eimer/camera_debug.json"
)