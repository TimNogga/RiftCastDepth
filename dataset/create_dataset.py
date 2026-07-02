import os
import json
import numpy as np
import cv2
import torch
import shutil

# Setup Directories
root_dir = "/mnt/data/synthetic_dataset_bowl"
os.makedirs(root_dir, exist_ok=True)
frame_dir = os.path.join(root_dir, "frame_00000")
os.makedirs(os.path.join(frame_dir, "rgb"), exist_ok=True)
os.makedirs(os.path.join(frame_dir, "mask"), exist_ok=True)
os.makedirs(os.path.join(frame_dir, "depth"), exist_ok=True)

# 1. Generate dataset.json
dataset_json = {
    "type": "VCI",
    "version": "2.0",
    "name": "Concave_Bowl_Test",
    "dataset": {
        "frame_count": 1,
        "start_frame": 0,
        "path": root_dir,
        "camera_path": os.path.join(root_dir, "calibration_dome.json"),
        "to_world": [1.0, 0.0, 0.0, 0.0, 
                     0.0, 1.0, 0.0, 0.0, 
                     0.0, 0.0, 1.0, 0.0, 
                     0.0, 0.0, 0.0, 1.0], 
        "flip_images": False,
        "flip_masks": False
    },
    "depth": {
        "has_depth": True,
        "scale": 1000.0,
        "extension": ".pth"
    }
}
with open(os.path.join(root_dir, "dataset.json"), 'w') as f:
    json.dump(dataset_json, f, indent=4)

# Camera Parameters
W, H = 1920, 1080
fx, fy = 1000.0, 1000.0
cx, cy = 960.0, 540.0
camera_matrix = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]

# Shape Definition: A Hemispherical Bowl
bowl_radius = 0.5
bowl_thickness = 0.05
bowl_center = np.array([0.0, 0.0, 0.0])

def ray_sphere_intersect(rays_o, rays_d, center, radius):
    oc = rays_o - center
    b = np.einsum('hwd, hwd -> hw', rays_d, oc)
    c = np.einsum('hwd, hwd -> hw', oc, oc) - radius**2
    discriminant = b**2 - c
    
    # Initialize distances to infinity
    t1 = np.full_like(discriminant, np.inf)
    t2 = np.full_like(discriminant, np.inf)
    
    hit = discriminant > 0
    t1[hit] = -b[hit] - np.sqrt(discriminant[hit])
    t2[hit] = -b[hit] + np.sqrt(discriminant[hit])
    
    # We only care about positive distances (in front of camera)
    t1 = np.where(t1 > 0, t1, np.inf)
    t2 = np.where(t2 > 0, t2, np.inf)
    
    return hit, t1, t2

# Prepare Ray Grid in Camera Space
u, v = np.meshgrid(np.arange(W), np.arange(H))
x_c = (u - cx) / fx
y_c = (v - cy) / fy
z_c = np.ones_like(x_c)
ray_c = np.stack([x_c, y_c, z_c], axis=-1)
norm_c = np.linalg.norm(ray_c, axis=-1, keepdims=True)
ray_c_dir = ray_c / norm_c

cameras = []
camera_distance = 3.0

# 4 Cameras positioned in a dome structure
# Cam 1: Look down from top (Can see concavity)
# Cam 2,3,4: Side views at slight elevation (See mostly profile)
angles = [
    (0, 80),   # Top camera (looking almost straight down)
    (0, 20),   # Front-side
    (120, 20), # Right-side
    (240, 20)  # Left-side
]

for i, (azimuth, elevation) in enumerate(angles):
    az = np.radians(azimuth)
    el = np.radians(elevation)
    
    # Spherical to Cartesian (Y-up)
    cam_pos = np.array([
        camera_distance * np.cos(el) * np.sin(az),
        camera_distance * np.sin(el),
        camera_distance * np.cos(el) * np.cos(az)
    ])
    
    # OpenCV Camera Coordinate System
    Z_c = -cam_pos / np.linalg.norm(cam_pos)
    Y_world_up = np.array([0, 1, 0])
    X_c = np.cross(Y_world_up, Z_c)
    if np.linalg.norm(X_c) < 1e-6: # Handle looking straight down
        X_c = np.array([1, 0, 0])
    else:
        X_c = X_c / np.linalg.norm(X_c)
    Y_c = np.cross(Z_c, X_c)
    
    # Y down for OpenCV
    Y_c = -Y_c
    X_c = -X_c
    
    R = np.vstack([X_c, Y_c, Z_c])
    t = -R @ cam_pos
    view_matrix = np.eye(4)
    view_matrix[:3, :3] = R
    view_matrix[:3, 3] = t
    
    cam_id = f"cam{i+1:02d}"
    depth_id = f"D{i+1:02d}"
    
    cameras.append({
        "camera_id": cam_id, "camera_type": "rgb",
        "intrinsics": {"resolution": [W, H], "camera_matrix": camera_matrix},
        "extrinsics": {"view_matrix": view_matrix.flatten().tolist()}
    })
    cameras.append({
        "camera_id": depth_id, "camera_type": "depth",
        "intrinsics": {"resolution": [W, H], "camera_matrix": camera_matrix},
        "extrinsics": {"view_matrix": view_matrix.flatten().tolist()}
    })
    
    R_wc = R.T
    ray_w = np.einsum('hwc, cd -> hwd', ray_c_dir, R_wc)
    
    # --- Intersect with the Bowl ---
    # The bowl is a hemisphere where Y < 0
    # Outer sphere intersection
    hit_outer, t1_out, t2_out = ray_sphere_intersect(cam_pos, ray_w, bowl_center, bowl_radius)
    # Inner sphere intersection
    hit_inner, t1_in, t2_in = ray_sphere_intersect(cam_pos, ray_w, bowl_center, bowl_radius - bowl_thickness)
    
    # Find exact hit points
    p1_out = cam_pos + ray_w * t1_out[..., np.newaxis]
    p2_out = cam_pos + ray_w * t2_out[..., np.newaxis]
    p1_in = cam_pos + ray_w * t1_in[..., np.newaxis]
    p2_in = cam_pos + ray_w * t2_in[..., np.newaxis]
    
    # Validate hits: Must be lower hemisphere (Y <= 0) OR on the rim
    valid_p1_out = hit_outer & (p1_out[..., 1] <= 0)
    valid_p2_out = hit_outer & (p2_out[..., 1] <= 0)
    valid_p1_in = hit_inner & (p1_in[..., 1] <= 0)
    
    # The logic:
    # 1. If we hit the outer shell first, that's our depth.
    # 2. If we hit the flat top "rim" (Y=0 between inner and outer radius)
    # 3. If we go through the top opening and hit the inner shell.
    
    t_final = np.full((H, W), np.inf)
    
    # Check top rim (Plane intersection at Y=0)
    # ray_w_y * t + cam_pos_y = 0  => t = -cam_pos_y / ray_w_y
    t_plane = -cam_pos[1] / (ray_w[..., 1] + 1e-6)
    p_plane = cam_pos + ray_w * t_plane[..., np.newaxis]
    r_sq = p_plane[..., 0]**2 + p_plane[..., 2]**2
    hit_rim = (t_plane > 0) & (r_sq <= bowl_radius**2) & (r_sq >= (bowl_radius - bowl_thickness)**2)
    t_final[hit_rim] = t_plane[hit_rim]
    
    # Check outer shell
    hit_outer_shell = valid_p1_out & (t1_out < t_final)
    t_final[hit_outer_shell] = t1_out[hit_outer_shell]
    
    # Check inner concavity (Ray passes through opening, hits inside back wall)
    ray_enters_top = (t_plane > 0) & (r_sq < (bowl_radius - bowl_thickness)**2)
    hit_inner_wall = ray_enters_top & valid_p2_out # Hits the back of the outer shell from inside
    # If it hits the inner shell, it's hitting the inner wall
    hit_inner_wall_true = ray_enters_top & valid_p1_in
    
    t_final[hit_inner_wall_true] = t1_in[hit_inner_wall_true]
    
    # Generate Mask
    mask = ((t_final < np.inf) * 255).astype(np.uint8)
    
    # Convert distance to orthogonal Depth
    cos_theta = np.einsum('hwd, d -> hw', ray_w, Z_c)
    depth_z = t_final * cos_theta
    depth_z[mask == 0] = 0.0
    
    rgb_img = np.zeros((H, W, 3), dtype=np.uint8)
    cv2.imwrite(os.path.join(frame_dir, "rgb", f"{cam_id}.jpg"), rgb_img)
    
    with open(os.path.join(frame_dir, "mask", f"mask_{cam_id}.bin"), 'wb') as f:
        f.write(mask.tobytes())
        
    depth_tensor = torch.from_numpy(depth_z * 1000.0).float()
    torch.save(depth_tensor, os.path.join(frame_dir, "depth", f"{depth_id}.pth"))

with open(os.path.join(root_dir, "calibration_dome.json"), 'w') as f:
    json.dump({"cameras": cameras}, f, indent=4)
    
shutil.make_archive('/mnt/data/synthetic_dataset_bowl', 'zip', '/mnt/data', 'synthetic_dataset_bowl')
print("Bowl Dataset created and zipped.")