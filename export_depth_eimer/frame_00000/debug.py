import os
import numpy as np
import torch
import matplotlib.pyplot as plt

def ultimate_projection_test(vertices_bin, depth_pth, output_dir="outputs/projection_ultimate"):
    print("1. Loading raw data...")
    verts = np.fromfile(vertices_bin, dtype=np.float32).reshape(-1, 3)
    
    depth_raw = torch.load(depth_pth, map_location='cpu')
    if depth_raw.numel() == 6220800:
        depth_map = depth_raw.reshape(3, 1080, 1920)[0, :, :].numpy() * 4.0
    else:
        depth_map = depth_raw.numpy() * 4.0
        
    print("2. Setting up RIFTCast Exact View-Projection Matrix...")
    # Extracted directly from D003L in your camera_debug.json
    P_flat = [
        0.021082766354084015,  2.0636472702026367, -0.0888671725988388, -0.08886539191007614,
        0.504758894443512,     0.1625124216079712,  0.8928878307342529,  0.8928699493408203,
       -1.0508840084075928,    0.11966755986213684, 0.44146695733070374, 0.4414581060409546,
        1.7733519077301025,   -2.7514522075653076,  0.06361725926399231, 0.08361577987670898
    ]
    
    # Reshape and transpose to match standard row-major math
    P = np.array(P_flat).reshape(4, 4).T 

    print("3. Executing Exact Matrix Math...")
    # Convert to homogeneous (add 1s)
    verts_h = np.hstack((verts, np.ones((len(verts), 1))))
    
    # Apply the master projection matrix
    clip = (P @ verts_h.T).T
    
    # Filter points completely behind the camera (where W <= 0)
    cw = clip[:, 3]
    front_mask = cw > 0.01  
    
    clip_f = clip[front_mask]
    cw_f = cw[front_mask]
    
    # Calculate Normalized Device Coordinates (NDC)
    ndc_x = clip_f[:, 0] / cw_f
    ndc_y = clip_f[:, 1] / cw_f
    
    # Scale to Image Dimensions (1920x1080)
    W, H = 1920, 1080
    px = (ndc_x + 1.0) * 0.5 * W
    
    # Two variations to catch image-coordinate flip
    py_standard = (ndc_y + 1.0) * 0.5 * H
    py_flipped  = (1.0 - ndc_y) * 0.5 * H
    
    print("4. Generating overlay images...")
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Image 1: Standard Y ---
    plt.figure(figsize=(16, 9))
    plt.imshow(depth_map, cmap='gray')
    plt.scatter(px, py_standard, s=0.1, c='red', alpha=0.5)
    plt.xlim(0, 1920)
    plt.ylim(1080, 0)
    plt.savefig(os.path.join(output_dir, "01_ultimate_standard_Y.png"), bbox_inches='tight', dpi=150)
    plt.close()

    # --- Image 2: Flipped Y ---
    plt.figure(figsize=(16, 9))
    plt.imshow(depth_map, cmap='gray')
    plt.scatter(px, py_flipped, s=0.1, c='red', alpha=0.5)
    plt.xlim(0, 1920)
    plt.ylim(1080, 0)
    plt.savefig(os.path.join(output_dir, "02_ultimate_flipped_Y.png"), bbox_inches='tight', dpi=150)
    plt.close()

    print(f"Saved to: {output_dir}/")

# --- Run the Script ---
base_dir = "/home/timnogga/bachelorthesis/biwi_kinect_head_pose/RIFTCast"
vert_file = f"{base_dir}/export_depth_eimer/frame_00000/vertices.bin" 
pth_file = f"{base_dir}/data/2026_03_20_orbbec_002_standard/frame_00000/rgb/D003L.pth"

ultimate_projection_test(vert_file, pth_file)