import json
import numpy as np

backup_path = "/home/timnogga/bachelorthesis/biwi_kinect_head_pose/RIFTCast/data/bonn_tele_bench_001/calibration_dome_backup.json"
output_path = "/home/timnogga/bachelorthesis/biwi_kinect_head_pose/RIFTCast/data/bonn_tele_bench_001/calibration_dome.json"

with open(backup_path, 'r') as f:
    data = json.load(f)

# Rotation to map X -> -Y, Y -> X (Roll by 90 deg clockwise looking along Z)
R_roll = np.array([
    [ 0, -1,  0,  0],
    [ 1,  0,  0,  0],
    [ 0,  0,  1,  0],
    [ 0,  0,  0,  1]
], dtype=float)

for cam in data.get("cameras", []):
    if "extrinsics" in cam and "view_matrix" in cam["extrinsics"]:
        # The view matrix in the file is stored in row-major or column-major?
        # The C++ code did: glm::transpose(glm::make_mat4(extr_vec.data()))
        # This implies the JSON array is row-major.
        V_old = np.array(cam["extrinsics"]["view_matrix"]).reshape(4, 4)
        
        # Apply the rotation
        V_new = R_roll @ V_old
        
        cam["extrinsics"]["view_matrix"] = V_new.flatten().tolist()
        
    if "intrinsics" in cam:
        H_old = 4608  # Old height
        fx_old = cam["intrinsics"]["camera_matrix"][0]
        fy_old = cam["intrinsics"]["camera_matrix"][4]
        cx_old = cam["intrinsics"]["camera_matrix"][2]
        cy_old = cam["intrinsics"]["camera_matrix"][5]
        
        fx_new = fy_old
        fy_new = fx_old
        cx_new = H_old - cy_old
        cy_new = cx_old
        
        cam["intrinsics"]["camera_matrix"][0] = fx_new
        cam["intrinsics"]["camera_matrix"][4] = fy_new
        cam["intrinsics"]["camera_matrix"][2] = cx_new
        cam["intrinsics"]["camera_matrix"][5] = cy_new
        
        cam["intrinsics"]["resolution"] = [4608, 5328]

with open(output_path, 'w') as f:
    json.dump(data, f, indent=2)

print("Applied 90-degree clockwise image rotation mapping to extrinsics and intrinsics.")
