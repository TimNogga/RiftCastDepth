import json
import numpy as np
import copy

input_path = "/home/timnogga/bachelorthesis/biwi_kinect_head_pose/RIFTCast/data/bonn_tele_bench_001/calibration_train.json"
output_path = "/home/timnogga/bachelorthesis/biwi_kinect_head_pose/RIFTCast/data/bonn_tele_bench_001/calibration_dome.json"

with open(input_path, 'r') as f:
    data = json.load(f)

R_roll = np.array([
    [ 0, -1,  0,  0],
    [ 1,  0,  0,  0],
    [ 0,  0,  1,  0],
    [ 0,  0,  0,  1]
], dtype=float)

scale = 2.0
H_old = 2304

for cam in data.get("cameras", []):
    if "extrinsics" in cam and "view_matrix" in cam["extrinsics"]:
        V_old = np.array(cam["extrinsics"]["view_matrix"]).reshape(4, 4)
        V_new = R_roll @ V_old
        cam["extrinsics"]["view_matrix"] = V_new.flatten().tolist()
        
    if "intrinsics" in cam:
        fx_old = cam["intrinsics"]["camera_matrix"][0]
        fy_old = cam["intrinsics"]["camera_matrix"][4]
        cx_old = cam["intrinsics"]["camera_matrix"][2]
        cy_old = cam["intrinsics"]["camera_matrix"][5]
        
        # 1x scale rotation
        fx_1x = fy_old
        fy_1x = fx_old
        cx_1x = H_old - cy_old
        cy_1x = cx_old
        
        # 2x scale
        cam["intrinsics"]["camera_matrix"][0] = fx_1x * scale
        cam["intrinsics"]["camera_matrix"][4] = fy_1x * scale
        cam["intrinsics"]["camera_matrix"][2] = cx_1x * scale
        cam["intrinsics"]["camera_matrix"][5] = cy_1x * scale
        
        cam["intrinsics"]["resolution"] = [4608, 5328]

with open(output_path, 'w') as f:
    json.dump(data, f, indent=2)

print("Applied clean rotation + scale from calibration_train.json")
