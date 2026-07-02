import json
import numpy as np

backup_path = "/home/timnogga/bachelorthesis/biwi_kinect_head_pose/RIFTCast/data/bonn_tele_bench_001/calibration_dome_backup.json"
with open(backup_path, 'r') as f:
    data = json.load(f)
V_arr = np.array(data["cameras"][0]["extrinsics"]["view_matrix"])

print("Array:", V_arr)
print("Reshaped row-major:\n", V_arr.reshape(4,4))
print("Reshaped col-major:\n", V_arr.reshape(4,4, order='F'))
