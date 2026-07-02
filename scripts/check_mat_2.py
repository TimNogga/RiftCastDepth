import json

path1 = "/home/timnogga/bachelorthesis/biwi_kinect_head_pose/RIFTCast/data/bonn_tele_bench_001/calibration_dome_backup.json"
path2 = "/home/timnogga/bachelorthesis/biwi_kinect_head_pose/RIFTCast/data/bonn_tele_bench_001/calibration_dome.json"

with open(path1) as f:
    d1 = json.load(f)
with open(path2) as f:
    d2 = json.load(f)

for i in range(1):
    c1 = d1["cameras"][i]
    c2 = d2["cameras"][i]
    
    print("Old:", type(c1["intrinsics"]["resolution"][0]), c1["intrinsics"]["resolution"])
    print("New:", type(c2["intrinsics"]["resolution"][0]), c2["intrinsics"]["resolution"])
