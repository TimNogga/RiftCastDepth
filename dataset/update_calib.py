import json

path = "/home/timnogga/bachelorthesis/biwi_kinect_head_pose/RIFTCast/data/bonn_tele_bench_001/calibration_dome.json"
with open(path, 'r') as f:
    data = json.load(f)

for cam in data["cameras"]:
    if "intrinsics" in cam:
        cam["intrinsics"]["resolution"] = [4608, 5328]
        # We also need to scale the principal point and focal length by the resize factor
        # Original: W=2664, H=2304 (Note: resize was to 4608, 5328? Wait OpenCV: w*2=2664*2=5328, h*2=2304*2=4608)
        # But wait! I rotated it! 
        # width = 2304*2 = 4608
        # height = 2664*2 = 5328
        # We MUST rotate the intrinsics matrix!
        
        fx = cam["intrinsics"]["camera_matrix"][0]
        fy = cam["intrinsics"]["camera_matrix"][4]
        cx = cam["intrinsics"]["camera_matrix"][2]
        cy = cam["intrinsics"]["camera_matrix"][5]

        # 1. Rotate 90 degrees clockwise
        # new_cx = H - cy, new_cy = cx
        # new_fx = fy, new_fy = fx
        new_cx = 2304 - cy
        new_cy = cx
        new_fx = fy
        new_fy = fx

        # 2. Scale by 2
        new_cx *= 2
        new_cy *= 2
        new_fx *= 2
        new_fy *= 2

        cam["intrinsics"]["camera_matrix"][0] = new_fx
        cam["intrinsics"]["camera_matrix"][4] = new_fy
        cam["intrinsics"]["camera_matrix"][2] = new_cx
        cam["intrinsics"]["camera_matrix"][5] = new_cy

with open(path, 'w') as f:
    json.dump(data, f, indent=4)
print("Updated calibration_dome.json")
