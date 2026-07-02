import cv2
import glob
import os
import numpy as np
import shutil

tele_dir = "/home/timnogga/bachelorthesis/biwi_kinect_head_pose/RIFTCast/data/bonn_tele_bench_001"

print("1. Removing all frames except the first 5...")
frames = sorted(glob.glob(os.path.join(tele_dir, "frame_*")))
for frame_dir in frames[5:]:
    shutil.rmtree(frame_dir)

frames = sorted(glob.glob(os.path.join(tele_dir, "frame_*")))

print("2. Converting remaining frames...")
for frame_dir in frames:
    print(f"Processing {frame_dir}...")
    
    # Process RGB
    rgb_dir = os.path.join(frame_dir, "rgb")
    if os.path.exists(rgb_dir):
        for img_path in glob.glob(os.path.join(rgb_dir, "*.jpg")):
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                if w == 2664 and h == 2304: # Original size, process it
                    # Rotate 90 degrees clockwise
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    # Resize to Orbbec standard
                    img = cv2.resize(img, (4608, 5328), interpolation=cv2.INTER_LINEAR)
                    # Overwrite and create PNG
                    cv2.imwrite(img_path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    cv2.imwrite(img_path.replace('.jpg', '.png'), img)

    # Process Mask
    mask_dir = os.path.join(frame_dir, "mask")
    if os.path.exists(mask_dir):
        # Remove any old small .bin files
        for old_bin in glob.glob(os.path.join(mask_dir, "*.bin")):
            os.remove(old_bin)

        # Process the original JPG masks
        for mask_jpg in glob.glob(os.path.join(mask_dir, "mask_*.jpg")):
            basename = os.path.basename(mask_jpg)
            cid_str = basename.replace('mask_', '').replace('.jpg', '')
            try:
                cid = int(cid_str)
            except ValueError:
                continue
                
            img = cv2.imread(mask_jpg, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                h, w = img.shape[:2]
                if w == 2664 and h == 2304:
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    img = cv2.resize(img, (4608, 5328), interpolation=cv2.INTER_NEAREST)
                    
                # The correct name structure for masks: mask_C0000.bin
                bin_path = os.path.join(mask_dir, f"mask_C{cid:04d}.bin")
                img.tofile(bin_path)
                
            # Clean up the old original mask jpgs
            os.remove(mask_jpg)

print("Done. Dataset has been reduced to 5 frames and converted to match the Orbbec format.")
