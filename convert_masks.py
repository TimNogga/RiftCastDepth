import os
import cv2
import glob

# Your main dataset directory
dataset_dir = "/home/timnogga/bachelorthesis/biwi_kinect_head_pose/RIFTCast/2026_02_03_orbbec_test/"

# Find all frame directories (e.g., frame_00000, frame_00001, etc.)
frame_dirs = sorted(glob.glob(os.path.join(dataset_dir, "frame_*")))

if not frame_dirs:
    print(f"No frame directories found in {dataset_dir}")
else:
    print(f"Found {len(frame_dirs)} frames. Starting conversion...")
    
    for frame_dir in frame_dirs:
        input_dir = os.path.join(frame_dir, "rgb", "mask")
        output_dir = os.path.join(frame_dir, "mask")
        
        # Find all PNG masks in this specific frame
        png_files = glob.glob(os.path.join(input_dir, "mask_*.png"))
        
        if not png_files:
            continue # Skip if no pngs found in this frame
            
        # Create the missing mask/ directory
        os.makedirs(output_dir, exist_ok=True)
        
        converted_count = 0
        for png_path in png_files:
            # Read the PNG as a grayscale 8-bit image
            img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"Warning: Failed to read {png_path}")
                continue
                
            # Swap .png for .bin
            filename = os.path.basename(png_path).replace(".png", ".bin")
            out_path = os.path.join(output_dir, filename)
            
            # Dump the raw bytes for the C++ importer
            with open(out_path, "wb") as f:
                f.write(img.tobytes())
            
            converted_count += 1
            
        print(f"[{os.path.basename(frame_dir)}] Successfully converted {converted_count} masks.")

print("\nBatch conversion complete! You are ready to export.")