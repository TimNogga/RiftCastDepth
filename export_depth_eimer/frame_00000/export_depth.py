import os
import torch
import matplotlib.pyplot as plt

def load_and_inspect_depth(pth_path, output_dir="outputs"):
    print(f"Loading {pth_path}...")
    
    try:
        depth_normalized = torch.load(pth_path, map_location=torch.device('cpu'))
        
        # Reshape the flat array
        if depth_normalized.numel() == 6220800:
            depth_reshaped = depth_normalized.reshape(1080, 1920, 3)
            # Grab just the first channel
            depth_normalized = depth_reshaped[:, :, 0]
            print("Successfully reshaped flat tensor to [1080, 1920].")
            
        # Denormalize (0 to 1 -> 0 to 4 meters)
        depth_meters = depth_normalized * 4.0
        
        print(f"Final Shape: {depth_meters.shape}")
        print(f"Min Depth: {depth_meters.min().item():.3f} meters")
        print(f"Max Depth: {depth_meters.max().item():.3f} meters")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "D003L_depth_map.png")
        
        # Visualize and save
        plt.figure(figsize=(10, 6))
        plt.title("D003L Depth Map (Meters)")
        plt.imshow(depth_meters.numpy(), cmap='turbo') 
        plt.colorbar(label='Meters')
        
        # Save to disk instead of trying to open a window
        plt.savefig(output_file, bbox_inches='tight')
        plt.close() # Free up memory
        
        print(f"Saved visualization to: {output_file}")
        
        return depth_meters

    except Exception as e:
        print(f"Failed to load the depth map: {e}")
        return None

# --- Run the Script ---
pth_file = "/home/timnogga/bachelorthesis/biwi_kinect_head_pose/RIFTCast/data/2026_03_20_orbbec_002_standard/frame_00000/rgb/D003L.pth"
depth_data = load_and_inspect_depth(pth_file)