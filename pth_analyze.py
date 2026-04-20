import argparse
from pathlib import Path

import torch

def find_pth_max_min(file_path):
    """
    Loads a .pth file and finds the global maximum and minimum 
    values across all its tensors.
    """
    try:
        # Load the file, mapping to CPU to avoid CUDA device errors
        data = torch.load(file_path, map_location='cpu', weights_only=False)
        
        global_max = float('-inf')
        global_min = float('inf')
        found_tensor = False

        # Case 1: The file is a dictionary (e.g., a state_dict)
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, torch.Tensor) and value.numel() > 0:
                    # Ignore non-numeric tensors like booleans
                    if value.dtype != torch.bool:
                        global_max = max(global_max, value.max().item())
                        global_min = min(global_min, value.min().item())
                        found_tensor = True
                        
        # Case 2: The file is a single PyTorch tensor
        elif isinstance(data, torch.Tensor) and data.numel() > 0:
             if data.dtype != torch.bool:
                global_max = data.max().item()
                global_min = data.min().item()
                found_tensor = True
                
        else:
            print("Unsupported format: The .pth file does not contain a standard dictionary or tensor.")
            return None, None

        if not found_tensor:
            print("No valid numerical tensors were found in the file.")
            return None, None

        return global_max, global_min

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None, None

def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect min/max value in a .pth tensor file")
    parser.add_argument(
        "file_path",
        nargs="?",
        default="data/2026_03_20_orbbec_002_standard/frame_00000/rgb/D003L.pth",
        help="Path to .pth file (default: data/2026_03_20_orbbec_002_standard/frame_00000/rgb/D003L.pth)",
    )
    args = parser.parse_args()

    file_path = Path(args.file_path)
    max_val, min_val = find_pth_max_min(str(file_path))

    if max_val is not None and min_val is not None:
        print(f"File: {file_path}")
        print(f"Global Maximum: {max_val:.6f}")
        print(f"Global Minimum: {min_val:.6f}")


if __name__ == "__main__":
    main()