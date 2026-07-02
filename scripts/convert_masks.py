import argparse
from pathlib import Path

import cv2


def convert_dataset_masks(dataset_dir: Path) -> None:
    frame_dirs = sorted(dataset_dir.glob("frame_*"))

    if not frame_dirs:
        print(f"No frame directories found in {dataset_dir}")
        return

    print(f"Found {len(frame_dirs)} frames. Starting conversion...")

    for frame_dir in frame_dirs:
        input_dir = frame_dir / "rgb" / "mask"
        output_dir = frame_dir / "mask"
        png_files = sorted(input_dir.glob("mask_*.png"))

        if not png_files:
            continue

        output_dir.mkdir(parents=True, exist_ok=True)

        converted_count = 0
        for png_path in png_files:
            img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Failed to read {png_path}")
                continue

            out_path = output_dir / png_path.name.replace(".png", ".bin")
            with out_path.open("wb") as f:
                f.write(img.tobytes())

            converted_count += 1

        print(f"[{frame_dir.name}] Successfully converted {converted_count} masks.")

    print("\nBatch conversion complete! You are ready to export.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert frame_*/rgb/mask/mask_*.png files to frame_*/mask/*.bin")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("2026_02_03_orbbec_test"),
        help="Dataset root directory (default: 2026_02_03_orbbec_test)",
    )
    args = parser.parse_args()
    convert_dataset_masks(args.dataset_dir)


if __name__ == "__main__":
    main()