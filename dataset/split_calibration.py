#!/usr/bin/env python3
"""
Split a calibration_dome.json into a training and a test calibration file.

Usage:
    python3 dataset/split_calibration.py \
        --calib  data/2026_03_20_orbbec_002_standard/calibration_dome.json \
        --test   C0005 C0026 C0037 \
        --out    configs/splits/

Produces:
    configs/splits/calib_train.json   (all cameras EXCEPT test cameras)
    configs/splits/calib_test.json    (ONLY the test cameras)
"""

import argparse
import json
from pathlib import Path


def split_calib(calib_path: Path, test_ids: list[str], out_dir: Path) -> None:
    with open(calib_path) as f:
        full = json.load(f)

    test_set = set(test_ids)

    def filter_list(lst: list, key: str = "camera_id") -> tuple[list, list]:
        train = [e for e in lst if e.get(key, "") not in test_set]
        test  = [e for e in lst if e.get(key, "") in test_set]
        return train, test

    train_cams, test_cams = filter_list(full["cameras"])
    train_proc, test_proc = filter_list(full.get("processing_applied", []))
    train_raw,  test_raw  = filter_list(full.get("raw_calibration", []))

    def build(cams, proc, raw):
        d = {
            "meta": full["meta"],
            "cameras": cams,
        }
        if proc:
            d["processing_applied"] = proc
        if raw:
            d["raw_calibration"] = raw
        return d

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "calib_train.json"
    test_path  = out_dir / "calib_test.json"

    with open(train_path, "w") as f:
        json.dump(build(train_cams, train_proc, train_raw), f, indent=2)
    with open(test_path, "w") as f:
        json.dump(build(test_cams, test_proc, test_raw), f, indent=2)

    print(f"Train calibration: {len(train_cams)} cameras -> {train_path}")
    print(f"Test  calibration: {len(test_cams)}  cameras -> {test_path}")
    print(f"Test cameras: {[c['camera_id'] for c in test_cams]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--test",  nargs="+", required=True, metavar="CAM_ID")
    parser.add_argument("--out",   type=Path, default=Path("configs/splits"))
    args = parser.parse_args()
    split_calib(args.calib, args.test, args.out)


if __name__ == "__main__":
    main()
