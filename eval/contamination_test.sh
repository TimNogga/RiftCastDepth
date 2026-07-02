#!/usr/bin/env bash
# contamination_test.sh
#
# Empirically shows that the main-loop renders (--exclude_nearest) are silhouette-
# contaminated and produce inflated PSNR/SSIM/LPIPS compared to the clean hold-out
# via --test_file.
#
# Test cameras: C0005 (id=5), C0026 (id=26), C0037 (id=37)
# Config used:  real_01_no_depth (VH baseline, no depth — clean comparison)
# Frame:        0 only (fast)
#
# Usage:
#   bash eval/contamination_test.sh
#
# Outputs:
#   output/contamination_test/
#       run_a/   <- main-loop renders (contaminated, --exclude_nearest)
#       run_b/   <- clean test-file renders
#       results_run_a.json
#       results_run_b.json
#       summary.txt

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

EXPORTER=(xvfb-run -a -e /dev/stdout -s "-screen 0 1280x720x24" env __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./bin/RIFTCast_exporter)
CONFIG="configs/real_01_no_depth.json"
DATASET="data/2026_03_20_orbbec_002_standard"
OUT="output/contamination_test"
SPLITS="$OUT/splits"
GT_DIR="$DATASET/frame_00000/rgb"
MASK_DIR="$DATASET/frame_00000/mask"
TEST_CAMS=(5 26 37)
TEST_NAMES=(C0005 C0026 C0037)

echo "=== Step 1: Build split calibrations ==="
.venv/bin/python3 dataset/split_calibration.py \
    --calib "$DATASET/calibration_dome.json" \
    --test  C0005 C0026 C0037 \
    --out   "$SPLITS"

echo ""
echo "=== Step 2: Run A — contaminated (all cams, --exclude_nearest true) ==="
"${EXPORTER[@]}" \
    --dataset "$CONFIG" \
    --output  "$OUT/run_a" \
    --exclude_nearest true \
    --inpaint false \
    --frame 0

echo ""
echo "=== Step 3: Run B — clean hold-out (--test_file, test cams excluded from geometry) ==="
# Override the calibration path so train cams don't include the test cameras,
# and pass test cams via --test_file
"${EXPORTER[@]}" \
    --dataset     "$CONFIG" \
    --output      "$OUT/run_b" \
    --calibration "$SPLITS/calib_train.json" \
    --test_file   "$SPLITS/calib_test.json" \
    --test_only   true \
    --inpaint     false \
    --frame       0

echo ""
echo "=== Step 4: Install lpips if missing ==="
.venv/bin/pip install lpips scikit-image -q

echo ""
echo "=== Step 5: Compute metrics for Run A (contaminated) ==="
.venv/bin/python3 eval/compute_metrics.py \
    --pred  "$OUT/run_a/frame_00000/rgb" \
    --gt    "$GT_DIR" \
    --mask  "$MASK_DIR" \
    --cams  "${TEST_CAMS[@]}" \
    --cam-names "${TEST_NAMES[@]}" \
    --label "run_a (contaminated, exclude_nearest)" \
    --out   "$OUT/results_run_a.json"

echo ""
echo "=== Step 6: Compute metrics for Run B (clean hold-out) ==="
.venv/bin/python3 eval/compute_metrics.py \
    --pred  "$OUT/run_b/frame_00000/test" \
    --gt    "$GT_DIR" \
    --mask  "$MASK_DIR" \
    --cams  "${TEST_CAMS[@]}" \
    --cam-names "${TEST_NAMES[@]}" \
    --label "run_b (clean hold-out)" \
    --out   "$OUT/results_run_b.json"

echo ""
echo "=== Summary ==="
.venv/bin/python3 - <<'EOF'
import json
from pathlib import Path

out = Path("output/contamination_test")
a = json.load(open(out / "results_run_a.json"))
b = json.load(open(out / "results_run_b.json"))

hdr = f"{'Config':<40} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7}"
sep = "-" * len(hdr)
print(hdr)
print(sep)
for r in (a, b):
    m = r["mean"]
    lp = f"{m['lpips']:.4f}" if m["lpips"] is not None else "  n/a "
    print(f"{r['label']:<40} {m['psnr']:>7.2f} {m['ssim']:>7.4f} {lp:>7}")

print()
da = a["mean"]; db = b["mean"]
print(f"Delta (A - B):  PSNR {da['psnr']-db['psnr']:+.2f} dB   "
      f"SSIM {da['ssim']-db['ssim']:+.4f}   "
      f"LPIPS {(da['lpips'] or 0)-(db['lpips'] or 0):+.4f}")
print()
print("Positive PSNR/SSIM delta => contaminated run inflates scores.")
print("Negative LPIPS delta     => contaminated run looks artificially better.")
EOF

tee "$OUT/summary.txt" <<'ENDSUM'
See results_run_a.json and results_run_b.json for per-camera breakdown.
ENDSUM

echo ""
echo "Done. Results in $OUT/"
