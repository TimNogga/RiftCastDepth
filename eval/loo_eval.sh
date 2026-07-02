#!/usr/bin/env bash
# loo_eval.sh
#
# Leave-one-out PSNR/SSIM/LPIPS evaluation across ALL color cameras.
#
# For each evaluable camera C:
#   - train calib = every camera EXCEPT C
#   - test  calib = only C
#   - reconstruct geometry from the train cams, render the held-out view C,
#     compare it to the real captured image (GT) for C.
# Metrics are then averaged over all held-out cameras (true LOO cross-val).
#
# Configs evaluated:
#   no_depth : configs/real_01_no_depth.json            (visual hull baseline)
#   no_edge  : configs/real_03_tsdf_no_edge_gradient.json (TSDF, no edge protection)
#   edge     : configs/real_02_tsdf_edge_gradient.json    (TSDF, edge protection)
#
# Excluded from the LOO camera set:
#   D003L, D005Z              -> depth cameras (no RGB GT)
#   C0024, C0030, C1001       -> known bad masks (binary disables them anyway)
#
# Outputs -> output/loo_eval/
#   _splits/<CAM>/              per-camera train/test calibrations
#   <config>/<CAM>/             per-camera reconstruction + held-out render
#   <config>/all_test/rgb_<id>.png   collected held-out renders
#   results_<config>.json      per-camera + mean metrics
#   summary.txt

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

EXPORTER=(xvfb-run -a -s "-screen 0 1280x720x24" env __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./bin/RIFTCast_exporter)
DATASET="data/2026_03_20_orbbec_002_standard"
CALIB="$DATASET/calibration_dome.json"
GT_DIR="$DATASET/frame_00000/rgb"
MASK_DIR="$DATASET/frame_00000/mask"
OUT="output/loo_eval"
SPLITS="$OUT/_splits"
PY=.venv/bin/python3

# camera name -> integer id (id = digits of the name; matches rgb_<id>.png)
declare -A CAM_ID=(
  [C0000]=0    [C0001]=1    [C0004]=4    [C0005]=5    [C0006]=6    [C0007]=7
  [C0008]=8    [C0009]=9    [C0010]=10   [C0012]=12   [C0013]=13   [C0014]=14
  [C0016]=16   [C0018]=18   [C0019]=19   [C0020]=20   [C0021]=21   [C0022]=22
  [C0025]=25   [C0026]=26   [C0028]=28   [C0029]=29   [C0031]=31   [C0034]=34
  [C0037]=37   [C0038]=38   [C0039]=39   [C1000]=1000 [C1002]=1002 [C1004]=1004
  [C1005]=1005
)
EVAL_CAMS=(C0000 C0001 C0004 C0005 C0006 C0007 C0008 C0009 C0010 C0012 C0013 \
           C0014 C0016 C0018 C0019 C0020 C0021 C0022 C0025 C0026 C0028 C0029 \
           C0031 C0034 C0037 C0038 C0039 C1000 C1002 C1004 C1005)

declare -A CONFIGS=(
  [no_depth]="configs/real_01_no_depth.json"
  [no_edge]="configs/real_03_tsdf_no_edge_gradient.json"
  [edge]="configs/real_02_tsdf_edge_gradient.json"
)
CONFIG_ORDER=(no_depth no_edge edge)

mkdir -p "$OUT"

echo "=== Step 1: build per-camera train/test splits ($(date +%H:%M:%S)) ==="
for cam in "${EVAL_CAMS[@]}"; do
    if [[ ! -f "$SPLITS/$cam/calib_test.json" ]]; then
        $PY dataset/split_calibration.py --calib "$CALIB" --test "$cam" \
            --out "$SPLITS/$cam" >/dev/null
    fi
done
echo "Built splits for ${#EVAL_CAMS[@]} cameras."

echo ""
echo "=== Step 2: leave-one-out reconstructions ($(date +%H:%M:%S)) ==="
for config in "${CONFIG_ORDER[@]}"; do
    cfg_path="${CONFIGS[$config]}"
    collect="$OUT/$config/all_test"
    mkdir -p "$collect"
    echo ""
    echo "--- config: $config ($cfg_path) ---"
    n=0
    for cam in "${EVAL_CAMS[@]}"; do
        n=$((n+1))
        id="${CAM_ID[$cam]}"
        run_dir="$OUT/$config/$cam"
        dst="$collect/rgb_${id}.png"
        if [[ -f "$dst" ]]; then
            echo "  [$n/${#EVAL_CAMS[@]}] $cam -> cached"
            continue
        fi
        echo -n "  [$n/${#EVAL_CAMS[@]}] $cam (id=$id) ... "
        t0=$SECONDS
        if "${EXPORTER[@]}" \
                --dataset     "$cfg_path" \
                --output      "$run_dir" \
                --calibration "$REPO_ROOT/$SPLITS/$cam/calib_train.json" \
                --test_file   "$REPO_ROOT/$SPLITS/$cam/calib_test.json" \
                --test_only   true \
                --inpaint     false \
                --frame       0 > "$run_dir.log" 2>&1; then
            src="$run_dir/frame_00000/test/rgb_${id}.png"
            if [[ -f "$src" ]]; then
                cp "$src" "$dst"
                echo "ok ($((SECONDS-t0))s)"
            else
                echo "FAILED (no render; see $run_dir.log)"
            fi
        else
            echo "FAILED (exit; see $run_dir.log)"
        fi
    done
done

echo ""
echo "=== Step 3: compute metrics per config ($(date +%H:%M:%S)) ==="
CAM_NAMES=("${EVAL_CAMS[@]}")
CAM_IDS=()
for cam in "${EVAL_CAMS[@]}"; do CAM_IDS+=("${CAM_ID[$cam]}"); done

for config in "${CONFIG_ORDER[@]}"; do
    echo ""
    echo "--- metrics: $config ---"
    $PY eval/compute_metrics.py \
        --pred  "$OUT/$config/all_test" \
        --gt    "$GT_DIR" \
        --mask  "$MASK_DIR" \
        --cams  "${CAM_IDS[@]}" \
        --cam-names "${CAM_NAMES[@]}" \
        --label "$config (LOO mean over ${#EVAL_CAMS[@]} cams)" \
        --out   "$OUT/results_${config}.json"
done

echo ""
echo "=== Summary ($(date +%H:%M:%S)) ===" | tee "$OUT/summary.txt"
$PY - "${CONFIG_ORDER[@]}" <<'EOF' | tee -a "$OUT/summary.txt"
import json, sys
from pathlib import Path
out = Path("output/loo_eval")
configs = sys.argv[1:]
hdr = f"{'Config':<10} {'N':>3} {'PSNR':>7} {'SSIM':>8} {'LPIPS':>8}"
print(hdr); print("-" * len(hdr))
for c in configs:
    p = out / f"results_{c}.json"
    if not p.exists():
        print(f"{c:<10}  (no results)"); continue
    r = json.load(open(p)); m = r["mean"]
    lp = f"{m['lpips']:.4f}" if m["lpips"] is not None else "  n/a "
    print(f"{c:<10} {len(r['cameras']):>3} {m['psnr']:>7.2f} {m['ssim']:>8.4f} {lp:>8}")
EOF

echo ""
echo "Done. Results in $OUT/"
