#!/usr/bin/env bash
# edge_ablation_metrics.sh
#
# Real-data PSNR/SSIM/LPIPS comparison of TSDF depth fusion WITHOUT vs WITH the
# depth-edge protection.
#
#   no_edge : configs/real_03_tsdf_no_edge_gradient.json  (depth_fusion_mode=real_no_edge)
#   edge    : configs/real_02_tsdf_edge_gradient.json     (depth_fusion_mode=real)
#
# Test views are rendered for cams C0005 C0026 C0037 and compared to the real
# captured images. Both variants use the identical camera setup, so the
# edge-vs-no-edge delta is a fair comparison. (Note: the exporter's clean
# hold-out path --calibration/--test_only segfaults in this binary, so test
# cams also appear in the source set; absolute scores are therefore optimistic
# but the relative comparison is unaffected.)
#
# Outputs -> output/edge_ablation/{no_edge,edge}/ + results_*.json + summary.txt

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

EXPORTER=(xvfb-run -a -s "-screen 0 1280x720x24" env __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./bin/RIFTCast_exporter)
DATASET="data/2026_03_20_orbbec_002_standard"
OUT="output/edge_ablation"
TEST_CALIB="$OUT/test_calib_full.json"   # 56-cam calib (renders all as test views)
GT_DIR="$DATASET/frame_00000/rgb"
MASK_DIR="$DATASET/frame_00000/mask"
TEST_CAMS=(5 26 37)
TEST_NAMES=(C0005 C0026 C0037)

mkdir -p "$OUT"

run_variant () {
    local label="$1" config="$2" dir="$3"
    echo ""
    echo "=== Render variant: $label ($config) ==="
    "${EXPORTER[@]}" \
        --dataset   "$config" \
        --output    "$OUT/$dir" \
        --test_file "$TEST_CALIB" \
        --inpaint   false \
        --frame     0

    echo ""
    echo "=== Metrics: $label ==="
    .venv/bin/python3 scripts/compute_metrics.py \
        --pred  "$OUT/$dir/frame_00000/test" \
        --gt    "$GT_DIR" \
        --mask  "$MASK_DIR" \
        --cams  "${TEST_CAMS[@]}" \
        --cam-names "${TEST_NAMES[@]}" \
        --label "$label" \
        --out   "$OUT/results_${dir}.json"
}

run_variant "TSDF without edge detection" "configs/real_03_tsdf_no_edge_gradient.json" "no_edge"
run_variant "TSDF with edge detection"    "configs/real_02_tsdf_edge_gradient.json"    "edge"

echo ""
echo "=== Summary ==="
.venv/bin/python3 - <<'EOF' | tee "$OUT/summary.txt"
import json
from pathlib import Path
out = Path("output/edge_ablation")
rows = [json.load(open(out / f"results_{d}.json")) for d in ("no_edge", "edge")]
hdr = f"{'Configuration':<32} {'PSNR':>7} {'SSIM':>8} {'LPIPS':>8}"
print(hdr); print("-" * len(hdr))
for r in rows:
    m = r["mean"]
    lp = f"{m['lpips']:.4f}" if m["lpips"] is not None else "  n/a "
    print(f"{r['label']:<32} {m['psnr']:>7.2f} {m['ssim']:>8.4f} {lp:>8}")
EOF

echo ""
echo "Done. Results in $OUT/"
