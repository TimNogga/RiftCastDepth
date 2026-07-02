#!/usr/bin/env bash
# Grazing-rejection experiment. Fix a SMALL carve band (deep pockets) and tighten the per-camera
# depth-gradient gate (is_depth_edge_strong) so each camera ABSTAINS on its grazing limb (steep
# foreshortening) while still carving head-on pockets. Sweep the gradient threshold. This targets the
# user's "not deep enough in pockets AND too much at wrong places (limb)" — a single band can't do both.
set -e
cd "$(dirname "$0")/.."
CU=RIFTCast/src/riftcast/_C/external/torchhull/src/torchhull/_C/src/visual_hull_cuda.cu

BAND=${BAND:-0.02}
RADIUS=${RADIUS:-3}
THRESHOLDS=(${THRESHOLDS:-0.06 0.03 0.015})

# Fixed knobs: small band + larger gradient baseline.
sed -i -E "s/(constexpr float kPositiveSdfThresholdMeters = )[0-9.]+f;/\1${BAND}f;/" "$CU"
sed -i -E "s/(use_synthetic_depth_tuning\(preset\) \? )[0-9]+( : 10;)/\1${RADIUS}\2/" "$CU"

EXPDIR=${EXPDIR:-output/grazing_experiment}
mkdir -p "$EXPDIR"
SUMMARY=$EXPDIR/grazing_summary.csv
echo "edge_thr,num_depth,chamfer_mm" > "$SUMMARY"

for T in "${THRESHOLDS[@]}"; do
  echo "############ band=$BAND radius=$RADIUS edge_thr=$T ############"
  sed -i -E "s/(use_synthetic_depth_tuning\(preset\) \? )[0-9.]+f( : 0\.04f;)/\1${T}f\2/" "$CU"
  grep -nE "kPositiveSdfThresholdMeters = |use_synthetic_depth_tuning\(preset\) \? [0-9.]+f : 0\.04f|use_synthetic_depth_tuning\(preset\) \? [0-9]+ : 10" "$CU"
  cmake --build build --target RIFTCast_exporter -j >/dev/null 2>&1
  TAG=$(echo "$T" | tr -d '.')
  OUT=$EXPDIR/thr_${TAG} DEPTH_COUNTS="1 2 4 8 12" bash scripts/no_voting_sweep.sh >"$EXPDIR/thr_${TAG}_sweep.log" 2>&1 || true
  awk -F, -v t="$T" 'NR>1{print t","$0}' "$EXPDIR/thr_${TAG}/chamfer_results.csv" >> "$SUMMARY"
done

echo "=== GRAZING SUMMARY (band=$BAND radius=$RADIUS) ==="
cat "$SUMMARY"
