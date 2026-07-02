#!/usr/bin/env bash
# Meta-sweep: for each carve-band value, patch the constexpr, rebuild the exporter, and run the
# dimpled depth-camera sweep (reusing datasets). Produces one results folder per band so we can see
# whether a LARGER band flattens the (currently anti-monotonic) chamfer-vs-cameras curve.
set -e
cd "$(dirname "$0")/.."
CU=RIFTCast/src/riftcast/_C/external/torchhull/src/torchhull/_C/src/visual_hull_cuda.cu
BANDS=(${BANDS:-0.06 0.10})
SUMMARY=output/band_meta_summary.csv
echo "band_m,num_depth,chamfer_mm" > "$SUMMARY"

for B in "${BANDS[@]}"; do
  echo "############ BAND = $B ############"
  sed -i -E "s/(constexpr float kPositiveSdfThresholdMeters = )[0-9.]+f;/\1${B}f;/" "$CU"
  grep -n "kPositiveSdfThresholdMeters = " "$CU"
  cmake --build build --target RIFTCast_exporter -j >/dev/null 2>&1
  TAG=$(echo "$B" | tr -d '.')
  OUT=output/band_${TAG} DEPTH_COUNTS="1 2 4 8 12" bash eval/no_voting_sweep.sh >/tmp/band_${TAG}.log 2>&1 || true
  awk -F, -v b="$B" 'NR>1{print b","$0}' output/band_${TAG}/chamfer_results.csv >> "$SUMMARY"
done

echo "=== SUMMARY (all bands) ==="
cat "$SUMMARY"
