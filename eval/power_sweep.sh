#!/usr/bin/env bash
# Sweep the Curless-Levoy incidence-weight exponent p in cos(theta)^p
set -e
cd "$(dirname "$0")/.."
CU=RIFTCast/src/riftcast/_C/external/torchhull/src/torchhull/_C/src/visual_hull_cuda.cu
SUBSET="1,2,3,4,5,6,8,10,12,16,20,25,30"

for P in 2 4 8 16; do
  echo "=================== power p=$P ==================="
  # set kIncidenceWeightPower = P.0f  (and make sure incidence weighting is ON)
  sed -i -E "s/(kIncidenceWeightPower[[:space:]]*=[[:space:]]*)[0-9.]+f/\1${P}.0f/" "$CU"
  sed -i -E "s/(kUseIncidenceWeight[[:space:]]*=[[:space:]]*)(true|false)/\1true/" "$CU"
  grep -nE "kIncidenceWeightPower[[:space:]]*=" "$CU" | head -1

  cmake --build build --target RIFTCast_exporter -j >/tmp/build_p$P.log 2>&1 || { echo "BUILD FAILED p=$P"; tail -20 /tmp/build_p$P.log; exit 1; }
  echo "built p=$P"

  SWEEP_OUT="power_sweep/p$P" DEPTH_LIST="$SUBSET" python3 -u eval/sweep_1to30.py \
      > output/power_sweep_p$P.run.log 2>&1
  echo "--- p=$P results ---"
  cat "output/power_sweep/p$P/sweep_1to30.csv"
done

# restore power to 1.0 (Curless-Levoy default) — leave the tree in a clean state
sed -i -E "s/(kIncidenceWeightPower[[:space:]]*=[[:space:]]*)[0-9.]+f/\11.0f/" "$CU"
echo "restored kIncidenceWeightPower=1.0f (rebuild needed to apply)"
