#!/usr/bin/env bash
# "NO VOTING" test sweep: kPositiveSdfThresholdMeters = 0 -> any positive SDF carves, no veto.
# Reuses the already-generated dimpled-sphere datasets/configs (no re-render) and the shared GT.
# Everything lands in a single fresh folder for easy inspection; one .obj per depth count.
set -e
cd "$(dirname "$0")/.."

OUT=${OUT:-output/no_voting_thr0}
SRC=output/dimpled_experiment          # source of the shared ground truth
DEPTH_COUNTS=(${DEPTH_COUNTS:-1 2 4 8 12 16})
mkdir -p "$OUT"
GT="$OUT/ground_truth.obj"
cp "$SRC/ground_truth.obj" "$GT"
RESULTS="$OUT/chamfer_results.csv"
echo "num_depth,chamfer_mm" > "$RESULTS"

for N in "${DEPTH_COUNTS[@]}"; do
  echo "=== depth cameras: $N ==="
  CFG=configs/_dimpled_d$N.json
  env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
      ./bin/RIFTCast_exporter --dataset "$CFG" --output "$OUT/recon_d$N" --frame 0 >"$OUT/recon_d$N.log" 2>&1

  python3 scripts/recon_to_obj.py --recon "$OUT/recon_d$N/frame_00000" --out "$OUT/recon_d$N.obj"

  CH=$(python3 scripts/chamfer.py --recon "$OUT/recon_d$N/frame_00000" --gt "$GT" --samples 150000 \
        | grep -oP 'chamfer=\K[0-9.]+')
  echo "  chamfer = $CH mm"
  echo "$N,$CH" >> "$RESULTS"
done

echo "=== results ==="
cat "$RESULTS"

python3 - <<EOF
import csv, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
N=[];C=[]
for r in csv.DictReader(open("$RESULTS")):
    N.append(int(r["num_depth"])); C.append(float(r["chamfer_mm"]))
plt.figure(figsize=(8,5))
plt.plot(N,C,"o-",lw=2,ms=7,color="indianred")
plt.xlabel("number of depth cameras"); plt.ylabel("Chamfer distance to GT [mm]")
plt.title("No-voting carve (threshold=0): Chamfer vs depth-camera count\n(dimpled sphere, 60 RGB fixed, 12 pockets)")
plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("$OUT/chamfer_vs_depth.png",dpi=130)
print("plot saved: $OUT/chamfer_vs_depth.png")
EOF
