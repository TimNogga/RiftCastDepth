#!/usr/bin/env bash
# Chamfer-distance vs number-of-depth-cameras sweep, on the working (frontal, non-grazing) setup.
# Fixed RGB cameras (constant visual hull); only the depth-camera count varies.
set -e
cd "$(dirname "$0")/.."

RGB=40
HALF_ANGLE=30
RADIUS=2.5
DEPTH_COUNTS=(1 2 3 4 6 8 12 16 24 32)
OUT=output/tests/synth/depth_sweep
RESULTS=$OUT/chamfer_results.csv
mkdir -p "$OUT"
echo "num_depth,chamfer_mm" > "$RESULTS"

# Ground truth (open box, dataset frame)
python3 scripts/generate_gt_mesh.py --output "$OUT/ground_truth.obj" >/dev/null

for N in "${DEPTH_COUNTS[@]}"; do
  echo "=== depth cameras: $N ==="
  DS=data/_sweep_d$N
  CFG=configs/_sweep_d$N.json
  python3 scripts/generate_synth_data.py \
      --num-cameras $RGB --num-depth-cameras $N --depth-layout frontal --frontal-half-angle $HALF_ANGLE \
      --sphere-radius $RADIUS --dataset-root "$DS" --config-out "$CFG" \
      --depth-format png --force >/dev/null
  python3 -c "import json;p='$CFG';d=json.load(open(p));d['depth']['enable_cutter']=False;json.dump(d,open(p,'w'),indent=2)"

  xvfb-run -a -s "-screen 0 1280x720x24" \
      env __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
      ./bin/RIFTCast_exporter --dataset "$CFG" --output "$OUT/recon_d$N" --frame 0 >/dev/null 2>&1

  CH=$(python3 scripts/chamfer.py --recon "$OUT/recon_d$N/frame_00000" --gt "$OUT/ground_truth.obj" --samples 150000 \
        | grep -oP 'chamfer=\K[0-9.]+')
  echo "  chamfer = $CH mm"
  echo "$N,$CH" >> "$RESULTS"
done

echo "=== results ==="
cat "$RESULTS"

python3 - <<EOF
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import csv
N=[];C=[]
for r in csv.DictReader(open("$RESULTS")):
    N.append(int(r["num_depth"])); C.append(float(r["chamfer_mm"]))
plt.figure(figsize=(8,5))
plt.plot(N,C,"o-",lw=2,ms=7,color="steelblue")
plt.xlabel("number of depth cameras (frontal)"); plt.ylabel("Chamfer distance to GT [mm]")
plt.title("Reconstruction accuracy vs depth-camera count\n(open box, ${RGB} RGB fixed, frontal depth layout)")
plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("$OUT/chamfer_vs_depth.png",dpi=130)
print("plot saved: $OUT/chamfer_vs_depth.png")
EOF
