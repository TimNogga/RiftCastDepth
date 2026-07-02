#!/usr/bin/env bash
# Chamfer-distance vs number-of-depth-cameras sweep on the DIMPLED SPHERE dataset.
set -e
cd "$(dirname "$0")/.."

RGB=60
SPHERE_R=0.5
CAM_R=2.5
POCKETS=12
LEVEL=8
DEPTH_COUNTS=(${DEPTH_COUNTS:-1 2 4 8 12 16 24})
OUT=output/dimpled_experiment
RESULTS=$OUT/chamfer_results.csv
GT=$OUT/ground_truth.obj
mkdir -p "$OUT"
echo "num_depth,chamfer_mm" > "$RESULTS"

for N in "${DEPTH_COUNTS[@]}"; do
  echo "=== depth cameras: $N ==="
  DS=data/_dimpled_d$N
  CFG=configs/_dimpled_d$N.json
  GT_ARG=()
  # Regenerate the shared GT only on the first iteration (same pocket params every run).
  if [ "$N" = "${DEPTH_COUNTS[0]}" ]; then GT_ARG=(--gt-out "$GT"); fi

  python3 dataset/generate_dimpled_object.py \
      --num-rgb $RGB --num-depth $N --num-pockets $POCKETS \
      --sphere-radius $SPHERE_R --cam-radius $CAM_R \
      --dataset-root "$DS" --config-out "$CFG" "${GT_ARG[@]}" --force
  # Octree level 9 OOMs on this (solid-sphere) object at 24 GiB; level 8 (~6mm) fits and
  # comfortably resolves the 0.19 m pockets. Keep it constant across the whole sweep.
  python3 -c "import json;p='$CFG';d=json.load(open(p));d['reconstructor']['level']=$LEVEL;json.dump(d,open(p,'w'),indent=2)"

  xvfb-run -a -s "-screen 0 1280x720x24" \
      env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
      ./bin/RIFTCast_exporter --dataset "$CFG" --output "$OUT/recon_d$N" --frame 0 >/dev/null 2>&1

  python3 eval/recon_to_obj.py --recon "$OUT/recon_d$N/frame_00000" --out "$OUT/recon_d$N.obj"

  CH=$(python3 eval/chamfer.py --recon "$OUT/recon_d$N/frame_00000" --gt "$GT" --samples 150000 \
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
plt.plot(N,C,"o-",lw=2,ms=7,color="steelblue")
plt.xlabel("number of depth cameras"); plt.ylabel("Chamfer distance to GT [mm]")
plt.title("Reconstruction accuracy vs depth-camera count\n(dimpled sphere, ${RGB} RGB fixed, ${POCKETS} pockets)")
plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("$OUT/chamfer_vs_depth.png",dpi=130)
print("plot saved: $OUT/chamfer_vs_depth.png")
EOF
