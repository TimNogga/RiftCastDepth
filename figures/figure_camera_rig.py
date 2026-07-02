#!/usr/bin/env python3
"""Camera-rig figure: 3D layout of the 30 depth cameras (and the 60 RGB cameras) around the object."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "data/_dimpled_d30"
OUT = ROOT / "output/thesis_evaluation/figures/06_camera_rig.png"
R_OBJ = 0.5

cams = json.loads((DS / "calibration_dome.json").read_text())["cameras"]
rgb = np.array([c["extrinsics"]["position"] for c in cams if c["camera_type"] == "rgb"], float)
dep = [c for c in cams if c["camera_type"] == "depth"]
dep.sort(key=lambda c: int(c["camera_id"][1:]))            # nested FPS order D000..D029
depp = np.array([c["extrinsics"]["position"] for c in dep], float)
order = np.arange(len(depp))

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="3d")

# object: translucent sphere at the origin
us, vs = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
ax.plot_surface(R_OBJ * np.cos(us) * np.sin(vs), R_OBJ * np.sin(us) * np.sin(vs),
                R_OBJ * np.cos(vs), color="0.6", alpha=0.25, linewidth=0, shade=True)

# sightlines from each depth camera to the origin
segs = [[(0, 0, 0), tuple(p)] for p in depp]
ax.add_collection3d(Line3DCollection(segs, colors="0.7", linewidths=0.5, alpha=0.5))

# RGB cameras (faint context)
ax.scatter(rgb[:, 0], rgb[:, 1], rgb[:, 2], s=14, c="0.6", marker="^",
           alpha=0.45, label=f"RGB cameras (n={len(rgb)})", depthshade=False)
# depth cameras, coloured by nested FPS order
sc = ax.scatter(depp[:, 0], depp[:, 1], depp[:, 2], s=70, c=order, cmap="viridis",
                edgecolors="k", linewidths=0.4, label=f"depth cameras (n={len(depp)})", depthshade=False)
for i, p in enumerate(depp):
    ax.text(p[0] * 1.06, p[1] * 1.06, p[2] * 1.06, str(i), fontsize=6.5, ha="center", va="center")

lim = 2.7
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
ax.set_box_aspect([1, 1, 1])
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
ax.view_init(elev=20, azim=35)
cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.10)
cbar.set_label("depth-camera order (nested farthest-point)")
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("wrote", OUT)
