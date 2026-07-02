#!/usr/bin/env python3
"""Capture-stage figure: 3D layout of the VCI dome with all camera frustums.

- RGB cameras (blue) vs. Orbbec depth cameras (red), each frustum textured with
  its synchronized image from frame_00000 (depth cameras show the depth map).
- Semi-transparent cylinder + floor disk sketch the physical stage geometry.
Output -> output/thesis_evaluation/figures/01_capture_stage.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "data/2026_03_20_orbbec_002_standard"
FRAME = DS / "frame_00000/rgb"
OUT = ROOT / "output/thesis_evaluation/figures/01_capture_stage.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

FRUSTUM_DEPTH = 0.72          # metres from pinhole to image plane in the sketch
TEX_W = 96                    # texture facets across the image plane
N_TEX_RGB = 9                 # how many RGB cameras get a textured image plane
C_RGB, C_DEPTH = "#2E6FA8", "#C0392B"

# world (x, y-down, z) -> plot (x, z, height): mpl draws z as the vertical axis
FLIP = np.array([[1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0],
                 [0.0, -1.0, 0.0]])

cams = json.loads((DS / "calibration_dome.json").read_text())["cameras"]

fig = plt.figure(figsize=(14, 7.2))
ax = fig.add_subplot(111, projection="3d")
ax.set_proj_type("persp", focal_length=0.38)

# ---- stage geometry: cylindrical wall, green floor, ceiling ring ----
R_STAGE, H_STAGE = 2.62, 2.95
th = np.linspace(0, 2 * np.pi, 80)
hh = np.linspace(0.0, H_STAGE, 2)
TH, HH = np.meshgrid(th, hh)
ax.plot_surface(R_STAGE * np.cos(TH), R_STAGE * np.sin(TH), HH,
                color="0.78", alpha=0.10, linewidth=0, shade=False)
rr = np.linspace(0, R_STAGE, 2)
RR, TH2 = np.meshgrid(rr, th)
ax.plot_surface(RR * np.cos(TH2), RR * np.sin(TH2), np.zeros_like(RR),
                color="#3D7A45", alpha=0.30, linewidth=0, shade=False)
for h, lw in [(0.0, 1.4), (H_STAGE, 1.4)]:
    ax.plot(R_STAGE * np.cos(th), R_STAGE * np.sin(th), np.full_like(th, h),
            color="0.55", lw=lw, alpha=0.65)
for t0 in np.linspace(0, 2 * np.pi, 12, endpoint=False):
    ax.plot([R_STAGE * np.cos(t0)] * 2, [R_STAGE * np.sin(t0)] * 2, [0, H_STAGE],
            color="0.62", lw=0.7, alpha=0.4)


def load_texture(cam):
    cid, is_depth = cam["camera_id"], cam["camera_type"] == "depth"
    img = Image.open(FRAME / f"{cid}.jpg").convert("L" if is_depth else "RGB")
    w, h = TEX_W, max(2, round(TEX_W * img.height / img.width))
    a = np.asarray(img.resize((w, h), Image.LANCZOS), float) / 255.0
    if is_depth:  # colormap the depth visualisation so the modality is obvious
        a = plt.get_cmap("inferno")(a)[..., :3]
    return a


def cam_pose(cam):
    M = np.linalg.inv(np.asarray(cam["extrinsics"]["view_matrix"]).reshape(4, 4))
    return FLIP @ M[:3, 3], FLIP @ M[:3, 0], FLIP @ M[:3, 1], FLIP @ M[:3, 2]


def draw_camera(cam, textured):
    is_depth = cam["camera_type"] == "depth"
    col = C_DEPTH if is_depth else C_RGB
    pos, right, down, fwd = cam_pose(cam)
    K = np.asarray(cam["intrinsics"]["camera_matrix"]).reshape(3, 3)
    W, H = cam["intrinsics"]["resolution"]
    hw = FRUSTUM_DEPTH * (W / 2) / K[0, 0]
    hh = FRUSTUM_DEPTH * (H / 2) / K[1, 1]
    ctr = pos + FRUSTUM_DEPTH * fwd
    corners = [ctr + sx * hw * right + sy * hh * down
               for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    segs = [[pos, c] for c in corners] + \
           [[corners[i], corners[(i + 1) % 4]] for i in range(4)]
    if is_depth:
        lw, lalpha = 1.6, 0.95
    elif textured:
        lw, lalpha = 1.0, 0.65
    else:
        lw, lalpha = 0.6, 0.28
    ax.add_collection3d(Line3DCollection(segs, colors=col, linewidths=lw,
                                         alpha=lalpha))
    ax.scatter(*pos, c=col, s=36 if is_depth else 20, depthshade=False,
               edgecolors="white", linewidths=0.5, zorder=5)
    if not textured:
        return pos

    tex = load_texture(cam)
    h, w = tex.shape[:2]
    u = np.linspace(-1, 1, w + 1)
    v = np.linspace(-1, 1, h + 1)
    U, V = np.meshgrid(u, v)
    X = ctr[0] + U * hw * right[0] + V * hh * down[0]
    Y = ctr[1] + U * hw * right[1] + V * hh * down[1]
    Z = ctr[2] + U * hw * right[2] + V * hh * down[2]
    ax.plot_surface(X, Y, Z, facecolors=tex, rstride=1, cstride=1,
                    shade=False, linewidth=0, antialiased=False)
    return pos


# texture only an azimuthally well-spread subset of the mid-height RGB cameras
# (floor/ceiling rings look too steep when textured and just add clutter)
rgb_cams = [c for c in cams if c["camera_type"] != "depth"
            and 0.7 < cam_pose(c)[0][2] < 2.3]
rgb_cams.sort(key=lambda c: np.arctan2(*cam_pose(c)[0][[1, 0]]))
step = len(rgb_cams) / N_TEX_RGB
tex_ids = {rgb_cams[int(i * step)]["camera_id"] for i in range(N_TEX_RGB)}

depth_pos = []
for cam in cams:
    p = draw_camera(cam, cam["camera_type"] == "depth"
                    or cam["camera_id"] in tex_ids)
    if cam["camera_type"] == "depth":
        depth_pos.append(p)

lim = 2.95
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(0, 3.2)
ax.set_box_aspect([1, 1, 0.56])
ax.set_axis_off()

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)


def save_view(path, elev, azim, dpi=300):
    ax.view_init(elev=elev, azim=azim)
    fig.savefig(path, dpi=dpi)
    # Axes3D reports the full axes box as its tight bbox -> crop manually.
    im = Image.open(path)
    a = np.asarray(im.convert("L"))
    ys, xs = np.where(a < 250)
    pad = 14
    box = (max(xs.min() - pad, 0), max(ys.min() - pad, 0),
           min(xs.max() + pad, im.width), min(ys.max() + pad, im.height))
    im.crop(box).save(path)
    print("wrote", path)


if "--sweep" in sys.argv:
    sweep_dir = OUT.parent / "angle_options"
    sweep_dir.mkdir(exist_ok=True)
    views = [(e, a) for e in (12, 26) for a in range(-180, 180, 36)]
    tiles = []
    for e, a in views:
        p = sweep_dir / f"elev{e:02d}_azim{a:+04d}.png"
        save_view(p, e, a, dpi=130)
        tiles.append((p, f"elev={e}  azim={a}"))
    # labelled contact sheet, 4 rows x 5 cols
    from PIL import ImageDraw
    tw, thh = 560, 360
    sheet = Image.new("RGB", (5 * tw, 4 * thh), "white")
    dr = ImageDraw.Draw(sheet)
    for i, (p, lbl) in enumerate(tiles):
        im = Image.open(p)
        s = min((tw - 8) / im.width, (thh - 30) / im.height)
        im = im.resize((int(im.width * s), int(im.height * s)))
        x0, y0 = (i % 5) * tw, (i // 5) * thh
        sheet.paste(im, (x0 + (tw - im.width) // 2, y0 + 24))
        dr.text((x0 + 10, y0 + 4), f"#{i + 1}  {lbl}", fill="black")
        dr.rectangle([x0, y0, x0 + tw - 1, y0 + thh - 1], outline=(200, 200, 200))
    sheet.save(sweep_dir / "contact_sheet.png")
    print("wrote", sweep_dir / "contact_sheet.png")
else:
    save_view(OUT, elev=12, azim=-180)
