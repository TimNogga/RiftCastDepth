#!/usr/bin/env python3
"""
Schematic: a depth pixel near the silhouette straddles the object and the
background. Its returned range is the average of the near and far depth, which
falls behind the true surface, so the free-space vote erodes a thin shell of
real geometry (silhouette over-carving at grazing incidence).

Deliberately minimal: one object, one camera aimed along its pixel, two rays,
and two callouts. Outputs grazing_pixel.pdf and grazing_pixel.png.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
})

INK   = "#222222"   # camera, text
SURF  = "#1f4e79"   # true surface
ERODE = "#c0392b"   # eroded surface / removed material
RAY   = "#9a9a9a"   # rays
FILL  = "#e9eef3"   # object body

fig, ax = plt.subplots(figsize=(6.2, 4.6))
ax.set_aspect("equal"); ax.axis("off")

# ---- object: a convex arc -------------------------------------------------
R = 6.0
O = np.array([5.0, -2.2])
phi = np.linspace(np.deg2rad(82), np.deg2rad(150), 240)
arc = np.column_stack([O[0] + R*np.cos(phi), O[1] + R*np.sin(phi)])
body = np.vstack([arc, [[O[0]+2.5, O[1]], [O[0]+2.5, arc[0, 1]]]])
ax.add_patch(Polygon(body, closed=True, facecolor=FILL, edgecolor="none", zorder=0))
ax.plot(arc[:, 0], arc[:, 1], color=SURF, lw=2.4, zorder=4, solid_capstyle="round")

# ---- camera, aimed along its central pixel ray ----------------------------
C = np.array([-3.4, -1.4])
d = O - C; D = np.linalg.norm(d)
alpha = np.arcsin(R/D)                       # half-angle of the tangent cone
axis  = np.arctan2(d[1], d[0]) + alpha       # central ray grazes the silhouette
Tang  = C + np.sqrt(D*D - R*R)*np.array([np.cos(axis), np.sin(axis)])

def rot(pts, a):
    c, s = np.cos(a), np.sin(a)
    return pts @ np.array([[c, -s], [s, c]]).T

def draw_camera(pos, a, s=0.62):
    lens   = np.array([[0, -0.50], [-0.55, -0.32], [-0.55, 0.32], [0, 0.50]])
    bodyc  = np.array([[-0.55, -0.42], [-1.65, -0.42], [-1.65, 0.42], [-0.55, 0.42]])
    finder = np.array([[-1.42, 0.42], [-1.08, 0.42], [-1.08, 0.70], [-1.42, 0.70]])
    for shape in (bodyc, lens, finder):
        ax.add_patch(Polygon(rot(shape*s, a) + pos, closed=True,
                             facecolor=INK, edgecolor="none", zorder=6))

draw_camera(C, axis)
lab = C + rot(np.array([[-1.1, -0.95]]), axis)[0]
ax.text(lab[0], lab[1], "depth camera", ha="center", va="top",
        fontsize=10.5, color=INK)

# ---- the pixel: two rays straddling the silhouette ------------------------
def along(a, t): return C + t*np.array([np.cos(a), np.sin(a)])
def hit(a):
    u = np.array([np.cos(a), np.sin(a)]); f = C - O
    disc = np.dot(f, u)**2 - (np.dot(f, f) - R*R)
    if disc < 0: return None
    t = -np.dot(f, u) - np.sqrt(disc)
    return C + t*u if t > 0 else None

dth = np.deg2rad(2.4)
A = hit(axis - dth)                          # near ray hits the object
B = along(axis + dth, 11.5)                  # far ray slips into the background
ax.add_patch(Polygon([C, A, Tang, B], closed=True, facecolor=RAY, alpha=0.14,
                     edgecolor="none", zorder=1))
for p in (A, B):
    ax.plot([C[0], p[0]], [C[1], p[1]], color=RAY, lw=1.0, zorder=2)

# averaged depth = halfway between near and far range, behind the surface
d_mix = 0.5*(np.linalg.norm(A - C) + 11.5)
M = along(axis, d_mix)
ax.plot([C[0], M[0]], [C[1], M[1]], color=RAY, lw=1.0, ls=(0, (4, 3)), zorder=2)
ax.plot(*M, "o", color=INK, ms=5.5, zorder=7)
ax.plot(*A, "o", color=SURF, ms=5, zorder=7)

# ---- eroded surface: pulled inward, concentrated at the silhouette --------
phi_t = np.arctan2(*(Tang - O)[::-1])
bump = 0.38*np.exp(-((phi - (phi_t - 0.08))/0.22)**2)        # shell thickness
nrm = np.column_stack([np.cos(phi), np.sin(phi)])            # outward normals
er = arc - nrm*bump[:, None]
ax.fill(np.r_[arc[:, 0], er[::-1, 0]], np.r_[arc[:, 1], er[::-1, 1]],
        facecolor=ERODE, alpha=0.28, edgecolor="none", zorder=3)
ax.plot(er[:, 0], er[:, 1], color=ERODE, lw=1.8, ls=(0, (5, 2)), zorder=4)

# ---- callouts -------------------------------------------------------------
ax.annotate("near", A + np.array([0.05, -0.7]), fontsize=9, color=RAY,
            ha="center", va="top")
far_lbl = C + (np.linalg.norm(Tang - C) + 0.6)*np.array([np.cos(axis+dth), np.sin(axis+dth)])
ax.annotate("far", far_lbl + np.array([-0.45, 0.2]), fontsize=9, color=RAY, ha="right")
ax.annotate("averaged depth\nlands behind the surface", M, M + np.array([1.7, -2.0]),
            fontsize=9.5, color=INK, ha="center", va="center",
            arrowprops=dict(arrowstyle="-", color=INK, lw=0.9))
ax.annotate("eroded surface", er[155], er[155] + np.array([0, -1.2]),
            fontsize=9.5, color=ERODE, ha="center", va="center",
            arrowprops=dict(arrowstyle="-", color=ERODE, lw=0.9))
# label the surface in the clear upper-right area, with a leader to the curve
i = 24
lpos = arc[i] + nrm[i]*0.9 + np.array([0.4, 0.0])
ax.annotate("object surface", arc[i], lpos, fontsize=9.5, color=SURF,
            ha="center", va="bottom",
            arrowprops=dict(arrowstyle="-", color=SURF, lw=0.9))

ax.set_xlim(-4.6, 7.6); ax.set_ylim(-3.4, 4.6)
fig.tight_layout()
fig.savefig("grazing_pixel.pdf", bbox_inches="tight")
fig.savefig("grazing_pixel.png", dpi=200, bbox_inches="tight")
print("wrote grazing_pixel.pdf / .png")
