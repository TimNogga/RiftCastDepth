#!/usr/bin/env python3
"""
Schematic for the thesis: why grazing-incidence depth cameras erode a convex
surface (silhouette over-carving).

Panel (a): a single pixel near the silhouette straddles object and background.
           Its footprint is stretched at grazing incidence, the returned depth
           is an average that lands *behind* the true surface, and the resulting
           free-space TSDF vote carves a sliver of true surface away.
Panel (b): the geometric (not random) error accumulates with every additional
           grazing camera instead of averaging out.

Outputs grazing_erosion.pdf (vector, for LaTeX) and grazing_erosion.png.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Wedge, Circle
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
})

# ---- colours -------------------------------------------------------------
C_OBJ    = "#cfe3f2"   # object fill
C_SURF   = "#1f4e79"   # true surface
C_ERODE  = "#c44e52"   # eroded / carved
C_RAY    = "#7f7f7f"   # rays
C_CAM    = "#2e2e2e"   # camera
C_FREE   = "#f4c6a0"   # carved free space
C_PHANT  = "#b8860b"   # phantom depth sample

# =========================================================================
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.6))

# -------------------------------------------------------------------------
# Panel (a): the mechanism, zoomed on one pixel at the silhouette
# -------------------------------------------------------------------------
ax = axA
ax.set_aspect("equal")
ax.axis("off")

# convex object: a big circle, we only show the left-facing arc region
R = 6.0
O = np.array([5.2, -2.0])          # circle centre (off to the right/below)
th = np.linspace(np.deg2rad(95), np.deg2rad(160), 200)
arc = np.column_stack([O[0] + R*np.cos(th), O[1] + R*np.sin(th)])
# fill the object body (close the polygon toward lower-right)
body = np.vstack([arc, [[O[0]+2, O[1]-1], [O[0]+2, arc[0,1]]]])
ax.add_patch(Polygon(body, closed=True, facecolor=C_OBJ, edgecolor="none", zorder=0))
ax.plot(arc[:,0], arc[:,1], color=C_SURF, lw=2.2, zorder=4, label="true surface")

# camera at lower-left, viewing the surface at grazing incidence
C = np.array([-3.6, -1.7])
# camera body
ax.add_patch(plt.Rectangle((C[0]-0.45, C[1]-0.32), 0.9, 0.64,
                           facecolor=C_CAM, edgecolor="none", zorder=5))
ax.add_patch(Polygon([[C[0]+0.45, C[1]+0.32],[C[0]+0.45, C[1]-0.32],
                      [C[0]+0.95, C[1]-0.55],[C[0]+0.95, C[1]+0.55]],
                     closed=True, facecolor=C_CAM, edgecolor="none", zorder=5))
ax.text(C[0], C[1]-0.7, "depth camera", ha="center", va="top", fontsize=10)

# silhouette / tangent point: where camera ray is tangent to the circle
d = O - C
D = np.linalg.norm(d)
alpha = np.arcsin(R/D)                       # half angle of the tangent cone
base = np.arctan2(d[1], d[0])
tdir = base + alpha                          # upper tangent direction
Tang = C + np.sqrt(D*D - R*R) * np.array([np.cos(tdir), np.sin(tdir)])

# the pixel = an angular wedge straddling the tangent direction
dth = np.deg2rad(2.6)
r_lo = tdir - dth                           # ray that hits the surface (foreground)
r_hi = tdir + dth                           # ray that slips past -> background

def ray_circle_hit(Cpt, ang, O, R):
    """nearest intersection of ray from Cpt with circle, or None."""
    u = np.array([np.cos(ang), np.sin(ang)])
    f = Cpt - O
    b = 2*np.dot(f, u); c = np.dot(f, f) - R*R
    disc = b*b - 4*c
    if disc < 0: return None
    t = (-b - np.sqrt(disc))/2
    return Cpt + t*u if t > 0 else None

def along(Cpt, ang, t): return Cpt + t*np.array([np.cos(ang), np.sin(ang)])
# the wedge straddles the tangent: one bounding ray hits the body, one misses.
hit_lo = ray_circle_hit(C, r_lo, O, R)
hit_hi = ray_circle_hit(C, r_hi, O, R)
if hit_lo is not None:                       # r_lo hits -> foreground, r_hi misses
    A, B = hit_lo, along(C, r_hi, 12.5)
else:                                         # the other way round
    A, B = hit_hi, along(C, r_lo, 12.5)

# central measured depth = average of near and far range along central ray
d_near = np.linalg.norm(A - C)
d_far  = 12.5
d_mix  = 0.5*(d_near + d_far)               # averaged / mixed depth
M = along(C, tdir, d_mix)                    # phantom sample, behind true surface

# free-space carved region: from camera up to (phantom depth - band) inside object
band = 0.9
carve_stop = d_mix - band
P_stop = along(C, tdir, carve_stop)

# shade the eroded sliver: between true surface and carve stop, near the edge
# approximate by polygon: along surface from A to Tang, then back along central ray
seg_th = np.linspace(np.arctan2(*(A-O)[::-1]), np.arctan2(*(Tang-O)[::-1]), 30)
seg = np.column_stack([O[0]+R*np.cos(seg_th), O[1]+R*np.sin(seg_th)])
erode_poly = np.vstack([seg, [P_stop]])
ax.add_patch(Polygon(erode_poly, closed=True, facecolor=C_ERODE, alpha=0.55,
                     edgecolor="none", zorder=3))

# the two bounding rays of the pixel
for ang, pt, c in [(r_lo, A, C_RAY), (r_hi, B, C_RAY)]:
    ax.add_line(Line2D([C[0], pt[0]], [C[1], pt[1]], color=c, lw=1.0,
                       ls="--", zorder=2))
# central ray to the phantom depth
ax.add_line(Line2D([C[0], M[0]], [C[1], M[1]], color=C_PHANT, lw=1.4, zorder=2))

# markers
ax.plot(*A, "o", color=C_SURF, ms=6, zorder=6)
ax.plot(*M, "D", color=C_PHANT, ms=8, zorder=6)
ax.plot(*Tang, "o", mfc="white", mec=C_SURF, ms=6, zorder=6)

# annotations
ax.annotate("foreground\n(near depth)", A, A+np.array([-2.4, 0.2]),
            fontsize=9, ha="center", va="center",
            arrowprops=dict(arrowstyle="->", color=C_SURF, lw=1))
ax.annotate("background\n(far depth)", (B+Tang)/2 + np.array([0.1, 0.4]),
            fontsize=9, ha="center", color=C_RAY)
ax.annotate("averaged depth\nfalls behind surface", M, M+np.array([2.3, -1.7]),
            fontsize=9, ha="center", va="center", color=C_PHANT,
            arrowprops=dict(arrowstyle="->", color=C_PHANT, lw=1.1))
ax.annotate("eroded sliver\n(spurious free-space vote)",
            (P_stop+seg[15])/2 + np.array([0.1,-0.05]),
            np.array([4.7, -2.3]),
            fontsize=9, ha="center", va="center", color=C_ERODE,
            arrowprops=dict(arrowstyle="->", color=C_ERODE, lw=1.1))
ax.text(Tang[0]-0.25, Tang[1]+0.45, "silhouette", fontsize=9, color=C_SURF,
        ha="right")

ax.set_title("(a)  A grazing pixel straddles the silhouette",
             fontsize=11, loc="left")
ax.set_xlim(-4.6, 7.4); ax.set_ylim(-3.4, 4.6)

# -------------------------------------------------------------------------
# Panel (b): the error is geometric -> accumulates with more cameras
# -------------------------------------------------------------------------
ax = axB
ax.set_aspect("equal")
ax.axis("off")

cx, cy, Rb = 0.0, 0.0, 2.4
phi = np.linspace(0, 2*np.pi, 400)
# true surface
ax.plot(cx + Rb*np.cos(phi), cy + Rb*np.sin(phi), color=C_SURF, lw=2.2, zorder=4)
ax.add_patch(Circle((cx, cy), Rb, facecolor=C_OBJ, edgecolor="none", zorder=0))

# grazing cameras around the object, each carving a small inward bite
n_cams = 7
cam_ang = np.linspace(0.15, 2*np.pi-0.15, n_cams)
erosion = 0.0
# build an eroded radius profile: each camera removes a scallop near its grazing edge
prof = np.full_like(phi, Rb)
for a in cam_ang:
    cpos = np.array([cx + 5.4*np.cos(a), cy + 5.4*np.sin(a)])
    # camera marker
    ax.add_patch(plt.Rectangle((cpos[0]-0.16, cpos[1]-0.16), 0.32, 0.32,
                               angle=0, facecolor=C_CAM, edgecolor="none", zorder=5))
    ax.add_line(Line2D([cpos[0], cx+Rb*np.cos(a+0.0)],
                       [cpos[1], cy+Rb*np.sin(a+0.0)],
                       color=C_RAY, lw=0.7, ls="--", zorder=1))
    # each camera erodes a band offset to the grazing side of its view
    bite = 0.28*np.exp(-((np.angle(np.exp(1j*(phi-(a+0.55))))/0.6)**2))
    prof = prof - bite

prof = np.clip(prof, 1.2, Rb)
ex = cx + prof*np.cos(phi); ey = cy + prof*np.sin(phi)

# shade the ring of removed material between true and eroded surface
ring_outer = np.column_stack([cx+Rb*np.cos(phi), cy+Rb*np.sin(phi)])
ring_inner = np.column_stack([ex, ey])[::-1]
ax.add_patch(Polygon(np.vstack([ring_outer, ring_inner]), closed=True,
                     facecolor=C_ERODE, alpha=0.40, edgecolor="none", zorder=2))
ax.plot(ex, ey, color=C_ERODE, lw=1.8, ls="-", zorder=4)

ax.set_title("(b)  Geometric error accumulates with camera count",
             fontsize=11, loc="left")
ax.text(cx, cy, "convex\nobject", ha="center", va="center", fontsize=9,
        color=C_SURF, zorder=6)
ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)

# shared legend
legend_elems = [
    Line2D([0],[0], color=C_SURF, lw=2.2, label="true surface"),
    Line2D([0],[0], color=C_ERODE, lw=2.0, label="reconstructed (eroded) surface"),
    Line2D([0],[0], marker='D', color=C_PHANT, lw=0, label="averaged / phantom depth"),
    Line2D([0],[0], color=C_RAY, lw=1.0, ls="--", label="depth-camera rays"),
]
fig.legend(handles=legend_elems, loc="lower center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, -0.02), fontsize=9.5)

fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig("grazing_erosion.pdf", bbox_inches="tight")
fig.savefig("grazing_erosion.png", dpi=200, bbox_inches="tight")
print("wrote grazing_erosion.pdf / .png")
