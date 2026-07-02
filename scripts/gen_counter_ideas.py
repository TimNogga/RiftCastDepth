#!/usr/bin/env python3
"""Generate SVG mockups of slide page-counter ideas themed around the
RIFTCast reconstruction pipeline (visual-hull carving, depth cameras,
TSDF scalar field, point clouds, octree, the 'Eimer' bucket dataset).

No dependencies. Writes one SVG per concept + an index.html into
presentation/counter_ideas/.
"""
import os, math

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "presentation", "counter_ideas"))
os.makedirs(OUT, exist_ok=True)

# palette
BG     = "#0d1117"
PANEL  = "#161b22"
TEAL   = "#2dd4bf"
TEAL_D = "#0f766e"
GREY   = "#30363d"
FAINT  = "#21262d"
TXT    = "#c9d1d9"
SUB    = "#8b949e"
ACCENT = "#f0883e"

# 5x7 pixel font
FONT = {
 '0':["01110","10001","10011","10101","11001","10001","01110"],
 '1':["00100","01100","00100","00100","00100","00100","01110"],
 '2':["01110","10001","00001","00010","00100","01000","11111"],
 '3':["11111","00010","00100","00010","00001","10001","01110"],
 '4':["00010","00110","01010","10010","11111","00010","00010"],
 '5':["11111","10000","11110","00001","00001","10001","01110"],
 '6':["00110","01000","10000","11110","10001","10001","01110"],
 '7':["11111","00001","00010","00100","01000","01000","01000"],
 '8':["01110","10001","10001","01110","10001","10001","01110"],
 '9':["01110","10001","10001","01111","00001","00010","01100"],
 '/':["00001","00001","00010","00100","01000","10000","10000"],
 ' ':["00000","00000","00000","00000","00000","00000","00000"],
}

def cells(text):
    """yield (col, row, on) over the whole string laid out left-to-right."""
    x = 0
    for ch in text:
        g = FONT[ch]
        for r in range(7):
            for c in range(5):
                yield (x + c, r, g[r][c] == '1')
        x += 6  # 5 wide + 1 gap

def width_cols(text):
    return len(text) * 6 - 1

def header(w, h, title, sub):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="ui-sans-serif,Segoe UI,Helvetica,Arial,sans-serif">
<rect width="{w}" height="{h}" fill="{BG}"/>
<text x="40" y="52" fill="{TXT}" font-size="26" font-weight="700">{title}</text>
<text x="40" y="82" fill="{SUB}" font-size="16">{sub}</text>'''

def footer_note(w, h, note):
    return f'<text x="40" y="{h-28}" fill="{SUB}" font-size="14" font-style="italic">{note}</text></svg>'

def save(name, svg):
    with open(os.path.join(OUT, name), "w") as f:
        f.write(svg)
    print("wrote", name)

# ---------------------------------------------------------------- 1. CARVING
def concept_carving():
    w, h = 900, 360
    text = "07/24"
    cw = width_cols(text)
    s = 22
    ox = (w - cw * s) / 2
    oy = 150
    svg = [header(w, h, "1 — Visual-Hull Carving",
                  "Slide number is the survivor of the carve: ghost voxels = material removed, solid = the digit.")]
    # ghost (carved-away) voxels first
    for (c, r, on) in cells(text):
        x, y = ox + c * s, oy + r * s
        if not on:
            svg.append(f'<rect x="{x+1}" y="{y+1}" width="{s-2}" height="{s-2}" rx="2" fill="{FAINT}" stroke="{GREY}" stroke-width="0.8" stroke-dasharray="2 2"/>')
    # solid voxels with a little cube top/side for 3D feel
    for (c, r, on) in cells(text):
        if not on: continue
        x, y = ox + c * s, oy + r * s
        d = 4
        svg.append(f'<polygon points="{x},{y} {x+s-2},{y} {x+s-2+d},{y-d} {x+d},{y-d}" fill="{TEAL_D}"/>')
        svg.append(f'<polygon points="{x+s-2},{y} {x+s-2},{y+s-2} {x+s-2+d},{y+s-2-d} {x+s-2+d},{y-d}" fill="#0b5a54"/>')
        svg.append(f'<rect x="{x}" y="{y}" width="{s-2}" height="{s-2}" fill="{TEAL}"/>')
    svg.append(footer_note(w, h, "Animate: slide 1 = full block, each slide carves more away until the number appears."))
    save("1_carving.svg", "\n".join(svg))

# ---------------------------------------------------------- 2. POINT CLOUD
def concept_pointcloud():
    w, h = 900, 360
    text = "07/24"
    cw = width_cols(text)
    s = 22
    ox = (w - cw * s) / 2
    oy = 150
    import random
    random.seed(7)
    svg = [header(w, h, "2 — Point-Cloud Densification",
                  "Number drawn as a depth point cloud; later slides are denser / more complete.")]
    frac = 0.6  # how 'reconstructed' this slide is
    for (c, r, on) in cells(text):
        if not on: continue
        x, y = ox + c * s, oy + r * s
        for _ in range(7):
            px = x + random.uniform(1, s-3)
            py = y + random.uniform(1, s-3)
            if random.random() < frac:
                svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="1.7" fill="{TEAL}"/>')
            else:
                svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="1.4" fill="{SUB}" opacity="0.5"/>')
    svg.append(footer_note(w, h, "Progress bar = point density: sparse on slide 1, fully fused by the last slide."))
    save("2_pointcloud.svg", "\n".join(svg))

# ------------------------------------------------------------- 3. TSDF FIELD
def concept_tsdf():
    w, h = 900, 360
    text = "07/24"
    cw = width_cols(text)
    s = 22
    ox = (w - cw * s) / 2
    oy = 150
    svg = [header(w, h, "3 — TSDF Scalar Field",
                  "Digits filled with your signed-distance heatmap (blue = far/empty, red = surface).")]
    svg.append(f'''<defs><linearGradient id="tsdf" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#3b82f6"/><stop offset="45%" stop-color="#e2e8f0"/>
      <stop offset="60%" stop-color="#fbbf24"/><stop offset="100%" stop-color="#ef4444"/>
    </linearGradient></defs>''')
    for (c, r, on) in cells(text):
        x, y = ox + c * s, oy + r * s
        if on:
            svg.append(f'<rect x="{x}" y="{y}" width="{s-1}" height="{s-1}" fill="url(#tsdf)"/>')
        else:
            svg.append(f'<rect x="{x}" y="{y}" width="{s-1}" height="{s-1}" fill="{PANEL}"/>')
    # colorbar
    svg.append(f'<rect x="{w-70}" y="120" width="16" height="160" fill="url(#tsdf)" stroke="{GREY}"/>')
    svg.append(f'<text x="{w-48}" y="128" fill="{SUB}" font-size="12">+d</text>')
    svg.append(f'<text x="{w-48}" y="284" fill="{SUB}" font-size="12">-d</text>')
    svg.append(footer_note(w, h, "Bonus: marching-cubes iso-line could trace the digit outline as it animates."))
    save("3_tsdf.svg", "\n".join(svg))

# ---------------------------------------------------------- 4. CAMERA DOME
def concept_dome():
    w, h = 900, 410
    cx, cy, R = 450, 310, 200
    total, cur = 24, 7
    svg = [header(w, h, "4 — Capture-Dome Camera Count",
                  "“Camera 07 / 24”: cameras light up around the VCI dome as you advance.")]
    # dome arc
    svg.append(f'<path d="M {cx-R} {cy} A {R} {R} 0 0 1 {cx+R} {cy}" fill="none" stroke="{GREY}" stroke-width="2"/>')
    svg.append(f'<line x1="{cx-R-10}" y1="{cy}" x2="{cx+R+10}" y2="{cy}" stroke="{GREY}" stroke-width="2"/>')
    # head in centre
    svg.append(f'<circle cx="{cx}" cy="{cy-40}" r="26" fill="{FAINT}" stroke="{TEAL_D}" stroke-width="2"/>')
    svg.append(f'<circle cx="{cx-8}" cy="{cy-44}" r="3" fill="{TEAL}"/><circle cx="{cx+8}" cy="{cy-44}" r="3" fill="{TEAL}"/>')
    # cameras along the arc
    n = 12
    for i in range(n):
        a = math.pi * (i + 0.5) / n
        x = cx - R * math.cos(a)
        y = cy - R * math.sin(a)
        lit = i < round(cur / total * n)
        col = TEAL if lit else GREY
        dot = f'<circle r="2" fill="{ACCENT}"/>' if lit else ""
        svg.append(f'<g transform="translate({x:.1f},{y:.1f})">'
                   f'<rect x="-9" y="-7" width="18" height="14" rx="2" fill="{col}"/>'
                   f'<circle cx="0" cy="0" r="4" fill="{BG}"/>'
                   f'{dot}</g>')
    svg.append(f'<text x="{cx}" y="{cy+50}" fill="{TXT}" font-size="34" font-weight="700" text-anchor="middle">07 / 24</text>')
    svg.append(footer_note(w, h, "Ties the counter straight to your depth-camera ablation (N cameras)."))
    save("4_camera_dome.svg", "\n".join(svg))

# ------------------------------------------------------------- 5. OCTREE BAR
def concept_octree():
    w, h = 900, 360
    svg = [header(w, h, "5 — Octree Subdivision Bar",
                  "Progress as recursive subdivision: the more slides done, the finer the tree.")]
    x0, y0, W, H = 120, 130, 660, 150
    frac = 7 / 24
    def subdivide(x, y, ww, hh, depth, t):
        # cells left of the progress front subdivide deeper
        if depth >= 4 or x > x0 + W * (t + 0.18):
            on = x + ww/2 < x0 + W * t
            fill = TEAL if on else FAINT
            svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{ww-2:.1f}" height="{hh-2:.1f}" '
                       f'fill="{fill}" stroke="{GREY}" stroke-width="0.7"/>')
            return
        hw, hh2 = ww/2, hh/2
        for dx in (0, hw):
            for dy in (0, hh2):
                subdivide(x+dx, y+dy, hw, hh2, depth+1, t)
    subdivide(x0, y0, W, H, 0, frac)
    svg.append(f'<text x="{x0+W+20}" y="{y0+H/2+8}" fill="{TXT}" font-size="28" font-weight="700">07/24</text>')
    svg.append(footer_note(w, h, "Visual nod to your octree carving; doubles as a literal progress bar."))
    save("5_octree.svg", "\n".join(svg))

# --------------------------------------------------------------- 6. EIMER
def concept_eimer():
    w, h = 900, 360
    svg = [header(w, h, "6 — The “Eimer” Fills Up",
                  "Mascot from your export_depth_eimer set: the bucket fills with depth as you go.")]
    cx = 450
    topw, botw, top, bot = 150, 110, 130, 300
    frac = 7 / 24
    # bucket back
    svg.append(f'<path d="M {cx-topw/2} {top} L {cx+topw/2} {top} L {cx+botw/2} {bot} L {cx-botw/2} {bot} Z" fill="{PANEL}" stroke="{GREY}" stroke-width="3"/>')
    # liquid (depth) fill
    fy = bot - (bot - top) * frac
    fw_top = botw + (topw - botw) * frac
    svg.append(f'<defs><linearGradient id="dep" x1="0" y1="0" x2="0" y2="1">'
               f'<stop offset="0%" stop-color="{TEAL}"/><stop offset="100%" stop-color="{TEAL_D}"/></linearGradient></defs>')
    svg.append(f'<path d="M {cx-fw_top/2:.1f} {fy:.1f} L {cx+fw_top/2:.1f} {fy:.1f} L {cx+botw/2} {bot} L {cx-botw/2} {bot} Z" fill="url(#dep)"/>')
    svg.append(f'<ellipse cx="{cx}" cy="{fy:.1f}" rx="{fw_top/2:.1f}" ry="7" fill="{TEAL}" opacity="0.8"/>')
    # rim
    svg.append(f'<ellipse cx="{cx}" cy="{top}" rx="{topw/2}" ry="12" fill="none" stroke="{GREY}" stroke-width="3"/>')
    # falling depth droplet
    svg.append(f'<circle cx="{cx}" cy="{top-30}" r="6" fill="{ACCENT}"/>')
    svg.append(f'<text x="{cx+120}" y="{(top+bot)//2}" fill="{TXT}" font-size="30" font-weight="700">07/24</text>')
    svg.append(footer_note(w, h, "Memorable and funny — the bucket is full on the last slide."))
    save("6_eimer.svg", "\n".join(svg))

concept_carving(); concept_pointcloud(); concept_tsdf()
concept_dome(); concept_octree(); concept_eimer()

# ---------------------------------------------------------------- index.html
order = [
 ("1_carving.svg", "Visual-Hull Carving"),
 ("2_pointcloud.svg", "Point-Cloud Densification"),
 ("3_tsdf.svg", "TSDF Scalar Field"),
 ("4_camera_dome.svg", "Capture-Dome Camera Count"),
 ("5_octree.svg", "Octree Subdivision Bar"),
 ("6_eimer.svg", "The Eimer Fills Up"),
]
cards = "\n".join(
 f'<figure><img src="{f}" alt="{t}"/><figcaption>{t}</figcaption></figure>' for f, t in order)
html = f'''<!doctype html><meta charset="utf-8"><title>RIFTCast page-counter ideas</title>
<style>body{{background:{BG};color:{TXT};font-family:ui-sans-serif,Segoe UI,Arial;margin:0;padding:40px}}
h1{{font-weight:700}} p{{color:{SUB};max-width:760px}}
figure{{margin:0 0 36px}} img{{width:100%;max-width:900px;border:1px solid {GREY};border-radius:10px;display:block}}
figcaption{{color:{SUB};margin-top:8px}}</style>
<h1>RIFTCast — page-counter concept gallery</h1>
<p>Six ideas for a slide page counter themed around the reconstruction pipeline. Each SVG is editable
vector art — recolor via the palette constants in <code>scripts/gen_counter_ideas.py</code>.</p>
{cards}'''
with open(os.path.join(OUT, "index.html"), "w") as f:
    f.write(html)
print("wrote index.html ->", OUT)
