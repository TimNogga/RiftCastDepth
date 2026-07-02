#!/usr/bin/env python3
"""Page-counter ideas that use the REAL dataset, shown intact (no dicing)."""
import os, io, base64, glob
from PIL import Image

TOTAL = 34
CUR = 7                       # the slide shown in every mockup
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "presentation", "data_counter_ideas")
os.makedirs(OUT, exist_ok=True)

RGB_DIR = os.path.join(ROOT, "data/2026_03_20_orbbec_002_standard/frame_00000/rgb")
CAMS = sorted(glob.glob(os.path.join(RGB_DIR, "C*.png")))
STAGES = [os.path.join(ROOT, "output/spy", f) for f in
          ("01_baseline_front.png", "02_cutter_front.png", "03_tsdf_front.png")]

BG    = "#0d1117"
CARD  = "#161b22"
TEAL  = "#2dd4bf"
GREY  = "#30363d"
TXT   = "#e6edf3"
SUB   = "#8b949e"

_cache = {}
def uri(path, size, square=True, circle=False):
    key = (path, size, square, circle)
    if key in _cache:
        return _cache[key]
    im = Image.open(path).convert("RGBA")
    bbox = im.getbbox()
    if bbox and "spy" in path:          # trim dark margins on renders only
        im = im.crop(bbox)
    w, h = im.size
    if square:
        s = min(w, h)
        im = im.crop(((w-s)//2, (h-s)//2, (w-s)//2+s, (h-s)//2+s))
    tw = size
    th = size if square else int(h*size/w)
    im = im.resize((tw, th), Image.LANCZOS)
    buf = io.BytesIO()
    if circle:
        im.save(buf, format="PNG")
    else:
        im.convert("RGB").save(buf, format="JPEG", quality=82)
    fmt = "png" if circle else "jpeg"
    u = f"data:image/{fmt};base64," + base64.b64encode(buf.getvalue()).decode()
    _cache[key] = (u, (tw, th))
    return _cache[key]

def header(w, h, title, sub):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'font-family="ui-sans-serif,Segoe UI,Helvetica,Arial,sans-serif">'
            f'<rect width="{w}" height="{h}" fill="{BG}"/>'
            f'<text x="40" y="50" fill="{TXT}" font-size="25" font-weight="700">{title}</text>'
            f'<text x="40" y="78" fill="{SUB}" font-size="15">{sub}</text>')

def counter(x, y, F, anchor="start"):
    """Page counter with each part at an absolute x (robust to whitespace handling)."""
    dw, sw, gap = 0.60*F, 0.45*F, 0.30*F
    total = 4*dw + 2*gap + sw
    X0 = x - total if anchor == "end" else x - total/2 if anchor == "middle" else x
    xsl = X0 + 2*dw + gap
    xn2 = xsl + sw + gap
    return (f'<text y="{y}" font-size="{F}" font-weight="800" fill="{TXT}">'
            f'<tspan x="{X0:.1f}">{CUR:02d}</tspan>'
            f'<tspan x="{xsl:.1f}" fill="{SUB}" font-weight="600">/</tspan>'
            f'<tspan x="{xn2:.1f}" fill="{SUB}" font-weight="600">{TOTAL}</tspan></text>')

def big_counter(x, y, anchor="end"):
    return counter(x, y, 46, anchor)

# ---------------------------------------------------------- A: contact sheet
def concept_grid():
    cols, t, gap = 6, 96, 8
    cams = CAMS[:36]
    rows = (len(cams)+cols-1)//cols
    gx, gy = 40, 110
    w = max(900, gx*2 + cols*(t+gap))
    h = gy + rows*(t+gap) + 90
    s = [header(w, h, "A — Multi-view contact sheet",
                "Your real capture cameras; the active view lights up, the rest dim. One per slide.")]
    for i, c in enumerate(cams):
        u, _ = uri(c, t)
        x = gx + (i % cols)*(t+gap)
        y = gy + (i//cols)*(t+gap)
        lit = i < CUR
        op = 1.0 if lit else 0.32
        s.append(f'<image href="{u}" x="{x}" y="{y}" width="{t}" height="{t}" '
                 f'preserveAspectRatio="xMidYMid slice" opacity="{op}" rx="6"/>')
        if i == CUR-1:
            s.append(f'<rect x="{x-2}" y="{y-2}" width="{t+4}" height="{t+4}" fill="none" '
                     f'stroke="{TEAL}" stroke-width="4"/>')
    s.append(big_counter(w-55, h-30))
    s.append('</svg>')
    save("A_contact_sheet.svg", "\n".join(s))

# ------------------------------------------------------------- B: filmstrip
def concept_filmstrip():
    w, h = 1100, 240
    s = [header(w, h, "B — Camera filmstrip (footer)",
                "A strip of real views; the current camera is centred and in colour.")]
    n = 9
    cw, ch = 150, 110
    cy = 110
    cx0 = w/2
    for k in range(-(n//2), n//2+1):
        idx = (CUR-1 + k) % len(CAMS)
        u, _ = uri(CAMS[idx], 150, square=False)
        scale = 1.0 if k == 0 else 0.8
        ww, hh = cw*scale, ch*scale
        x = cx0 + k*(cw*0.78) - ww/2
        y = cy - hh/2
        dist = abs(k)
        op = max(0.18, 1.0 - dist*0.22)
        s.append(f'<image href="{u}" x="{x:.1f}" y="{y:.1f}" width="{ww:.1f}" height="{hh:.1f}" '
                 f'preserveAspectRatio="xMidYMid slice" opacity="{op:.2f}"/>')
        if k == 0:
            s.append(f'<rect x="{x-3:.1f}" y="{y-3:.1f}" width="{ww+6:.1f}" height="{hh+6:.1f}" '
                     f'fill="none" stroke="{TEAL}" stroke-width="4"/>')
    s.append(counter(w/2, h-22, 30, "middle"))
    s.append('</svg>')
    save("B_filmstrip.svg", "\n".join(s))

# -------------------------------------------------------- C: progress ring
def concept_ring():
    import math
    w, h = 900, 420
    h = 440
    s = [header(w, h, "C — Reconstruction + progress ring",
                "The render shown whole; a ring fills with the deck. (Cross-fade hull->carved->fused across slides.)")]
    # the recon card (full figure, fit to a portrait card)
    u, (iw, ih) = uri(STAGES[2], 104, square=False)   # ~104 x 295
    cardx, cardy, cardw = 60, 110, 230
    s.append(f'<rect x="{cardx}" y="{cardy}" width="{cardw}" height="{ih+24}" rx="14" fill="{CARD}"/>')
    figx = cardx + (cardw - iw)/2
    s.append(f'<image href="{u}" x="{figx:.1f}" y="{cardy+12}" width="{iw}" height="{ih}"/>')
    # progress ring on the right
    cx, cy, R = 660, 230, 110
    frac = CUR/TOTAL
    s.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="none" stroke="{GREY}" stroke-width="14"/>')
    a0 = -math.pi/2
    a1 = a0 + 2*math.pi*frac
    large = 1 if frac > 0.5 else 0
    x0, y0 = cx+R*math.cos(a0), cy+R*math.sin(a0)
    x1, y1 = cx+R*math.cos(a1), cy+R*math.sin(a1)
    s.append(f'<path d="M {x0:.1f} {y0:.1f} A {R} {R} 0 {large} 1 {x1:.1f} {y1:.1f}" '
             f'fill="none" stroke="{TEAL}" stroke-width="14" stroke-linecap="round"/>')
    s.append(f'<text x="{cx}" y="{cy-4}" text-anchor="middle" font-size="52" font-weight="800" fill="{TXT}">{CUR:02d}</text>')
    s.append(f'<text x="{cx}" y="{cy+34}" text-anchor="middle" font-size="22" fill="{SUB}">of {TOTAL}</text>')
    s.append('</svg>')
    save("C_progress_ring.svg", "\n".join(s))

# ------------------------------------------------------- D: dome of views
def concept_dome():
    import math
    w, h = 980, 440
    cx, base, R = 490, 360, 300
    s = [header(w, h, "D — Dome of real views",
                "The dome you liked, but every node is an actual camera frame; lit ones in colour.")]
    s.append(f'<path d="M {cx-R} {base} A {R} {R} 0 0 1 {cx+R} {base}" fill="none" stroke="{GREY}" stroke-width="2"/>')
    s.append(f'<line x1="{cx-R-14}" y1="{base}" x2="{cx+R+14}" y2="{base}" stroke="{GREY}" stroke-width="2"/>')
    n = 13
    rad = 30
    s.append('<defs>')
    for k in range(n):
        s.append(f'<clipPath id="cc{k}"><circle cx="0" cy="0" r="{rad}"/></clipPath>')
    s.append('</defs>')
    for k in range(n):
        a = math.pi*(k+0.5)/n
        x = cx - R*math.cos(a)
        y = base - R*math.sin(a)
        idx = int(k/n*len(CAMS))
        lit = k < round(CUR/TOTAL*n)
        u, _ = uri(CAMS[idx], 70, circle=True)
        op = 1.0 if lit else 0.4
        ring = TEAL if lit else GREY
        s.append(f'<g transform="translate({x:.1f},{y:.1f})">'
                 f'<image href="{u}" x="{-rad}" y="{-rad}" width="{2*rad}" height="{2*rad}" '
                 f'clip-path="url(#cc{k})" opacity="{op}"/>'
                 f'<circle cx="0" cy="0" r="{rad}" fill="none" stroke="{ring}" stroke-width="3"/></g>')
    s.append(counter(cx, base+58, 40, "middle"))
    s.append('</svg>')
    save("D_dome_of_views.svg", "\n".join(s))

def save(name, svg):
    p = os.path.join(OUT, name)
    with open(p, "w") as f:
        f.write(svg)
    try:
        import cairosvg
        cairosvg.svg2png(url=p, write_to=p.replace(".svg", ".png"), output_width=1300)
    except ImportError:
        pass
    print("wrote", name)

concept_grid(); concept_filmstrip(); concept_ring(); concept_dome()

cards = "\n".join(
    f'<figure><img src="{f}"><figcaption>{t}</figcaption></figure>' for f, t in [
        ("A_contact_sheet.svg", "A — Multi-view contact sheet"),
        ("B_filmstrip.svg", "B — Camera filmstrip"),
        ("C_progress_ring.svg", "C — Reconstruction + progress ring"),
        ("D_dome_of_views.svg", "D — Dome of real views")])
html = f'''<!doctype html><meta charset="utf-8"><title>data-driven counter ideas</title>
<style>body{{background:{BG};color:{TXT};font-family:ui-sans-serif,Segoe UI,Arial;margin:0;padding:36px}}
img{{width:100%;max-width:1000px;border:1px solid {GREY};border-radius:10px;display:block}}
figure{{margin:0 0 34px}} figcaption{{color:{SUB};margin-top:8px}}</style>
<h1>RIFTCast — counter ideas using the real dataset</h1>
{cards}'''
with open(os.path.join(OUT, "index.html"), "w") as f:
    f.write(html)
print("found", len(CAMS), "camera views -> ", OUT)
