#!/usr/bin/env python3
"""TSDF-style page counter filled with REAL data.

The page number is cut out of the actual TSDF reconstruction render
(output/spy/03_tsdf_front.png): each digit is a window onto the scalar
field, with blocky 5x7 voxel cells. Proof of concept: renders 3 numbers.

Outputs SVG + PNG into presentation/tsdf_counter/.
"""
import os, io, base64
from PIL import Image

TOTAL = 34
SAMPLES = [5, 18, 31]            # three example slide numbers
SRC = "output/spy/03_tsdf_front.png"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "presentation", "tsdf_counter")
os.makedirs(OUT, exist_ok=True)

# palette (transparent bg; tuned to sit on a dark slide)
GHOST = "#1e3a5f"     # faint voxel-grid outline for "off" cells
SUBCOL = "#7f9cc0"    # the "/ NN" label

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
}

# --- embed a downscaled copy of the real TSDF render as a data URI -----------
def data_uri(path, target_w=300):
    im = Image.open(path).convert("RGBA")
    bbox = im.getbbox()                      # trim empty margins
    if bbox:
        im = im.crop(bbox)
    w, h = im.size
    im = im.resize((target_w, int(h * target_w / w)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return "data:image/png;base64," + b64, im.size

IMG, IMGSZ = data_uri(os.path.join(ROOT, SRC))

# geometry
S = 40                       # voxel cell size
GW, GH = 5 * S, 7 * S        # one glyph bounding box
GAP = S                      # gap between digits
PAD = 50


def glyph_clip(cid, gx, gy, pattern):
    rects = []
    for r in range(7):
        for c in range(5):
            if pattern[r][c] == '1':
                rects.append(f'<rect x="{gx+c*S}" y="{gy+r*S}" width="{S}" height="{S}"/>')
    return f'<clipPath id="{cid}">{"".join(rects)}</clipPath>'


def ghost_cells(gx, gy, pattern):
    out = []
    for r in range(7):
        for c in range(5):
            if pattern[r][c] == '0':
                out.append(f'<rect x="{gx+c*S+1}" y="{gy+r*S+1}" width="{S-2}" height="{S-2}" '
                           f'rx="3" fill="none" stroke="{GHOST}" stroke-width="1" opacity="0.5"/>')
    return "".join(out)


def frame(n):
    digits = f"{n:02d}"
    nblock_w = len(digits) * GW + (len(digits) - 1) * GAP
    label = f" / {TOTAL}"
    label_w = 150
    W = PAD * 2 + nblock_w + label_w
    H = PAD * 2 + GH
    clips, ghosts, imgs = [], [], []
    x = PAD
    for i, ch in enumerate(digits):
        pat = FONT[ch]
        cid = f"g{i}"
        clips.append(glyph_clip(cid, x, PAD, pat))
        ghosts.append(ghost_cells(x, PAD, pat))
        # the real reconstruction, sliced to fill this glyph, masked by the cells
        imgs.append(f'<image href="{IMG}" x="{x}" y="{PAD}" width="{GW}" height="{GH}" '
                    f'preserveAspectRatio="xMidYMid slice" clip-path="url(#{cid})"/>')
        x += GW + GAP
    # "/ NN" label, vertically centred
    lx = PAD + nblock_w + 26
    ly = PAD + GH * 0.62
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"
 font-family="ui-sans-serif,Segoe UI,Helvetica,Arial,sans-serif">
<defs>{"".join(clips)}</defs>
{"".join(ghosts)}
{"".join(imgs)}
<text x="{lx}" y="{ly}" font-size="86" font-weight="700" fill="{SUBCOL}">{label.strip()}</text>
</svg>'''
    return svg


def main():
    try:
        import cairosvg
        have_png = True
    except ImportError:
        have_png = False
    for n in SAMPLES:
        svg = frame(n)
        sp = os.path.join(OUT, f"tsdf_{n:02d}.svg")
        with open(sp, "w") as f:
            f.write(svg)
        if have_png:
            cairosvg.svg2png(url=sp, write_to=os.path.join(OUT, f"tsdf_{n:02d}.png"),
                             output_width=1400, background_color="#0d1117")
        print("wrote", os.path.basename(sp))
    print(f"source render embedded at {IMGSZ[0]}x{IMGSZ[1]}px -> {OUT}")


if __name__ == "__main__":
    main()
