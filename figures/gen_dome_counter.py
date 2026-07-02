#!/usr/bin/env python3
"""Per-slide page counter: the VCI capture dome."""
import os, math

TOTAL = 34
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "presentation", "dome_counter"))

THEMES = {
    "light": dict(arc="#94a3b8", unlit="#94a3b8", lit="#0d9488",
                  lens="#ea580c", head="#0d9488", eye="#0d9488",
                  num="#0f172a", numsub="#64748b"),
    "dark":  dict(arc="#475569", unlit="#64748b", lit="#2dd4bf",
                  lens="#f97316", head="#2dd4bf", eye="#2dd4bf",
                  num="#e2e8f0", numsub="#94a3b8"),
}

# geometry
W, H = 900, 480
CX, BASE, R = 450, 400, 350
HEAD = (CX, BASE - 52)


def camera(x, y, faces, lit, c):
    """A little camera glyph at (x,y) rotated to face point `faces`."""
    ang = math.degrees(math.atan2(faces[1] - y, faces[0] - x))
    if lit:
        body = f'<rect x="-7" y="-7" width="14" height="14" rx="2.5" fill="{c["lit"]}"/>'
        lens = f'<circle cx="9" cy="0" r="3.5" fill="{c["lens"]}"/>'
    else:
        body = f'<rect x="-7" y="-7" width="14" height="14" rx="2.5" fill="none" stroke="{c["unlit"]}" stroke-width="1.6"/>'
        lens = f'<circle cx="9" cy="0" r="3.2" fill="none" stroke="{c["unlit"]}" stroke-width="1.6"/>'
    return (f'<g transform="translate({x:.1f},{y:.1f}) rotate({ang:.1f})">'
            f'{lens}{body}</g>')


def frame(n, theme):
    c = THEMES[theme]
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="ui-sans-serif,Segoe UI,Helvetica,Arial,sans-serif">']
    # dome: outer + faint inner arc + baseline
    s.append(f'<path d="M {CX-R} {BASE} A {R} {R} 0 0 1 {CX+R} {BASE}" fill="none" '
             f'stroke="{c["arc"]}" stroke-width="2" opacity="0.55"/>')
    s.append(f'<path d="M {CX-R+14} {BASE} A {R-14} {R-14} 0 0 1 {CX+R-14} {BASE}" fill="none" '
             f'stroke="{c["arc"]}" stroke-width="1" opacity="0.25"/>')
    s.append(f'<line x1="{CX-R-14}" y1="{BASE}" x2="{CX+R+14}" y2="{BASE}" '
             f'stroke="{c["arc"]}" stroke-width="2" opacity="0.55"/>')
    # head (the subject in the dome)
    hx, hy = HEAD
    s.append(f'<circle cx="{hx}" cy="{hy}" r="30" fill="none" stroke="{c["head"]}" stroke-width="2.5"/>')
    s.append(f'<circle cx="{hx-9}" cy="{hy-4}" r="3.4" fill="{c["eye"]}"/>'
             f'<circle cx="{hx+9}" cy="{hy-4}" r="3.4" fill="{c["eye"]}"/>')
    s.append(f'<path d="M {hx-8} {hy+12} Q {hx} {hy+18} {hx+8} {hy+12}" fill="none" '
             f'stroke="{c["eye"]}" stroke-width="2" stroke-linecap="round"/>')
    # cameras around the arc, filled left -> right; first n are lit
    for k in range(TOTAL):
        a = math.pi * (k + 0.5) / TOTAL
        x = CX - R * math.cos(a)
        y = BASE - R * math.sin(a)
        s.append(camera(x, y, HEAD, k < n, c))
    # counter text
    s.append(f'<text x="{CX}" y="{BASE+58}" text-anchor="middle" font-size="40" '
             f'font-weight="800" fill="{c["num"]}">{n:02d}'
             f'<tspan fill="{c["numsub"]}" font-weight="600"> / {TOTAL}</tspan></text>')
    s.append('</svg>')
    return "\n".join(s)


def main():
    try:
        import cairosvg
        have_png = True
    except ImportError:
        have_png = False
        print("cairosvg not found -> SVG only (no PNG)")

    for theme in THEMES:
        sdir = os.path.join(ROOT, theme, "svg")
        pdir = os.path.join(ROOT, theme, "png")
        os.makedirs(sdir, exist_ok=True)
        os.makedirs(pdir, exist_ok=True)
        for n in range(1, TOTAL + 1):
            svg = frame(n, theme)
            sp = os.path.join(sdir, f"dome_{n:02d}.svg")
            with open(sp, "w") as f:
                f.write(svg)
            if have_png:
                cairosvg.svg2png(url=sp, write_to=os.path.join(pdir, f"dome_{n:02d}.png"),
                                 output_width=1600)
        # preview grid
        thumbs = "\n".join(
            f'<div><img src="{theme}/svg/dome_{n:02d}.svg"><span>{n:02d}</span></div>'
            for n in range(1, TOTAL + 1))
        bg = "#ffffff" if theme == "light" else "#0d1117"
        fg = "#0f172a" if theme == "light" else "#c9d1d9"
        html = f'''<!doctype html><meta charset="utf-8"><title>dome counter — {theme}</title>
<style>body{{background:{bg};color:{fg};font-family:ui-sans-serif,Segoe UI,Arial;margin:0;padding:30px}}
h1{{font-weight:700}} .grid{{display:grid;grid-template-columns:repeat(6,1fr);gap:10px}}
.grid div{{position:relative;border:1px solid #8884;border-radius:8px;overflow:hidden}}
.grid img{{width:100%;display:block}} .grid span{{position:absolute;top:4px;left:6px;font-size:12px;opacity:.6}}</style>
<h1>VCI dome page counter — {theme} theme ({TOTAL} frames)</h1>
<div class="grid">{thumbs}</div>'''
        with open(os.path.join(ROOT, f"preview_{theme}.html"), "w") as f:
            f.write(html)
        print(f"{theme}: wrote {TOTAL} svg" + (" + png" if have_png else "") + " + preview")
    print("output ->", ROOT)


if __name__ == "__main__":
    main()
