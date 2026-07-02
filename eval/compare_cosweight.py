#!/usr/bin/env python3
"""Compare the uniform-average + binary-grazing-gate fusion (output/sweep_1to30) against the"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "output/sweep_1to30/sweep_1to30.csv"            # binary gate, uniform avg
B = ROOT / "output/sweep_1to30_cosweight/sweep_1to30.csv"  # Curless-Levoy cos(theta) weight
OUT = ROOT / "output/sweep_1to30_cosweight/compare_cosweight.png"


def load(p):
    d = {k: [] for k in ("n", "ch", "pc", "sa")}
    for r in csv.DictReader(open(p)):
        d["n"].append(int(r["num_depth"]))
        d["ch"].append(float(r["chamfer_mm"]))
        d["pc"].append(float(r["pocket_completeness_mm"]))
        d["sa"].append(float(r["smooth_accuracy_mm"]))
    return d


a, b = load(A), load(B)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
panels = [("ch", "global Chamfer (mm)", "Global Chamfer"),
          ("pc", "pocket completeness (mm)", "Pocket (concave) completeness"),
          ("sa", "smooth accuracy (mm)", "Smooth (convex) accuracy")]
for i, (key, ylab, title) in enumerate(panels):
    ax[i].plot(a["n"], a[key], "o-", c="C0", ms=4, label="uniform avg + binary grazing gate")
    ax[i].plot(b["n"], b[key], "s-", c="C3", ms=4, label="Curless-Levoy cos(θ) weight")
    ax[i].set_xlabel("# depth cameras"); ax[i].set_ylabel(ylab)
    ax[i].set_title(title); ax[i].grid(alpha=0.3); ax[i].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT, dpi=130, bbox_inches="tight")

# numeric summary at a few landmarks
def at(d, n):
    i = d["n"].index(n)
    return d["ch"][i], d["pc"][i], d["sa"][i]
print(f"{'N':>3} | {'chamfer A->B':>16} | {'pocket A->B':>16} | {'smooth A->B':>16}")
for n in (2, 5, 8, 12, 20, 30):
    ca, pa, sa = at(a, n); cb, pb, sb = at(b, n)
    print(f"{n:>3} | {ca:6.2f} -> {cb:6.2f}   | {pa:6.2f} -> {pb:6.2f}   | {sa:6.2f} -> {sb:6.2f}")
# means
import statistics as st
print("\nmean over N=1..30:")
for key, lab in (("ch","chamfer"),("pc","pocket_comp"),("sa","smooth_acc")):
    print(f"  {lab:11s}: binary {st.mean(a[key]):6.2f}  ->  cos-weight {st.mean(b[key]):6.2f}  "
          f"(Δ {st.mean(b[key])-st.mean(a[key]):+.2f} mm)")
print(f"\nwrote {OUT}")
