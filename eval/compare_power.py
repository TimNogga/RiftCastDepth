#!/usr/bin/env python3
"""Compare incidence-weight exponents p in cos(theta)^p (weighting face-on cameras ever harder)"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SERIES = [
    ("binary gate",      ROOT / "output/sweep_1to30/sweep_1to30.csv",            "C7", "--"),
    ("cos^1 (Curless-L.)", ROOT / "output/sweep_1to30_cosweight/sweep_1to30.csv", "C0", "-"),
    ("cos^2",            ROOT / "output/power_sweep/p2/sweep_1to30.csv",          "C2", "-"),
    ("cos^4",            ROOT / "output/power_sweep/p4/sweep_1to30.csv",          "C1", "-"),
    ("cos^8",            ROOT / "output/power_sweep/p8/sweep_1to30.csv",          "C3", "-"),
    ("cos^16",           ROOT / "output/power_sweep/p16/sweep_1to30.csv",         "C5", "-"),
]
OUT = ROOT / "output/power_sweep/compare_power.png"


def load(p):
    d = {k: [] for k in ("n", "ch", "pc", "sa")}
    for r in csv.DictReader(open(p)):
        d["n"].append(int(r["num_depth"])); d["ch"].append(float(r["chamfer_mm"]))
        d["pc"].append(float(r["pocket_completeness_mm"])); d["sa"].append(float(r["smooth_accuracy_mm"]))
    return d


data = [(lab, load(p), c, ls) for lab, p, c, ls in SERIES]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for key, ylab, title, i in [("ch", "global Chamfer (mm)", "Global Chamfer", 0),
                            ("pc", "pocket completeness (mm)", "Pocket (concave) completeness", 1),
                            ("sa", "smooth accuracy (mm)", "Smooth (convex) accuracy", 2)]:
    for lab, d, c, ls in data:
        ax[i].plot(d["n"], d[key], ls, marker="o", ms=3, c=c, label=lab)
    ax[i].set_xlabel("# depth cameras"); ax[i].set_ylabel(ylab)
    ax[i].set_title(title); ax[i].grid(alpha=0.3); ax[i].legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT, dpi=130, bbox_inches="tight")

print(f"{'p':>14} | {'cham N12':>8} {'cham N30':>8} | {'pock N12':>8} | {'smooth N12':>10}")
for lab, d, _, _ in data:
    def at(k, n): return d[k][d["n"].index(n)] if n in d["n"] else float("nan")
    print(f"{lab:>14} | {at('ch',12):8.2f} {at('ch',30):8.2f} | {at('pc',12):8.2f} | {at('sa',12):10.2f}")
print(f"\nwrote {OUT}")
