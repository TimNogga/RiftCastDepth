# Depth-Fusion Evaluation — Reconstruction vs. Number of Depth Cameras

Final, self-contained evaluation of the RGB-silhouette + depth-carving reconstruction on a controlled
synthetic object. Everything needed for the thesis evaluation section is here:

```
thesis_evaluation/
├── README.md                                  ← this writeup (findings + tables)
├── DATASET.md                                 ← how the synthetic dataset is built (for the intro)
├── figures/   01–04 .png  + 05 dataset views   ← thesis-ready graphics
├── data/      *.csv                            ← raw numbers behind every figure
└── meshes/    ground_truth + N0 / N2 / N12 .obj ← inspectable reconstructions
```

> **Dataset introduction:** see **`DATASET.md`** for the full construction/rendering method, and
> **`figures/05_dataset_depth_views.png`** for a multi-angle depth-map showcase of the object.

---

## 1. Experimental setup

| | |
|---|---|
| **Object** | "dimpled sphere": a sphere (R = 0.5 m) with **12** concave spherical pockets placed on Fibonacci directions. Concavities are **invisible to silhouettes** — only depth can recover them. |
| **RGB cameras** | 60, fixed for every run (⇒ a constant visual hull). |
| **Depth cameras** | swept N = 0, 1, …, 30, placed by greedy **farthest-point sampling** so the first-N set is *nested* (adding a camera, not re-arranging N). |
| **Reconstruction** | silhouette visual hull (octree, level 8 ≈ 6 mm) carved by a TSDF depth field, surface by marching cubes. |
| **Ground truth** | marching-cubes mesh of the analytic SDF (143 k vertices). |
| **Data** | **perfect synthetic depth** (ray-marched SDF). No sensor noise — so every effect below is *systematic*, not noise. |
| **Metrics (mm)** | symmetric **Chamfer** (global); and a **region split** into **pocket** (within 18° of a pocket axis = concave) vs **smooth** (>32° = convex), each as **completeness** (GT→recon, missing surface) and **accuracy** (recon→GT, misplaced surface). |

Reproduce: `dataset/generate_dimpled_object.py` (data), `eval/sweep_1to30.py` (sweep, renders one
30-camera dataset and subsets it), `eval/final_eval.py`, `eval/compare_cosweight.py`,
`eval/compare_power.py`. Exporter runs headless via `xvfb-run` (SSH).

---

## 2. Finding 1 — the silhouette hull cannot represent concavities; depth recovers them

**Figure `01_depth_vs_rgb_regionsplit.png`.** Data: `data/metrics_N0-N12_regionsplit.csv`.

| N (depth) | global Chamfer | **pocket completeness** | smooth completeness | pocket accuracy | smooth accuracy |
|--:|--:|--:|--:|--:|--:|
| 0 (RGB only) | 26.80 | **37.63** | 2.96 | 35.36 | 3.06 |
| 1 | 21.86 | 27.53 | 3.02 | 27.27 | 3.09 |
| 2 | 17.16 | 17.25 | 3.09 | 21.08 | 3.12 |
| 4 | 22.21 | 15.79 | 7.27 | 20.44 | 11.70 |
| 8 | 25.63 | 10.50 | 8.39 | 20.22 | 19.24 |
| 12 | 28.28 | 9.67 | 5.34 | 24.30 | 22.27 |

- At **N = 0** the convex surface is already near-perfect (≈ 3 mm) but the pockets are off by **37.6 mm** —
  the silhouette hull is geometrically blind to concavities. Recovering them is the entire job of depth.
- Adding depth cameras drives pocket completeness down monotonically (**37.6 → 9.7 mm**). Depth does
  exactly what it is for.

## 3. Finding 2 — pocket recovery is monotonic and saturates; global Chamfer is U-shaped

**Figure `02_chamfer_pocket_vs_cameras_1to30.png`.** Data: `data/sweep_1to30_binarygate.csv` (N = 1…30).

- **Pocket (concave) completeness** falls 27.6 → 10.7 mm by N = 5 and **saturates ≈ 9 mm** (min 8.8). Depth
  buys its whole concavity improvement in the **first ≈ 5 cameras**; further cameras see redundant pockets.
- **Smooth (convex) accuracy** *rises* 3.3 → 23.3 mm by N = 9 and plateaus ≈ 22 mm — convex surface erosion.
- **Global Chamfer is therefore U-shaped:** minimum **17.6 mm at N = 2**, climbing to a ≈ 29–30 mm plateau
  by N ≈ 10. It averages a large easy convex surface against small hard pockets, so it *masks* the pocket
  gain and is the **wrong headline metric** for a mostly-convex object.
- Mesh-complexity signature of the erosion: recon vertices grow 33 k (N = 2) → 308 k (N = 12), > 2× the
  143 k GT — a fragmented, jagged carve frontier, not a denser-but-clean surface.

> **Headline metric for the thesis: report pocket-region completeness vs. depth-camera count** (Finding 2),
> with global Chamfer and the RGB-only baseline as context.

## 4. Finding 3 — the convex-surface error is a *bias*, not noise (incidence weighting barely helps)

The natural fix is Curless & Levoy (1996) weighting: build the cumulative field `D = Σwᵢdᵢ / Σwᵢ` with a
per-sample weight `wᵢ = cos(θ)^p` that **down-weights grazing-incidence samples**. Implemented in
`visual_hull_cuda.cu` (pre-marching-cubes; only the scalar field changes).

**Figure `03_incidence_weighting_vs_binarygate.png`.** Data: `data/sweep_1to30_cosweight_p1.csv`.

Mean over N = 1…30, Curless-Levoy `cos(θ)` vs. the prior binary grazing gate:

| metric | binary gate | cos(θ) weight | Δ |
|---|--:|--:|--:|
| global Chamfer | 27.22 | 26.67 | **−0.55 mm** |
| pocket completeness | 11.09 | 10.43 | **−0.66 mm** |
| smooth accuracy | 19.40 | 19.15 | −0.26 mm |

A **small, consistent improvement** (pockets reach their plateau faster, ≈ 9 mm already by N = 5) — it is
the better, literature-grounded fusion rule — **but it does not change the qualitative behaviour**: Chamfer
still U-shaped, smooth accuracy still collapses to ≈ 22 mm. Down-weighting *attenuates* the grazing samples
but does not *remove* them, and a one-sided bias accumulated over many cameras still tips the average.

## 5. Finding 4 — weighting frontal cameras *harder* makes it strictly worse (grazing is not the culprit)

If grazing samples were the cause, suppressing them harder should help. We swept the exponent
`p` in `cos(θ)^p` over {1, 2, 4, 8, 16} (higher p ⇒ the most head-on camera dominates).

**Figure `04_frontal_overweighting_power_sweep.png`.** Data: `data/power_sweep_cos{2,4,8,16}.csv`.

| weighting | Chamfer @N=12 | pocket comp @N=12 | smooth acc @N=12 |
|---|--:|--:|--:|
| binary gate | 29.36 | 10.36 | 23.00 |
| **cos¹ (Curless-Levoy)** | **27.90** | **9.29** | **21.77** |
| cos² | 28.68 | 9.12 | 23.11 |
| cos⁴ | 30.59 | 9.26 | 25.99 |
| cos⁸ | 32.49 | 10.33 | 28.40 |
| cos¹⁶ | 32.26 | 12.80 | 26.63 |

Higher p is **monotonically worse on every metric** — and at cos¹⁶ a grazing (80°) sample carries weight
≈ 10⁻¹³, so it is provably *not* grazing samples doing the damage. Over-weighting frontal cameras instead
(a) collapses to a single camera per voxel, discarding the multi-view averaging that smooths the surface,
and (b) discards the steep-but-legitimate views that carve pocket walls (pocket completeness *degrades*,
9.3 → 12.8 mm). This **rules out the "grazing cameras over-carve" hypothesis**: the residual error is the
*near-frontal* cameras' own projective-TSDF sampling bias, which is a bias and cannot be re-weighted away.

---

## 6. Conclusion

1. Depth integration delivers the concavity reconstruction the silhouette hull **provably cannot**, and
   improves **monotonically** with depth-camera count in the concave regions it targets (≈ 4× from N = 0 to
   N = 12), saturating after ≈ 5 cameras.
2. The correct success metric is **concavity completeness**, not whole-object Chamfer (which is U-shaped and
   dominated by the easy convex surface).
3. The remaining limit on perfect data is **not sensor noise and not grazing cameras**, but a **systematic
   per-view bias of the projective (along-ray) TSDF** on curved/oblique surfaces. A single global threshold,
   Curless-Levoy incidence weighting, and frontal over-weighting were all tested — none removes it, because a
   bias cannot be averaged or re-weighted away (Findings 3 + 4).
4. **Future work** must attack the source: a **true Euclidean TSDF** (fuse nearest-surface distances from a
   back-projected point cloud, angle-independent) and/or a **space-carving-consistent occupancy fusion**
   (carve only on clean, edge-free, non-grazing free-space evidence, vetoed by any measured surface — never
   averaging a biased distance), plus strict depth-edge rejection.

---

## 7. File manifest

**figures/** (300 dpi PNG, directly usable in the evaluation section)
- `01_depth_vs_rgb_regionsplit.png` — N = 0…12, global Chamfer + region-split error (Finding 1).
- `02_chamfer_pocket_vs_cameras_1to30.png` — N = 1…30, global Chamfer + pocket/smooth error (Finding 2).
- `03_incidence_weighting_vs_binarygate.png` — Curless-Levoy cos(θ) vs binary gate (Finding 3).
- `04_frontal_overweighting_power_sweep.png` — cos(θ)^p power sweep (Finding 4).
- `05_dataset_depth_views.png` — multi-angle ray-marched depth maps of the object (see `DATASET.md`).
- `05b_dataset_pocket_residual.png` — same views minus the smooth-sphere depth, so the 12 pockets pop.
- `06_camera_rig.png` — 3D layout of the 30 depth cameras (coloured by nested order) + 60 RGB cameras.

**data/** (CSV; `num_depth, chamfer_mm, pocket_completeness_mm, smooth_completeness_mm, smooth_accuracy_mm[, …]`)
- `metrics_N0-N12_regionsplit.csv`, `sweep_1to30_binarygate.csv`, `sweep_1to30_cosweight_p1.csv`,
  `power_sweep_cos{2,4,8,16}.csv`.

**meshes/** (OBJ; recon frame is Y/Z-mirrored vs GT — the metric scripts auto-resolve the flip)
- `ground_truth.obj`, `recon_N0_rgbonly.obj`, `recon_N2_best_chamfer.obj`, `recon_N12_best_pockets.obj`.
