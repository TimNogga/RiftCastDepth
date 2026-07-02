# The synthetic "dimpled sphere" dataset — construction & rendering

This is the controlled test object used throughout the depth-fusion evaluation. It was designed so the
contribution of depth can be **isolated**: a smooth convex body that the RGB silhouette captures almost
perfectly, with concavities that silhouettes are geometrically blind to and only depth can recover.
Everything is generated analytically and noise-free, so any reconstruction error is *systematic*, not
sensor noise. Generator: `dataset/generate_dimpled_object.py`. Showcase: `figures/05_dataset_depth_views.png`.

## 1. Design rationale

| Requirement | Choice |
|---|---|
| Silhouette captures the body, **not** the detail | a **sphere** (its outline is a circle from every view, so the RGB visual hull is just the smooth sphere) |
| Detail that **only depth** can see | **concave pockets** carved into the surface (a concavity never changes the silhouette) |
| A meaningful *camera-count* ablation | pockets on **all sides** (Fibonacci-distributed) so each viewing direction reveals a different subset of pockets |
| "add a camera", not "rearrange N" | depth cameras placed by **nested** farthest-point sampling |
| Errors attributable to the method, not noise | **perfect, analytic depth** (no sensor model) |

## 2. Object definition — constructive solid geometry as a signed distance field

The object is a Boolean **difference** of spheres, expressed directly as a signed distance field (SDF,
negative inside). Starting from the base sphere `d = ‖p‖ − R`, each pocket is *subtracted* with the
standard CSG-difference operator `A∖B = max(A, −B)`:

```
d(p) = max(  ‖p‖ − R ,   max over pockets k of [ r_pocket − ‖p − c_k‖ ]  )
```

- **Base sphere:** radius `R = 0.5 m`.
- **Pockets:** `12` spheres of radius `r_pocket = 0.19 m`, whose centres `c_k` lie on Fibonacci-sphere
  directions at radius `R + offset = 0.60 m` — i.e. just *outside* the surface, so each carves a
  shallow spherical-cap "bowl" ≈ `0.09 m` deep into the sphere.

Because the geometry is an SDF, both the ground truth and the depth maps are derived from the *same*
analytic function — there is no mesh-vs-render mismatch.

## 3. Ground-truth mesh

The SDF is sampled on a `256³` grid over `[−0.75, 0.75]³ m` and triangulated with **marching cubes**
(`skimage.measure.marching_cubes`, iso-level 0), then the voxel indices are mapped back to world
coordinates. This yields the reference mesh (~143 k vertices) all metrics compare against.

## 4. Camera rig

All cameras look at the origin (OpenCV look-at convention; `view = [R | t]`).

- **RGB (60, fixed):** placed on a **Fibonacci sphere** of radius `2.5 m` for an even, dense, isotropic
  silhouette coverage → a constant, well-formed visual hull for every run.
- **Depth (N, swept 1…30):** placed by **greedy farthest-point sampling** over a 4000-point Fibonacci
  candidate cloud, radius `2.5 m`. The greedy order is deterministic, so the first *N* positions are a
  **prefix** of the first *M > N* — adding a depth camera keeps all previous ones and only appends new,
  maximally-separated ones. This makes the camera-count sweep strictly **additive** (a naïve Fibonacci
  prefix would clump at one pole). Cameras view the object head-on, avoiding artificial grazing setups.
- **Intrinsics:** ideal pinhole, `fx = fy = 1400 px`, principal point at the image centre, resolution
  **1920 × 1080**. (The reconstruction's mask reader only accepts this resolution or 5328 × 4608, so it
  is fixed at 1920 × 1080.)

## 5. Rendering

- **RGB / silhouette mask** — analytic **ray–sphere intersection**: a pixel is "object" iff its ray hits
  the base sphere (the discriminant of the ray–sphere quadratic is ≥ 0). Pockets are deliberately *not*
  in the silhouette (a concavity cannot change an outline). The RGB image is just this mask (texture is
  irrelevant — only the silhouette feeds the visual hull); depth cameras get a full-coverage mask.
- **Depth map** — a **sphere-tracing-style** SDF surface finder. Only rays inside the silhouette are
  marched, over the sphere's entry→exit interval `[t_near, t_far]` in `256` fixed steps; the first SDF
  sign change (`+ → −`) along the ray is the surface hit. The along-ray hit distance is converted to an
  optical-axis (Z) depth via the ray's camera-space z-component; pixels with no hit get a `5.0 m`
  background. The pockets show up as the **deeper (farther) patches** on the sphere face — see the
  showcase figure. Output is exact, smooth, noise-free metric depth.

## 6. Storage & calibration

- Depth: 16-bit PNG in **millimetres** (`depth_m × 1000`); `depth.scale = 1000` in the config.
- Masks: raw `uint8` `.bin`; RGB: 3-channel JPG of the mask.
- `calibration_dome.json`: per camera `camera_id`, `camera_type`, intrinsics (resolution + 3×3 matrix),
  extrinsics (position + 4×4 view matrix). The run config is `VCI_REAL` v2.1, `depth_fusion_mode =
  synthetic`, volume scale 1.6, octree level 8 (≈ 6 mm — comfortably resolves the 0.09 m pockets without
  OOM). The 30-camera dataset (`data/_dimpled_d30`) is rendered once; each *N* uses its first-N depth
  cameras (the nested-prefix property).

## 7. Parameter summary

| Parameter | Value |
|---|---|
| Sphere radius `R` | 0.50 m |
| Pockets | 12, Fibonacci-placed |
| Pocket radius / centre offset / depth | 0.19 m / 0.10 m / ≈ 0.09 m |
| RGB cameras | 60, Fibonacci, r = 2.5 m |
| Depth cameras | 1…30, nested farthest-point, r = 2.5 m |
| Intrinsics | fx = fy = 1400 px, 1920 × 1080, centred principal point |
| GT extraction | 256³ SDF grid over [−0.75, 0.75] m, marching cubes @ 0 |
| Depth render | 256-step SDF root-find, 5.0 m background, noise-free |

**Figures.** `05_dataset_depth_views.png` — ray-marched depth from 8 of the 30 depth cameras
(azimuth/elevation labelled), pockets = deeper patches, background masked. `05b_dataset_pocket_residual.png`
— the same views with the analytic smooth-sphere depth subtracted, so the 12 concave pockets stand out as
the recessed (red) bowls. `06_camera_rig.png` — 3D layout of the 30 depth cameras (coloured by nested
farthest-point order) and the 60 RGB cameras around the object.
Regenerate with `figures/figure_dataset_depths.py`, `figure_dataset_residual.py`, `figure_camera_rig.py`.
