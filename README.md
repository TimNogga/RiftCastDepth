# DRIFTCast: Leveraging Depth for Concave-Aware 3D-Reconstruction

**Bachelor thesis by Tim Nogga** · Institute of Computer Science II – Visual Computing, University of Bonn
First Reviewer: Prof. Dr. Matthias B. Hullin · Second Reviewer: Prof. Dr. Reinhard Klein · June 25, 2026

📄 **Thesis PDF:** [DRIFTCast__Leveraging_Depth_for_Concave_Aware_3D_Reconstruction_Tim_Nogga.pdf](DRIFTCast__Leveraging_Depth_for_Concave_Aware_3D_Reconstruction_Tim_Nogga.pdf)

## Abstract

<p align="justify">
Immersive telepresence aims to reproduce remote people and scenes in real-time. RIFTCast achieves this through a template-free, real-time multi-view pipeline that reconstructs a scene from RGB silhouettes. This reliance on the visual hull imposes geometric limitations regarding concavities. In this thesis, DRIFTCast is introduced, which extends RIFTCast with depth information to recover these concavities. We integrate indirect Time-of-Flight cameras into a pre-existing capture stage, including their hardware triggering and synchronization. To fuse the depth data with the visual hull, we develop a multitude of strategies, comprising both a pre- and post-Marching-Cubes implementation. The first relies on Truncated Signed Distance Function fusion to reposition the isosurface before extraction, while combating artifacts created by Time-of-Flight cameras. The second operates after Marching Cubes, carving the concavities at the mesh level. Finally, we introduce a synthetic dataset to evaluate our methods against a ground truth. The real dataset is evaluated using image-based metrics as well as geometric measures, while the synthetic dataset relies on geometry alone. We show that depth fusion successfully recovers concave geometry while preserving rendering quality with minimal additional latency.
</p>

## Repository additions over RIFTCast

- **Depth integration** — indirect Time-of-Flight (Orbbec) cameras integrated into the VCI capture stage, including hardware triggering, synchronization, and depth loading in the dataset importer.
- **Pre-Marching-Cubes TSDF fusion** — TSDF voxel carving fused with the visual hull (with and without edge-gradient artifact suppression), implemented in the CUDA reconstruction backend ([`RIFTCast/src/riftcast/_C/external/torchhull/`](RIFTCast/src/riftcast/_C/external/torchhull/)).
- **Post-Marching-Cubes mesh cutter** — carving concavities at the mesh level after surface extraction.
- **Synthetic dataset & evaluation** — ground-truth dataset generation, Chamfer/F-score/image-based metrics, and sweep tooling in [`dataset/`](dataset/) and [`eval/`](eval/).
- **Experiment configurations** — the evaluated pipeline variants (no-depth baseline, TSDF ± edge gradient, post-MC cutter, cutter-only) in [`configs/`](configs/).

## Repository structure

| Folder | Contents |
|---|---|
| [`configs/`](configs/) | Experiment configurations for all evaluated pipeline variants (real & synthetic) |
| [`dataset/`](dataset/) | Dataset preparation: capture conversion, calibration, and synthetic dataset generation (Thesis Sec. 5.3, 5.5) |
| [`eval/`](eval/) | Evaluation scripts and sweeps: Chamfer/F-score, image-based metrics, camera-count ablations (Thesis Ch. 6) |
| [`figures/`](figures/) | Scripts that render the figures used in the thesis and presentation |
| [`results/`](results/) | Evaluation results: [`results/synthetic/`](results/synthetic/) — metrics, figures, and reconstructed meshes of the synthetic ablation (Sec. 6.4); [`results/real/`](results/real/) — leave-one-out image-based evaluation on real data (Sec. 6.3) |
| [`RIFTCast/`](RIFTCast/), [`src/`](src/) | The reconstruction framework: upstream RIFTCast extended with the depth-fusion CUDA backend |

## Acknowledgment

This work builds on [RIFTCast](https://github.com/vc-bonn/RIFTCast) (Zingsheim et al., ACM Multimedia 2025, [project page](https://cg.cs.uni-bonn.de/publication/zingsheim-2025-riftcast)) by the Visual Computing Group of the University of Bonn. See the upstream repository for the original framework, build instructions, and benchmark.
