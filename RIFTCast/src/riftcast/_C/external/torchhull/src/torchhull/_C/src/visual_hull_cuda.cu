#include <array>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <iostream>

#include <ATen/Dispatch.h>
#include <ATen/cuda/ApplyGridUtils.cuh>
#include <ATen/cuda/ThrustAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/macros/Macros.h>
#include <cub/device/device_select.cuh>
#include <glm/common.hpp>
#include <glm/gtc/epsilon.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <torch/types.h>

#include "marching_cubes_table.h"
#include <torchhull/cuda_utils.h>
#include <torchhull/image_utils.h>
#include <torchhull/marching_cubes_utils.h>
#include <torchhull/math.h>
#include <torchhull/ravelled_sparse_tensor.h>
#include <torchhull/stdgpu_allocator.h>

namespace torchhull
{

// =============================================================================
// EXTENDED DEBUG COUNTER MAPPING (depth_debug array) - SIZE 20
// [0] : CLASSIFY - Carved by Depth (Empty Air inside concavity)
// [1] : CLASSIFY - Invalid Pixel bounds
// [2] : CLASSIFY - Invalid Sensor Range (e.g. 0 depth)
// [3] : CLASSIFY - Kept (Background leak / Ray shot past object)
// [4] : CLASSIFY - Kept (Inside structural margin)
// [5] : CLASSIFY - Valid lookups attempted
// [6] : TSDF - Valid Fusions (Voxel seen by depth)
// [7] : TSDF - Unseen Fallback (Voxel hidden, forced to solid)
// [8] : CLASSIFY - Shielded by Edge Gradient
// =============================================================================

template <typename ImageT, typename IntegralT>
__global__ void
check_integral_image(const torch::PackedTensorAccessor64<ImageT, 4, torch::RestrictPtrTraits> image,
                     const torch::PackedTensorAccessor64<IntegralT, 4, torch::RestrictPtrTraits> integral_image,
                     torch::PackedTensorAccessor64<bool, 4, torch::RestrictPtrTraits> valid)
{
    const auto sizes = glm::i64vec3{ image.size(2), image.size(1), image.size(0) };
    const auto channels = image.size(3);
    const auto N = numel(sizes);

    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for (auto tid = id; tid < N; tid += num_threads)
    {
        auto p = unravel_index(tid, sizes);

        for (auto c = int64_t{ 0 }; c < channels; ++c)
        {
            IntegralT integral_00 = sample_zeros_padding(integral_image, p.y - 1, p.x - 1, p.z, c);
            IntegralT integral_10 = sample_zeros_padding(integral_image, p.y, p.x - 1, p.z, c);
            IntegralT integral_01 = sample_zeros_padding(integral_image, p.y - 1, p.x, p.z, c);
            IntegralT integral_11 = sample_zeros_padding(integral_image, p.y, p.x, p.z, c);

            auto image_value = static_cast<IntegralT>(image[p.z][p.y][p.x][c]);
            auto image_value_integral = integral_11 + integral_00 - integral_10 - integral_01;

            if constexpr (std::is_floating_point_v<IntegralT>)
            {
                const auto epsilon = IntegralT{ 1e-1 };
                valid[p.z][p.y][p.x][c] =
                        glm::epsilonEqual(static_cast<IntegralT>(image_value), image_value_integral, epsilon);
            }
            else
            {
                valid[p.z][p.y][p.x][c] = (image_value == image_value_integral);
            }
        }
    }
}

torch::Tensor
integral_image(const torch::Tensor& self, c10::ScalarType dtype)
{
    TORCH_CHECK_EQ(self.dim(), 4); 

    auto result = self;
    result = torch::cumsum(result, 1, dtype);
    result = torch::cumsum(result, 2, dtype);

    TORCH_CHECK_EQ(result.dim(), 4); 

    return result;
}

bool
is_integral_image_valid(const torch::Tensor& image, const torch::Tensor& integral_image)
{
    TORCH_CHECK_EQ(image.device(), integral_image.device());
    TORCH_CHECK_EQ(image.sizes(), integral_image.sizes());

    at::cuda::CUDAGuard device_guard{ image.device() };
    const auto stream = at::cuda::getCurrentCUDAStream();

    const auto dtype_bool = torch::TensorOptions{}.dtype(torch::kBool).device(image.device());

    const int threads_per_block = 128;
    dim3 grid;
    at::cuda::getApplyGrid(numel({ image.size(2), image.size(1), image.size(0) }),
                           grid,
                           image.device().index(),
                           threads_per_block);
    dim3 threads = at::cuda::getApplyBlock(threads_per_block);

    auto valid = torch::empty(image.sizes(), dtype_bool);

    AT_DISPATCH_ALL_TYPES(
            image.scalar_type(),
            "is_integral_image_valid",
            [&]()
            {
                auto image_ = image.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>();

                AT_DISPATCH_ALL_TYPES(
                        integral_image.scalar_type(),
                        "is_integral_image_valid",
                        [&]()
                        {
                            auto integral_image_ =
                                    integral_image.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>();

                            check_integral_image<<<grid, threads, 0, stream>>>(
                                    image_,
                                    integral_image_,
                                    valid.packed_accessor64<bool, 4, torch::RestrictPtrTraits>());
                            AT_CUDA_CHECK(cudaGetLastError());
                            AT_CUDA_CHECK(cudaStreamSynchronize(stream));
                        });
            });

    auto valid_count = valid.sum(torch::kInt64).cpu().item<int64_t>();

    return valid_count == valid.numel();
}

template <typename ValueT>
inline C10_DEVICE glm::vec4
bmm_4x4_transforms(const glm::vec3& v,
                   const torch::PackedTensorAccessor64<ValueT, 3, torch::RestrictPtrTraits>& transforms,
                   const int batch)
{
#define A(batch, i, j) transforms[batch][i][j]

    return glm::vec4{ fmaf(A(batch, 0, 0), v.x, fmaf(A(batch, 0, 1), v.y, fmaf(A(batch, 0, 2), v.z, A(batch, 0, 3)))),
                      fmaf(A(batch, 1, 0), v.x, fmaf(A(batch, 1, 1), v.y, fmaf(A(batch, 1, 2), v.z, A(batch, 1, 3)))),
                      fmaf(A(batch, 2, 0), v.x, fmaf(A(batch, 2, 1), v.y, fmaf(A(batch, 2, 2), v.z, A(batch, 2, 3)))),
                      fmaf(A(batch, 3, 0), v.x, fmaf(A(batch, 3, 1), v.y, fmaf(A(batch, 3, 2), v.z, A(batch, 3, 3)))) };

#undef A
}

namespace
{
constexpr float kDepthMinMeters = 0.1f;
constexpr float kDepthMaxMeters = 3.95f;
constexpr float kBackgroundDepthDeltaMeters = 0.65f;
constexpr float kMaxPositiveSdfForFusionMeters = 0.70f;
constexpr float kDepthSurfaceSlackMeters = 0.00f;
// Carve only on DEEP free space (voxel >= 3cm in front of the measured surface). Genuine concavity
// interiors are deep; the shallow false-positives produced by GRAZING LIMB rays (steep per-pixel
// depth gradient near a camera's silhouette edge) are only ~1-2cm and must NOT trigger carving,
// otherwise more cameras over-carve the smooth surface. See depth-fusion-veto-anti-monotonic.
//
// PERFECT-DATA TEST (2026-06-03): tried 0 -> "any positive SDF carves". It OVER-CARVED and got
// strongly worse with more cameras (chamfer 26->68mm, N=1..12) because even with perfect synthetic
// depth the silhouette EDGE gives a shallow false-positive SDF (a depth pixel straddling object+
// background interpolates to an edge depth), which carves the convex limb. So the band is NOT
// noise-rejection, it is geometric grazing-limb rejection and must stay. Reverted to 0.03 (still NO
// cross-camera veto: carve iff any single camera sees DEEP free space > this; no camera blocks another).
constexpr float kPositiveSdfThresholdMeters = 0.01f;
constexpr float kTruncMarginMeters = 0.05f;  // TSDF truncation tau; also the remap scale in normalize
constexpr float kVetoSaturationCount = 2.0f; // # head-on surface votes at which protection saturates (soft veto)
constexpr float kGrazingAirMinMeters = 0.01f; // grazing observers: abstain on the surface band only (=kPositiveSdf),
                                              // carve on any free space past it (deep-air gate at 0.04 regressed)
// LIGHTWEIGHT GRAZING GATE (test): a camera abstains from carving where it views the surface too
// obliquely (its depth there is unreliable -> false free-space -> over-carves the convex limb).
// We use the central-difference per-pixel depth gradient as an incidence proxy: g = |grad z| in
// metres/pixel, with cos(incidence) ~ 1/sqrt(1+(g*f/z)^2). At this rig (z~2.2m, f~2460px) the
// thresholds map roughly: 0.0025 ~ 70deg, 0.005 ~ 80deg, 0.010 ~ 85deg. The existing abs-jump edge
// gate only fires ~88deg, missing the 70-88deg band that does the damage. 0 disables this gate.
constexpr float kGrazingGradThreshMetersPerPixel = 0.0025f;  // ~70deg; gates the SURFACE vote only (see veto)
constexpr float kSolidSdfThresholdMeters = -0.08f;
constexpr float kTruncMarginOctree = 0.05f;  // solid threshold used in octree depth carving

// CURLESS & LEVOY (1996) incidence weighting. Their cumulative SDF is D(x)=Sum(w_i d_i)/Sum(w_i) with
// per-sample weight w_i chosen to fall off at grazing incidence (greater range uncertainty). We use
// w_i = cos(theta)^p, where theta is the angle between the sensor ray and the surface, estimated from
// the per-pixel depth gradient: tan(theta) ~ (f/z)*|grad z| (same proxy as is_depth_grazing, but
// CONTINUOUS instead of a hard threshold). f (px) is recovered rotation-invariantly from the
// world->clip transform: fx = 0.5*W*||first-row spatial part||. Frontal -> w~1, grazing limb -> w~0,
// so the geometrically-false shallow free-space at the convex limb is suppressed smoothly rather than
// by a binary gate. kUseIncidenceWeight=true restores the old uniform (w=1) average + binary grazing gate.
constexpr bool  kUseIncidenceWeight    = true;
constexpr float kIncidenceWeightPower  = 1.0f;   // Curless-Levoy use cos(theta)^1; >1 sharpens the falloff
constexpr float kMinIncidenceWeight    = 0.05f;  // ~87deg: below this the sample is near-tangent garbage -> abstain

enum class DepthFusionPresetMode : int
{
    NoDepth         = 0,
    RealData        = 1,
    SyntheticData   = 2,
    RealDataNoEdge  = 3,
    SyntheticNoEdge = 4,
};

inline DepthFusionPresetMode
sanitize_depth_fusion_preset(const int preset_raw)
{
    if(preset_raw == static_cast<int>(DepthFusionPresetMode::NoDepth))         return DepthFusionPresetMode::NoDepth;
    if(preset_raw == static_cast<int>(DepthFusionPresetMode::SyntheticData))   return DepthFusionPresetMode::SyntheticData;
    if(preset_raw == static_cast<int>(DepthFusionPresetMode::RealDataNoEdge))  return DepthFusionPresetMode::RealDataNoEdge;
    if(preset_raw == static_cast<int>(DepthFusionPresetMode::SyntheticNoEdge)) return DepthFusionPresetMode::SyntheticNoEdge;
    return DepthFusionPresetMode::RealData;
}

inline bool
depth_fusion_enabled(const DepthFusionPresetMode preset)
{
    return preset != DepthFusionPresetMode::NoDepth;
}

inline bool
use_synthetic_depth_tuning(const DepthFusionPresetMode preset)
{
    return preset == DepthFusionPresetMode::SyntheticData
        || preset == DepthFusionPresetMode::SyntheticNoEdge;
}

inline bool
use_depth_fallback_neighborhood(const DepthFusionPresetMode preset)
{
    return use_synthetic_depth_tuning(preset) ? false : true;
}

inline int
depth_edge_radius(const DepthFusionPresetMode preset)
{
    // NoEdge variants: radius=0 → loop body is skipped → no shielding
    if (preset == DepthFusionPresetMode::RealDataNoEdge
     || preset == DepthFusionPresetMode::SyntheticNoEdge)
        return 0;
    return use_synthetic_depth_tuning(preset) ? 2 : 10;
}

inline float
depth_edge_threshold(const DepthFusionPresetMode preset)
{
    return use_synthetic_depth_tuning(preset) ? 0.06f  : 0.04f;
}

inline float
concavity_forward_offset_meters(const DepthFusionPresetMode preset)
{
    return use_synthetic_depth_tuning(preset) ? 0.0f  : 0.24f;
}

inline float
carve_boost(const DepthFusionPresetMode preset)
{
    return use_synthetic_depth_tuning(preset) ? 0.30f : 0.90f;
}
}

template <typename ValueT>
inline C10_DEVICE float
sample_depth_with_fallback(const torch::PackedTensorAccessor64<ValueT, 4, torch::RestrictPtrTraits> depths,
                           const float py,
                           const float px,
                           const int batch,
                           const bool use_depth_fallback)
{
    const auto px_i = static_cast<int64_t>(roundf(px));
    const auto py_i = static_cast<int64_t>(roundf(py));
    
    float z_sensor = sample_zeros_padding(depths, py_i, px_i, batch, 0);
    if (z_sensor > kDepthMinMeters && z_sensor <= kDepthMaxMeters)
    {
        return z_sensor;
    }

    if (!use_depth_fallback)
    {
        return 0.0f;
    }

    float z_best = 0.0f;
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            const float z_neighbor = sample_zeros_padding(depths, py_i + dy, px_i + dx, batch, 0);
            if (z_neighbor > z_best && z_neighbor <= kDepthMaxMeters)
            {
                z_best = z_neighbor;
            }
        }
    }

    return z_best;
}

// BILINEAR depth sampling at the exact sub-pixel projection. Nearest-neighbour (round) sampling reads
// the depth of a slightly different surface point than the voxel projects to; on an oblique/curved
// surface that mismatch reads as false free-space and CARVES TRUE SURFACE (the dominant error with
// perfect synthetic depth, and it compounds with camera count). Interpolating at the true sub-pixel
// location removes that to first order. Only interpolate when all four taps are valid (interior);
// at depth edges/invalids fall back to nearest, and the edge gate already excludes those pixels.
template <typename ValueT>
inline C10_DEVICE float
sample_depth_bilinear(const torch::PackedTensorAccessor64<ValueT, 4, torch::RestrictPtrTraits> depths,
                      const float py,
                      const float px,
                      const int batch)
{
    const int64_t x0 = static_cast<int64_t>(floorf(px));
    const int64_t y0 = static_cast<int64_t>(floorf(py));
    const float ax = px - static_cast<float>(x0);
    const float ay = py - static_cast<float>(y0);

    float z00 = sample_zeros_padding(depths, y0,     x0,     batch, 0);
    float z10 = sample_zeros_padding(depths, y0,     x0 + 1, batch, 0);
    float z01 = sample_zeros_padding(depths, y0 + 1, x0,     batch, 0);
    float z11 = sample_zeros_padding(depths, y0 + 1, x0 + 1, batch, 0);

    auto valid = [](float z) { return z > kDepthMinMeters && z <= kDepthMaxMeters; };
    if (valid(z00) && valid(z10) && valid(z01) && valid(z11))
    {
        float z0 = z00 * (1.0f - ax) + z10 * ax;
        float z1 = z01 * (1.0f - ax) + z11 * ax;
        return z0 * (1.0f - ay) + z1 * ay;
    }
    // fallback: nearest tap (edge gate normally rejects these pixels anyway)
    return sample_zeros_padding(depths, static_cast<int64_t>(roundf(py)), static_cast<int64_t>(roundf(px)), batch, 0);
}

template <typename ValueT>
inline C10_DEVICE bool
is_depth_edge_strong(const torch::PackedTensorAccessor64<ValueT, 4, torch::RestrictPtrTraits> depths,
                     const float py,
                     const float px,
                     const int batch,
                     const int edge_radius,
                     const float edge_threshold)
{
    const int64_t px_i = static_cast<int64_t>(roundf(px));
    const int64_t py_i = static_cast<int64_t>(roundf(py));

    float z_center = sample_zeros_padding(depths, py_i, px_i, batch, 0);
    if (z_center <= kDepthMinMeters || z_center > kDepthMaxMeters) return true;

    const int radius = edge_radius;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx == 0 && dy == 0) continue;
            
            float z_neighbor = sample_zeros_padding(depths, py_i + dy, px_i + dx, batch, 0);
            
            if (z_neighbor <= kDepthMinMeters || z_neighbor > kDepthMaxMeters) {
                return true; 
            }
            
            if (fabsf(z_neighbor - z_center) > edge_threshold) {
                return true;
            }
        }
    }
    return false;
}

// Per-camera GRAZING (incidence) gate: true if the surface is viewed too obliquely here, judged by
// the central-difference per-pixel depth gradient (an incidence-angle proxy). Unlike the abs-jump
// edge gate above, this catches the gentle-but-foreshortened limb band that over-carves the convex
// surface. grad_thresh <= 0 disables. Invalid centre/neighbours -> treat as grazing/edge (true).
template <typename ValueT>
inline C10_DEVICE bool
is_depth_grazing(const torch::PackedTensorAccessor64<ValueT, 4, torch::RestrictPtrTraits> depths,
                 const float py,
                 const float px,
                 const int batch,
                 const float grad_thresh)
{
    if (grad_thresh <= 0.0f) return false;
    const int64_t px_i = static_cast<int64_t>(roundf(px));
    const int64_t py_i = static_cast<int64_t>(roundf(py));

    float zl = sample_zeros_padding(depths, py_i, px_i - 1, batch, 0);
    float zr = sample_zeros_padding(depths, py_i, px_i + 1, batch, 0);
    float zu = sample_zeros_padding(depths, py_i - 1, px_i, batch, 0);
    float zd = sample_zeros_padding(depths, py_i + 1, px_i, batch, 0);
    if (zl <= kDepthMinMeters || zl > kDepthMaxMeters ||
        zr <= kDepthMinMeters || zr > kDepthMaxMeters ||
        zu <= kDepthMinMeters || zu > kDepthMaxMeters ||
        zd <= kDepthMinMeters || zd > kDepthMaxMeters) return true;

    float gx = 0.5f * (zr - zl);
    float gy = 0.5f * (zd - zu);
    return (gx * gx + gy * gy) > (grad_thresh * grad_thresh);
}

// CURLESS-LEVOY incidence weight w_i = cos(theta)^p for the cumulative weighted SDF. theta = angle
// between sensor ray and surface, from the per-pixel depth gradient: tan(theta) ~ fz*|grad z|, where
// fz = focal_px / depth_m. Returns 0 at an invalid/edge pixel (no usable sample) and at near-tangent
// incidence (below kMinIncidenceWeight); the caller treats 0 as "abstain". Frontal -> ~1.
template <typename ValueT>
inline C10_DEVICE float
depth_incidence_weight(const torch::PackedTensorAccessor64<ValueT, 4, torch::RestrictPtrTraits> depths,
                       const float py,
                       const float px,
                       const int batch,
                       const float fz)
{
    const int64_t px_i = static_cast<int64_t>(roundf(px));
    const int64_t py_i = static_cast<int64_t>(roundf(py));

    float zl = sample_zeros_padding(depths, py_i, px_i - 1, batch, 0);
    float zr = sample_zeros_padding(depths, py_i, px_i + 1, batch, 0);
    float zu = sample_zeros_padding(depths, py_i - 1, px_i, batch, 0);
    float zd = sample_zeros_padding(depths, py_i + 1, px_i, batch, 0);
    if (zl <= kDepthMinMeters || zl > kDepthMaxMeters ||
        zr <= kDepthMinMeters || zr > kDepthMaxMeters ||
        zu <= kDepthMinMeters || zu > kDepthMaxMeters ||
        zd <= kDepthMinMeters || zd > kDepthMaxMeters) return 0.0f;

    float gx = 0.5f * (zr - zl);
    float gy = 0.5f * (zd - zu);
    float tan2 = fz * fz * (gx * gx + gy * gy);   // tan^2(theta)
    float cos_theta = rsqrtf(1.0f + tan2);        // 1/sqrt(1+tan^2) = cos(theta)
    float w = (kIncidenceWeightPower == 1.0f) ? cos_theta : powf(cos_theta, kIncidenceWeightPower);
    return (w < kMinIncidenceWeight) ? 0.0f : w;
}

// Focal length in pixels recovered from the world->clip transform (rotation-invariant): the spatial
// part of the matrix's first row has norm |P00|, and fx = 0.5*W*|P00| for a centred GL frustum.
template <typename TransformT>
inline C10_DEVICE float
focal_px_from_transform(const torch::PackedTensorAccessor64<TransformT, 3, torch::RestrictPtrTraits>& transforms,
                        const int batch,
                        const int64_t W)
{
    float a00 = transforms[batch][0][0];
    float a01 = transforms[batch][0][1];
    float a02 = transforms[batch][0][2];
    return 0.5f * static_cast<float>(W) * sqrtf(a00 * a00 + a01 * a01 + a02 * a02);
}

template <typename Pair>
struct select1st
{
    inline C10_DEVICE typename Pair::first_type
    operator()(const Pair& pair) const
    {
        return pair.first;
    }
};

template <typename TransformT>
__global__ void
classify_children_full(const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> candidates,
                       const torch::PackedTensorAccessor64<float, 4, torch::RestrictPtrTraits> integral_masks,
                       const torch::PackedTensorAccessor64<float, 4, torch::RestrictPtrTraits> depths,
                       const torch::PackedTensorAccessor64<TransformT, 3, torch::RestrictPtrTraits> transforms,
                       const bool transforms_in_opengl,
                       const glm::i64vec3 resolution,
                       const glm::i64vec3 resolution_children,
                       const glm::vec3 cube_corner_bfl,
                       const float cube_length,
                       const int current_level,
                       const int max_level,
                       const bool use_tsdf,
                       const float concavity_forward_offset_meters,
                       const int edge_radius,
                       const float edge_threshold,
                       torch::PackedTensorAccessor64<uint8_t, 1, torch::RestrictPtrTraits> occupied_voxel,
                       int64_t* depth_debug)
{
    const auto N = occupied_voxel.size(0);
    const auto H = integral_masks.size(1);
    const auto W = integral_masks.size(2);
    
    // Octree per-voxel depth carving is DISABLED. The TSDF scalar field already opens the
    // aperture cleanly; the octree carve only helped with poorly-placed (grazing) cameras and,
    // off-axis, fabricated false-air that holed the side walls. With FRONTAL depth cameras
    // (a narrow cone around the opening axis, no grazing) the TSDF alone yields a clean open box
    // with solid walls. Re-enable only if you intentionally want the experimental octree carve.
    constexpr bool kEnableOctreeDepthCarve = false;
    const bool apply_depth_carving = kEnableOctreeDepthCarve && use_tsdf && (current_level == max_level - 1);

    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);

    for (auto tid = id; tid < N; tid += num_threads)
    {
        auto candidate_id = tid / 8;
        auto child_id = tid % 8;
        auto g = unravel_index(candidates[candidate_id], resolution);
        auto g_child = cube_vertex(int64_t{ 2 } * g, child_id);

        auto is_empty = false;
        auto should_refine = false;

        for (auto batch = int64_t{ 0 }; batch < integral_masks.size(0); ++batch) {
            float bb_min_x = FLT_MAX, bb_min_y = FLT_MAX;
            float bb_max_x = -FLT_MAX, bb_max_y = -FLT_MAX;
            for (auto i = 0; i < 8; ++i) {
                auto v = cube_vertex(g_child, i);
                auto v_world = glm::vec3{
                    cube_corner_bfl.x + static_cast<float>(v.x) / static_cast<float>(resolution_children.x) * cube_length,
                    cube_corner_bfl.y + static_cast<float>(v.y) / static_cast<float>(resolution_children.y) * cube_length,
                    cube_corner_bfl.z + static_cast<float>(v.z) / static_cast<float>(resolution_children.z) * cube_length
                };
                auto v_camera = bmm_4x4_transforms(v_world, transforms, batch);
                auto v_pixel = transforms_in_opengl ?
                    glm::vec2{unnormalize_ndc_false(v_camera.x/v_camera.w, W), unnormalize_ndc_false(v_camera.y/v_camera.w, H)} :
                    glm::vec2{align_cv_false(v_camera.x/v_camera.z), align_cv_false(v_camera.y/v_camera.z)};

                bb_min_x = fminf(bb_min_x, v_pixel.x); bb_min_y = fminf(bb_min_y, v_pixel.y);
                bb_max_x = fmaxf(bb_max_x, v_pixel.x); bb_max_y = fmaxf(bb_max_y, v_pixel.y);
            }

            int64_t min_x = static_cast<int64_t>(roundf(bb_min_x - 0.5f));
            int64_t min_y = static_cast<int64_t>(roundf(bb_min_y - 0.5f));
            int64_t max_x = static_cast<int64_t>(roundf(bb_max_x + 0.5f));
            int64_t max_y = static_cast<int64_t>(roundf(bb_max_y + 0.5f));

            min_x = min_x < 0 ? 0 : (min_x > W - 1 ? W - 1 : min_x);
            min_y = min_y < 0 ? 0 : (min_y > H - 1 ? H - 1 : min_y);
            max_x = max_x < 0 ? 0 : (max_x > W - 1 ? W - 1 : max_x);
            max_y = max_y < 0 ? 0 : (max_y > H - 1 ? H - 1 : max_y);

            auto integral_mask_00 = sample_zeros_padding(integral_masks, min_y - 1, min_x - 1, batch, 0);
            auto integral_mask_10 = sample_zeros_padding(integral_masks, max_y, min_x - 1, batch, 0);
            auto integral_mask_01 = sample_zeros_padding(integral_masks, min_y - 1, max_x, batch, 0);
            auto integral_mask_11 = sample_zeros_padding(integral_masks, max_y, max_x, batch, 0);

            auto int_bb = integral_mask_11 + integral_mask_00 - integral_mask_10 - integral_mask_01;

            if (int_bb <= 0.4f) { is_empty = true; break; } else { should_refine = true; }
        }

        if (is_empty) {
            atomicAdd(reinterpret_cast<unsigned long long*>(&depth_debug[9]), 1ULL);
        } else {
            atomicAdd(reinterpret_cast<unsigned long long*>(&depth_debug[10]), 1ULL);
        }

        // 3D DEPTH CARVING: at the finest octree level, classify interior voxels as empty
        // using majority vote across depth cameras. This creates boundary voxels at the
        // inner walls of concavities (e.g. the back wall of a hollow box).
        bool depth_carved = false;
        if (apply_depth_carving && !is_empty && should_refine) {
            // Compute voxel centre in world space
            auto vox_world = glm::vec3{
                cube_corner_bfl.x + (static_cast<float>(g_child.x) + 0.5f) / static_cast<float>(resolution_children.x) * cube_length,
                cube_corner_bfl.y + (static_cast<float>(g_child.y) + 0.5f) / static_cast<float>(resolution_children.y) * cube_length,
                cube_corner_bfl.z + (static_cast<float>(g_child.z) + 0.5f) / static_cast<float>(resolution_children.z) * cube_length
            };

            // Space-carving votes across depth cameras:
            //   air     = voxel sits clearly in FREE SPACE in front of the measured surface
            //   surface = voxel lies ON the measured surface (within trunc band) -> real geometry
            //   behind  = voxel is BEHIND the surface (occluded from this view)  -> unknown, ignore
            //
            // STRICT space-carving veto. Each depth camera classifies the voxel:
            //   air    : sdf > kShieldMarginM            -> DEEP free space (seen through the opening)
            //   surface: -kTruncMarginOctree <= sdf <= kShieldMarginM
            //            -> the voxel lies ON / just in front of a measured surface => it is WALL.
            //   behind : sdf < -kTruncMarginOctree       -> occluded, unknown -> ignore.
            // A wall voxel is, by definition, seen as a surface by the camera facing it, so a
            // single reliable surface vote VETOES carving. We only carve voxels that NO camera
            // sees as surface and that several cameras see as deep free space. The edge shield
            // first discards unreliable views at depth discontinuities (rims / silhouette /
            // background), which is what previously gave the aperture cap false surface votes
            // and walls false air votes. Result: opening opens, walls cannot be holed.
            // Tolerance budget: kCarveFrontMarginM is the max slack — a voxel within this
            // distance in front of a surface is treated as wall, so geometry is off by <= this.
            // NOTE on the tradeoff (this depth-camera layout, all on the +Z opening hemisphere):
            //  - side walls are only seen edge-on, so a SMALL kShieldMarginM lets foreshortened
            //    false-air carve holes through them;
            //  - a STRICT surface veto closes the aperture (grazing wall views vote false-surface).
            // The robust compromise is a generous shield (kShieldMarginM) + majority vote: the real
            // opening is seen as DEEP air by many cameras and still opens, while the shallow
            // false-air on walls stays below the shield. Cost: geometry near walls can be off by
            // up to (kShieldMarginM - forward_offset). Tightening this needs depth cameras that see
            // the side walls head-on, not just more of them on the opening hemisphere.
            // Side walls of an open box can only be seen at an angle through the single opening,
            // so their foreshortened depth leaks false-air. A protection margin below ~this floor
            // (tens of mm) holes them; above it they stay solid. kShieldMarginM sets that floor.
            constexpr int   kMinAirVotes   = 2;
            constexpr float kShieldMarginM = 0.10f; // hole-free; geometry near walls off by <= ~80 mm
            const     float kCarveFrontMarginM = kShieldMarginM - concavity_forward_offset_meters;
            int air_count = 0, surface_count = 0;
            for (auto batch = int64_t{0}; batch < depths.size(0); ++batch) {
                auto v_cam = bmm_4x4_transforms(vox_world, transforms, batch);
                // Metric depth: perspective-divide component (.w) for OpenGL clip space,
                // camera-space z for OpenCV. Matches accumulate_tsdf_full exactly.
                float metric_z = transforms_in_opengl ? fabsf(v_cam.w) : fabsf(v_cam.z);
                if (metric_z < kDepthMinMeters) continue;

                auto v_px = transforms_in_opengl ?
                    glm::vec2{unnormalize_ndc_false(v_cam.x/v_cam.w, (int64_t)depths.size(2)),
                              unnormalize_ndc_false(v_cam.y/v_cam.w, (int64_t)depths.size(1))} :
                    glm::vec2{align_cv_false(v_cam.x/v_cam.z), align_cv_false(v_cam.y/v_cam.z)};

                int64_t px = static_cast<int64_t>(roundf(v_px.x));
                int64_t py = static_cast<int64_t>(roundf(v_px.y));
                if (px < 1 || px >= (int64_t)depths.size(2) - 1 ||
                    py < 1 || py >= (int64_t)depths.size(1) - 1) continue;

                float z_sensor = depths[batch][py][px][0];
                if (z_sensor <= kDepthMinMeters || z_sensor > kDepthMaxMeters) continue;

                // EDGE / GRAZING SHIELD: skip views where the voxel projects onto a depth
                // discontinuity or steep gradient (box rim, silhouette boundary, background, or a
                // wall seen at grazing angle whose depth jumps to the far surface). This is the
                // main defense against a camera "deleting" a wall it only skims; it works as long
                // as some OTHER camera sees that wall reasonably head-on (deliberate placement).
                const int kCarveEdgeRadius = (edge_radius > 0) ? edge_radius : 2;
                if (is_depth_edge_strong(depths, v_px.y, v_px.x, static_cast<int>(batch), kCarveEdgeRadius, edge_threshold)) {
                    continue;
                }

                // SDF: positive = free air in front of surface, ~0 = on surface, negative = behind.
                float sdf = z_sensor - (metric_z - concavity_forward_offset_meters);
                if (sdf > kShieldMarginM)                  air_count++;      // deep free space
                else if (sdf >= -kTruncMarginOctree)       surface_count++;  // ON a surface -> WALL
                // else: occluded/behind -> ignore
            }
            // Majority vote: open the aperture (deep air dominates) while shallow false-air on
            // under-observed walls stays below the shield and cannot outvote the surface evidence.
            (void)kCarveFrontMarginM;
            if (air_count >= kMinAirVotes && air_count > surface_count) depth_carved = true;
            if (apply_depth_carving)
                atomicAdd(reinterpret_cast<unsigned long long*>(&depth_debug[5]), 1ULL); // candidates evaluated
            if (depth_carved)
                atomicAdd(reinterpret_cast<unsigned long long*>(&depth_debug[19]), 1ULL); // carved
        }

        occupied_voxel[tid] = (should_refine && !is_empty && !depth_carved);
    }
}

template <typename TransformT>
__global__ void
classify_children_partial(const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> candidates,
                          const torch::PackedTensorAccessor64<float, 4, torch::RestrictPtrTraits> integral_masks,
                          const torch::PackedTensorAccessor64<float, 4, torch::RestrictPtrTraits> depths,
                          const torch::PackedTensorAccessor64<TransformT, 3, torch::RestrictPtrTraits> transforms,
                          const bool transforms_in_opengl,
                          const glm::i64vec3 resolution,
                          const glm::i64vec3 resolution_children,
                          const glm::vec3 cube_corner_bfl,
                          const float cube_length,
                          const int current_level, 
                          const int max_level,
                          const bool use_tsdf,
                          torch::PackedTensorAccessor64<uint8_t, 1, torch::RestrictPtrTraits> occupied_voxel,
                          int64_t* depth_debug)
{
    const auto N = occupied_voxel.size(0);
    const auto H = integral_masks.size(1);
    const auto W = integral_masks.size(2);
    
    // FIX: Disabled octree depth carving so it doesn't delete the empty air we need for TSDF
    bool apply_depth_carving = false; // use_tsdf && (current_level == max_level - 1);
    const bool last_children = (current_level == max_level - 1);

    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    
    for (auto tid = id; tid < N; tid += num_threads)
    {
        auto candidate_id = tid / 8;
        auto child_id = tid % 8;
        auto g = unravel_index(candidates[candidate_id], resolution);
        auto g_child = cube_vertex(int64_t{ 2 } * g, child_id);

        auto is_empty = false;
        auto should_refine = false;
        auto fully_inside_one_frame = false;
        
        for (auto batch = int64_t{ 0 }; batch < integral_masks.size(0); ++batch) {
            float bb_min_x = FLT_MAX, bb_min_y = FLT_MAX; 
            float bb_max_x = -FLT_MAX, bb_max_y = -FLT_MAX;
            auto fully_inside = true;
            
            for (auto i = 0; i < 8; ++i) {
                auto v = cube_vertex(g_child, i);
                auto v_world = glm::vec3{ cube_corner_bfl.x + static_cast<float>(v.x) / static_cast<float>(resolution_children.x) * cube_length,
                                          cube_corner_bfl.y + static_cast<float>(v.y) / static_cast<float>(resolution_children.y) * cube_length,
                                          cube_corner_bfl.z + static_cast<float>(v.z) / static_cast<float>(resolution_children.z) * cube_length };
                auto v_camera = bmm_4x4_transforms(v_world, transforms, batch);
                auto v_pixel = transforms_in_opengl ? 
                    glm::vec2{unnormalize_ndc_false(v_camera.x/v_camera.w, W), unnormalize_ndc_false(v_camera.y/v_camera.w, H)} :
                    glm::vec2{align_cv_false(v_camera.x/v_camera.z), align_cv_false(v_camera.y/v_camera.z)};

                bb_min_x = fminf(bb_min_x, v_pixel.x); bb_min_y = fminf(bb_min_y, v_pixel.y);
                bb_max_x = fmaxf(bb_max_x, v_pixel.x); bb_max_y = fmaxf(bb_max_y, v_pixel.y);

                auto v_px_r = glm::i64vec2{ roundf(v_pixel.x), roundf(v_pixel.y) };
                if (!in_image(v_px_r.y, v_px_r.x, H, W, 1)) fully_inside = false;
            }

            int64_t min_x = static_cast<int64_t>(roundf(bb_min_x - 0.5f));
            int64_t min_y = static_cast<int64_t>(roundf(bb_min_y - 0.5f));
            int64_t max_x = static_cast<int64_t>(roundf(bb_max_x + 0.5f));
            int64_t max_y = static_cast<int64_t>(roundf(bb_max_y + 0.5f));

            min_x = min_x < 0 ? 0 : (min_x > W - 1 ? W - 1 : min_x);
            min_y = min_y < 0 ? 0 : (min_y > H - 1 ? H - 1 : min_y);
            max_x = max_x < 0 ? 0 : (max_x > W - 1 ? W - 1 : max_x);
            max_y = max_y < 0 ? 0 : (max_y > H - 1 ? H - 1 : max_y);

            auto area_bb = (max_y - min_y + 1) * (max_x - min_x + 1);
            auto full_area_bb = (roundf(bb_max_y+0.5f) - roundf(bb_min_y-0.5f) + 1) * (roundf(bb_max_x+0.5f) - roundf(bb_min_x-0.5f) + 1);

            auto integral_mask_00 = sample_zeros_padding(integral_masks, min_y - 1, min_x - 1, batch, 0);
            auto integral_mask_10 = sample_zeros_padding(integral_masks, max_y, min_x - 1, batch, 0);
            auto integral_mask_01 = sample_zeros_padding(integral_masks, min_y - 1, max_x, batch, 0);
            auto integral_mask_11 = sample_zeros_padding(integral_masks, max_y, max_x, batch, 0);

            auto int_bb = integral_mask_11 + integral_mask_00 - integral_mask_10 - integral_mask_01;

            if (int_bb <= 0.4f && area_bb == full_area_bb) is_empty = true;
            else if (max_y - min_y > 1 && max_x - min_x > 1) should_refine = true;

            fully_inside_one_frame |= fully_inside;
            if (is_empty) break; 
        } 

        if (is_empty) {
            atomicAdd(reinterpret_cast<unsigned long long*>(&depth_debug[9]), 1ULL);
        } else {
            atomicAdd(reinterpret_cast<unsigned long long*>(&depth_debug[10]), 1ULL);
        }

        // 3D DEPTH CARVING is disabled here.
        // It remains solely handled in the TSDF fusion kernels below.

        occupied_voxel[tid] = (should_refine && !is_empty && (!last_children || fully_inside_one_frame));
    }
}

template <typename MaskT, typename TransformT>
__global__ void
accumulate_tsdf_full(const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> sparse_indices,
                     const int64_t N,
                     const torch::PackedTensorAccessor64<MaskT, 4, torch::RestrictPtrTraits> masks,
                     const torch::PackedTensorAccessor64<float, 4, torch::RestrictPtrTraits> depths,
                     const torch::PackedTensorAccessor64<TransformT, 3, torch::RestrictPtrTraits> transforms,
                     const bool transforms_in_opengl,
                     const glm::i64vec3 resolution_grid,
                     const glm::vec3 cube_corner_bfl,
                     const float cube_length,
                     const int64_t batch,
                     torch::PackedTensorAccessor64<float, 1, torch::RestrictPtrTraits> sparse_vh,
                     torch::PackedTensorAccessor64<float, 1, torch::RestrictPtrTraits> sparse_tsdf,
                     torch::PackedTensorAccessor64<float, 1, torch::RestrictPtrTraits> sparse_weights,
                     torch::PackedTensorAccessor64<float, 1, torch::RestrictPtrTraits> sparse_positive_weights,
                     torch::PackedTensorAccessor64<float, 1, torch::RestrictPtrTraits> sparse_solid_weights,
                     int64_t* depth_debug,
                     const bool use_depth_fallback,
                     const int edge_radius,
                     const float edge_threshold,
                     const float concavity_forward_offset_meters)
{
    const auto resolution_cells = glm::i64vec3{ resolution_grid.x - 1, resolution_grid.y - 1, resolution_grid.z - 1 };
    const auto H = masks.size(1); const auto W = masks.size(2);

    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for (auto tid = id; tid < N; tid += num_threads)
    {
        atomicAdd(reinterpret_cast<unsigned long long*>(&depth_debug[11]), 1ULL);
        auto g = unravel_index(sparse_indices[tid], resolution_grid);
        auto g_world = glm::vec3{
            cube_corner_bfl.x + static_cast<float>(g.x) / static_cast<float>(resolution_cells.x) * cube_length,
            cube_corner_bfl.y + static_cast<float>(g.y) / static_cast<float>(resolution_cells.y) * cube_length,
            cube_corner_bfl.z + static_cast<float>(g.z) / static_cast<float>(resolution_cells.z) * cube_length
        };

        auto g_camera = bmm_4x4_transforms(g_world, transforms, batch);
        
        auto g_px = transforms_in_opengl ? 
            glm::vec2{unnormalize_ndc_false(g_camera.x/g_camera.w, W), unnormalize_ndc_false(g_camera.y/g_camera.w, H)} :
            glm::vec2{align_cv_false(g_camera.x/g_camera.z), align_cv_false(g_camera.y/g_camera.z)};
        
        int64_t px = static_cast<int64_t>(roundf(g_px.x));
        int64_t py = static_cast<int64_t>(roundf(g_px.y));

        if (px < 1 || px >= W - 1 || py < 1 || py >= H - 1) {
            atomicAdd(reinterpret_cast<unsigned long long*>(&depth_debug[13]), 1ULL);
            continue;
        }

        float mask_val = sample_bilinear_mode_zeros_padding(masks, g_px.y, g_px.x, batch, 0);
        sparse_vh[tid] *= mask_val; 
        
        if (mask_val <= 0.5f) {
            atomicAdd(reinterpret_cast<unsigned long long*>(&depth_debug[12]), 1ULL);
            continue;
        }

        if (batch >= depths.size(0)) continue;

        float z_sensor = sample_depth_bilinear(depths, g_px.y, g_px.x, static_cast<int>(batch));
        if (z_sensor <= kDepthMinMeters || z_sensor > kDepthMaxMeters) {
            atomicAdd(reinterpret_cast<unsigned long long*>(&depth_debug[14]), 1ULL);
            continue;
        }

        if (is_depth_edge_strong(depths, g_px.y, g_px.x, static_cast<int>(batch), edge_radius, edge_threshold)) {
            atomicAdd(reinterpret_cast<unsigned long long*>(&depth_debug[15]), 1ULL);
            continue;
        }

        float metric_z = transforms_in_opengl ? fabsf(g_camera.w) : fabsf(g_camera.z);

        // Standard TSDF Definition: (Sensor Depth) - (Voxel Depth)
        // Positive = Empty Air (in front of object)
        // Zero = On Surface
        // Negative = Solid Inside
        float sdf = (z_sensor - kDepthSurfaceSlackMeters) - (metric_z - concavity_forward_offset_meters);
       
        
        // FIX: Removed truncation continue block. Allow deeply positive SDF to fuse perfectly
        // so that the entire space in front of the object is reconstructed.
        /*
        if (sdf > kMaxPositiveSdfForFusionMeters) {
            atomicAdd(reinterpret_cast<unsigned long long*>(&depth_debug[16]), 1ULL);
            continue; 
        }
        */

        constexpr float kTruncMargin = 0.05f;
        // OCCLUSION-AWARE FUSION: if this voxel lies more than the surface band BEHIND the camera's
        // measured surface, the surface blocks the camera's line of sight to the voxel -> the camera
        // CANNOT SEE this spot -> it gets NO vote (not solid, not surface). A camera only votes when
        // its measured surface is AT the voxel (surface vote) or the voxel is in front of it (free
        // space). This is the principle "cameras with no view of the spot don't vote": it stops
        // occluded oblique cameras from vetoing concavities, so adding cameras carves MORE, not less.
        if (sdf < -kPositiveSdfThresholdMeters) {
            atomicAdd(reinterpret_cast<unsigned long long*>(&depth_debug[17]), 1ULL);
            continue;
        }

        // CURLESS-LEVOY INCIDENCE WEIGHTING (replaces the binary grazing gate when kUseIncidenceWeight).
        // Every sample is weighted by w_inc = cos(theta)^p from the depth-gradient proxy. The shallow
        // false-air at the convex limb is exactly the grazing (cos~0) case, so it is suppressed smoothly
        // instead of by a hard threshold; frontal samples dominate the average. A zero weight
        // (invalid/edge/near-tangent) means the camera abstains here. Toggle off -> old uniform avg + gate.
        float w_inc;
        if (kUseIncidenceWeight)
        {
            float fz = focal_px_from_transform(transforms, static_cast<int>(batch), W) / z_sensor;
            w_inc = depth_incidence_weight(depths, g_px.y, g_px.x, static_cast<int>(batch), fz);
            if (w_inc <= 0.0f)
            {
                atomicAdd(reinterpret_cast<unsigned long long*>(&depth_debug[16]), 1ULL);  // grazing/edge -> abstain
                continue;
            }
        }
        else
        {
            bool grazing = is_depth_grazing(depths, g_px.y, g_px.x, static_cast<int>(batch), kGrazingGradThreshMetersPerPixel);
            if (grazing && sdf < kGrazingAirMinMeters)
            {
                atomicAdd(reinterpret_cast<unsigned long long*>(&depth_debug[16]), 1ULL);  // grazing shallow -> abstain
                continue;
            }
            w_inc = 1.0f;
        }

        float tsdf = fmaxf(fminf(sdf, kTruncMargin), -kTruncMargin);
        atomicAdd(&sparse_tsdf[tid], w_inc * tsdf);            // Curless-Levoy: Sum(w_i * d_i)
        atomicAdd(&sparse_weights[tid], w_inc);               //                Sum(w_i)
        if (sdf > kPositiveSdfThresholdMeters)
        {
            atomicAdd(&sparse_positive_weights[tid], w_inc);  // free space -> CARVE vote (incidence-weighted)
        }
        else
        {
            atomicAdd(&sparse_solid_weights[tid], w_inc);     // SURFACE vote (veto) -> incidence-weighted
        }
        atomicAdd(reinterpret_cast<unsigned long long*>(&depth_debug[18]), 1ULL);
    }
}

template <typename MaskT, typename TransformT>
__global__ void
accumulate_tsdf_partial(const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> sparse_indices,
                        const int64_t N,
                        const torch::PackedTensorAccessor64<MaskT, 4, torch::RestrictPtrTraits> masks,
                        const torch::PackedTensorAccessor64<float, 4, torch::RestrictPtrTraits> depths,
                        const torch::PackedTensorAccessor64<TransformT, 3, torch::RestrictPtrTraits> transforms,
                        const bool transforms_in_opengl,
                        const glm::i64vec3 resolution_grid,
                        const glm::vec3 cube_corner_bfl,
                        const float cube_length,
                        const int64_t batch,
                        torch::PackedTensorAccessor64<float, 1, torch::RestrictPtrTraits> sparse_vh,
                        torch::PackedTensorAccessor64<float, 1, torch::RestrictPtrTraits> sparse_tsdf,
                        torch::PackedTensorAccessor64<float, 1, torch::RestrictPtrTraits> sparse_weights,
                        torch::PackedTensorAccessor64<float, 1, torch::RestrictPtrTraits> sparse_positive_weights,
                        torch::PackedTensorAccessor64<float, 1, torch::RestrictPtrTraits> sparse_solid_weights,
                        int64_t* depth_debug,
                        const bool use_depth_fallback,
                        const int edge_radius,
                        const float edge_threshold,
                        const float concavity_forward_offset_meters)
{
    const auto resolution_cells = glm::i64vec3{ resolution_grid.x - 1, resolution_grid.y - 1, resolution_grid.z - 1 };
    const auto H = masks.size(1); const auto W = masks.size(2);

    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for (auto tid = id; tid < N; tid += num_threads)
    {
        atomicAdd(reinterpret_cast<unsigned long long*>(&depth_debug[11]), 1ULL);
        auto g = unravel_index(sparse_indices[tid], resolution_grid);
        auto g_world = glm::vec3{
            cube_corner_bfl.x + static_cast<float>(g.x) / static_cast<float>(resolution_cells.x) * cube_length,
            cube_corner_bfl.y + static_cast<float>(g.y) / static_cast<float>(resolution_cells.y) * cube_length,
            cube_corner_bfl.z + static_cast<float>(g.z) / static_cast<float>(resolution_cells.z) * cube_length
        };

        auto g_camera = bmm_4x4_transforms(g_world, transforms, batch);

        auto g_px = transforms_in_opengl ? 
            glm::vec2{unnormalize_ndc_false(g_camera.x/g_camera.w, W), unnormalize_ndc_false(g_camera.y/g_camera.w, H)} :
            glm::vec2{align_cv_false(g_camera.x/g_camera.z), align_cv_false(g_camera.y/g_camera.z)};
        
        int64_t px = static_cast<int64_t>(roundf(g_px.x));
        int64_t py = static_cast<int64_t>(roundf(g_px.y));

        if (px < 1 || px >= W - 1 || py < 1 || py >= H - 1) {
            atomicAdd(reinterpret_cast<unsigned long long*>(&depth_debug[13]), 1ULL);
            continue;
        }

        float mask_val = sample_bilinear_mode_ones_padding(masks, g_px.y, g_px.x, batch, 0);
        sparse_vh[tid] *= mask_val;

        if (mask_val <= 0.5f) {
            atomicAdd(reinterpret_cast<unsigned long long*>(&depth_debug[12]), 1ULL);
            continue;
        }

        if (batch >= depths.size(0)) continue;

        float z_sensor = sample_depth_bilinear(depths, g_px.y, g_px.x, static_cast<int>(batch));
        if (z_sensor <= kDepthMinMeters || z_sensor > kDepthMaxMeters) {
            atomicAdd(reinterpret_cast<unsigned long long*>(&depth_debug[14]), 1ULL);
            continue;
        }

        if (is_depth_edge_strong(depths, g_px.y, g_px.x, static_cast<int>(batch), edge_radius, edge_threshold)) {
            atomicAdd(reinterpret_cast<unsigned long long*>(&depth_debug[15]), 1ULL);
            continue;
        }

        float metric_z = transforms_in_opengl ? fabsf(g_camera.w) : fabsf(g_camera.z);

        float sdf = (z_sensor - kDepthSurfaceSlackMeters) - (metric_z - concavity_forward_offset_meters);
    // atomicAdd(reinterpret_cast<unsigned long long*>(&depth_debug[16]), 1ULL);

        
        // FIX: Removed truncation continue block.
        /*
        if (sdf > kMaxPositiveSdfForFusionMeters) {
            atomicAdd(reinterpret_cast<unsigned long long*>(&depth_debug[16]), 1ULL);
            continue;
        }
        */

        constexpr float kTruncMargin = 0.05f;
        // OCCLUSION-AWARE FUSION: if this voxel lies more than the surface band BEHIND the camera's
        // measured surface, the surface blocks the camera's line of sight to the voxel -> the camera
        // CANNOT SEE this spot -> it gets NO vote (not solid, not surface). A camera only votes when
        // its measured surface is AT the voxel (surface vote) or the voxel is in front of it (free
        // space). This is the principle "cameras with no view of the spot don't vote": it stops
        // occluded oblique cameras from vetoing concavities, so adding cameras carves MORE, not less.
        if (sdf < -kPositiveSdfThresholdMeters) {
            atomicAdd(reinterpret_cast<unsigned long long*>(&depth_debug[17]), 1ULL);
            continue;
        }

        // CURLESS-LEVOY INCIDENCE WEIGHTING (replaces the binary grazing gate when kUseIncidenceWeight).
        // Every sample is weighted by w_inc = cos(theta)^p from the depth-gradient proxy. The shallow
        // false-air at the convex limb is exactly the grazing (cos~0) case, so it is suppressed smoothly
        // instead of by a hard threshold; frontal samples dominate the average. A zero weight
        // (invalid/edge/near-tangent) means the camera abstains here. Toggle off -> old uniform avg + gate.
        float w_inc;
        if (kUseIncidenceWeight)
        {
            float fz = focal_px_from_transform(transforms, static_cast<int>(batch), W) / z_sensor;
            w_inc = depth_incidence_weight(depths, g_px.y, g_px.x, static_cast<int>(batch), fz);
            if (w_inc <= 0.0f)
            {
                atomicAdd(reinterpret_cast<unsigned long long*>(&depth_debug[16]), 1ULL);  // grazing/edge -> abstain
                continue;
            }
        }
        else
        {
            bool grazing = is_depth_grazing(depths, g_px.y, g_px.x, static_cast<int>(batch), kGrazingGradThreshMetersPerPixel);
            if (grazing && sdf < kGrazingAirMinMeters)
            {
                atomicAdd(reinterpret_cast<unsigned long long*>(&depth_debug[16]), 1ULL);  // grazing shallow -> abstain
                continue;
            }
            w_inc = 1.0f;
        }

        float tsdf = fmaxf(fminf(sdf, kTruncMargin), -kTruncMargin);
        atomicAdd(&sparse_tsdf[tid], w_inc * tsdf);            // Curless-Levoy: Sum(w_i * d_i)
        atomicAdd(&sparse_weights[tid], w_inc);               //                Sum(w_i)
        if (sdf > kPositiveSdfThresholdMeters)
        {
            atomicAdd(&sparse_positive_weights[tid], w_inc);  // free space -> CARVE vote (incidence-weighted)
        }
        else
        {
            atomicAdd(&sparse_solid_weights[tid], w_inc);     // SURFACE vote (veto) -> incidence-weighted
        }
        atomicAdd(reinterpret_cast<unsigned long long*>(&depth_debug[18]), 1ULL);
    }
}

__global__ void 
normalize_tsdf_field(torch::PackedTensorAccessor64<float, 1, torch::RestrictPtrTraits> sparse_vh,
                     torch::PackedTensorAccessor64<float, 1, torch::RestrictPtrTraits> sparse_tsdf,
                     torch::PackedTensorAccessor64<float, 1, torch::RestrictPtrTraits> sparse_weights,
                     torch::PackedTensorAccessor64<float, 1, torch::RestrictPtrTraits> sparse_positive_weights,
                     torch::PackedTensorAccessor64<float, 1, torch::RestrictPtrTraits> sparse_solid_weights,
                     int64_t* depth_debug,
                     const float carve_boost,
                     const int64_t N)
{
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int64_t i = tid; i < N; i += gridDim.x * blockDim.x) {
        float w = sparse_weights[i];
        float vh = sparse_vh[i];

        if (w > 0.0f) {
            float positive_w = sparse_positive_weights[i];   // cameras seeing DEEP free space here
            float surface_w  = sparse_solid_weights[i];      // cameras that measured a surface here

            // APPROACH A + SOFT VETO. Place the surface at the sub-voxel ZERO-CROSSING of the averaged
            // TSDF: mean_sdf in [-tau,tau] remaps to the [0,1] hull scale (air 0, surface 0.5, solid 1).
            // The veto is no longer a hard keep/carve switch (which gave marching cubes a binary frontier
            // to stair-step). Instead protection scales CONTINUOUSLY with how many HEAD-ON cameras measured
            // a surface here: p in [0,1] saturating at kVetoSaturationCount. We then blend between the
            // carved zero-crossing value (p=0) and the untouched hull (p=1), and MIN-fuse so depth can
            // only remove geometry, never add it. A true surface gets head-on surface votes -> p high ->
            // kept smooth; a real void gets none -> p=0 -> carved; the frontier is now graded, not jagged.
            float mean_sdf  = sparse_tsdf[i] / w;                              // metres, in [-tau, tau]
            float depth_val = 0.5f - 0.5f * (mean_sdf / kTruncMarginMeters);   // air 0, surf 0.5, solid 1
            depth_val = fminf(fmaxf(depth_val, 0.0f), 1.0f);
            float p = fminf(surface_w / kVetoSaturationCount, 1.0f);           // soft protection fraction
            float fused = depth_val * (1.0f - p) + vh * p;                     // carve <-> keep, continuous
            sparse_vh[i] = fminf(vh, fused);                                  // min-fuse: never adds geometry
            atomicAdd(reinterpret_cast<unsigned long long*>(&depth_debug[(fused < vh) ? 6 : 7]), 1ULL);
        } else {
            sparse_vh[i] = vh;
            atomicAdd(reinterpret_cast<unsigned long long*>(&depth_debug[7]), 1ULL);
        }
    }
}

template <typename MaskT, typename TransformT>
__global__ void
accumulate_hull_counts_full(const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> sparse_indices,
                            const int64_t N,
                            const torch::PackedTensorAccessor64<MaskT, 4, torch::RestrictPtrTraits> masks,
                            const torch::PackedTensorAccessor64<TransformT, 3, torch::RestrictPtrTraits> transforms,
                            const bool transforms_in_opengl,
                            const glm::i64vec3 resolution_grid,
                            const glm::vec3 cube_corner_bfl,
                            const float cube_length,
                            const int64_t batch,
                            torch::PackedTensorAccessor64<float, 1, torch::RestrictPtrTraits> sparse_values)
{
    const auto resolution_cells = glm::i64vec3{ resolution_grid.x - 1, resolution_grid.y - 1, resolution_grid.z - 1 };
    const auto H = masks.size(1);
    const auto W = masks.size(2);

    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for (auto tid = id; tid < N; tid += num_threads)
    {
        auto g = unravel_index(sparse_indices[tid], resolution_grid);
        auto g_world = glm::vec3{
            cube_corner_bfl.x + static_cast<float>(g.x) / static_cast<float>(resolution_cells.x) * cube_length,
            cube_corner_bfl.y + static_cast<float>(g.y) / static_cast<float>(resolution_cells.y) * cube_length,
            cube_corner_bfl.z + static_cast<float>(g.z) / static_cast<float>(resolution_cells.z) * cube_length
        };

        auto g_camera = bmm_4x4_transforms(g_world, transforms, batch);
        auto g_pixel = glm::vec2{};
        if (transforms_in_opengl)
        {
            auto g_camera_ndc = glm::vec2{ g_camera.x / g_camera.w, g_camera.y / g_camera.w };
            g_pixel = glm::vec2{ unnormalize_ndc_false(g_camera_ndc.x, W), unnormalize_ndc_false(g_camera_ndc.y, H) };
        }
        else
        {
            auto g_camera_cv = glm::vec2{ g_camera.x / g_camera.z, g_camera.y / g_camera.z };
            g_pixel = glm::vec2{ align_cv_false(g_camera_cv.x), align_cv_false(g_camera_cv.y) };
        }

        sparse_values[tid] *= sample_bilinear_mode_zeros_padding(masks, g_pixel.y, g_pixel.x, batch, 0);
    }
}

template <typename MaskT, typename TransformT>
__global__ void
accumulate_hull_counts_partial(const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> sparse_indices,
                               const int64_t N,
                               const torch::PackedTensorAccessor64<MaskT, 4, torch::RestrictPtrTraits> masks,
                               const torch::PackedTensorAccessor64<TransformT, 3, torch::RestrictPtrTraits> transforms,
                               const bool transforms_in_opengl,
                               const glm::i64vec3 resolution_grid,
                               const glm::vec3 cube_corner_bfl,
                               const float cube_length,
                               const int64_t batch,
                               torch::PackedTensorAccessor64<float, 1, torch::RestrictPtrTraits> sparse_values)
{
    const auto resolution_cells = glm::i64vec3{ resolution_grid.x - 1, resolution_grid.y - 1, resolution_grid.z - 1 };
    const auto H = masks.size(1);
    const auto W = masks.size(2);

    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for (auto tid = id; tid < N; tid += num_threads)
    {
        auto g = unravel_index(sparse_indices[tid], resolution_grid);
        auto g_world = glm::vec3{
            cube_corner_bfl.x + static_cast<float>(g.x) / static_cast<float>(resolution_cells.x) * cube_length,
            cube_corner_bfl.y + static_cast<float>(g.y) / static_cast<float>(resolution_cells.y) * cube_length,
            cube_corner_bfl.z + static_cast<float>(g.z) / static_cast<float>(resolution_cells.z) * cube_length
        };

        auto g_camera = bmm_4x4_transforms(g_world, transforms, batch);
        auto g_pixel = glm::vec2{};
        if (transforms_in_opengl)
        {
            auto g_camera_ndc = glm::vec2{ g_camera.x / g_camera.w, g_camera.y / g_camera.w };
            g_pixel = glm::vec2{ unnormalize_ndc_false(g_camera_ndc.x, W), unnormalize_ndc_false(g_camera_ndc.y, H) };
        }
        else
        {
            auto g_camera_cv = glm::vec2{ g_camera.x / g_camera.z, g_camera.y / g_camera.z };
            g_pixel = glm::vec2{ align_cv_false(g_camera_cv.x), align_cv_false(g_camera_cv.y) };
        }

        auto g_pixel_rounded = glm::i64vec2{ roundf(g_pixel.x), roundf(g_pixel.y) };
        if (in_image(g_pixel_rounded.y, g_pixel_rounded.x, H, W, 1))
        {
            sparse_values[tid] *= sample_bilinear_mode_ones_padding(masks, g_pixel.y, g_pixel.x, batch, 0);
        }
    }
}

__global__ void
extract_sparse_indices(const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> sparse_indices,
                       const glm::i64vec3 resolution_grid,
                       torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> sparse_indices_unraveled)
{
    const auto N = sparse_indices.size(0);

    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for (auto tid = id; tid < N; tid += num_threads)
    {
        auto g = unravel_index(sparse_indices[tid], resolution_grid);

        sparse_indices_unraveled[0][tid] = 0;
        sparse_indices_unraveled[1][tid] = g.z;
        sparse_indices_unraveled[2][tid] = g.y;
        sparse_indices_unraveled[3][tid] = g.x;
    }
}

torch::Tensor
to_sparse_coo_tensor(const RavelledSparseTensor& ravelled_tensor)
{
    auto sparse_indices = ravelled_tensor.indices();

    at::cuda::CUDAGuard device_guard{ sparse_indices.device() };
    const auto stream = at::cuda::getCurrentCUDAStream();

    const auto dtype_int64 = torch::TensorOptions{}.dtype(torch::kInt64).device(sparse_indices.device());

    const auto resolution_grid =
            glm::i64vec3{ ravelled_tensor.size(3), ravelled_tensor.size(2), ravelled_tensor.size(1) };

    const auto N = sparse_indices.numel();

    auto sparse_indices_unraveled = torch::empty({ 4, N }, dtype_int64);

    if (N != 0)
    {
        const int threads_per_block = 128;
        dim3 grid;
        at::cuda::getApplyGrid(N, grid, sparse_indices.device().index(), threads_per_block);
        dim3 threads = at::cuda::getApplyBlock(threads_per_block);

        extract_sparse_indices<<<grid, threads, 0, stream>>>(
                sparse_indices.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                resolution_grid,
                sparse_indices_unraveled.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>());
        AT_CUDA_CHECK(cudaGetLastError());
        AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    auto sparse_field =
            torch::sparse_coo_tensor(sparse_indices_unraveled, ravelled_tensor.values(), ravelled_tensor.sizes())
                    .coalesce();

    return sparse_field;
}

std::tuple<RavelledSparseTensor, std::vector<torch::Tensor>>
sparse_visual_hull_field_cuda_ravelled(const torch::Tensor& masks,
                                       const torch::Tensor& depths,
                                       const torch::Tensor& transforms,
                                       const int level,
                                       const std::array<float, 3>& cube_corner_bfl,
                                       const float cube_length,
                                       const bool masks_partial,
                                       const std::string& transforms_convention,
                                       const int depth_fusion_preset)
{
    const auto preset = sanitize_depth_fusion_preset(depth_fusion_preset);
    const bool use_tsdf = depth_fusion_enabled(preset);
    const bool use_depth_fallback = use_depth_fallback_neighborhood(preset);
    const int edge_radius = depth_edge_radius(preset);
    const float edge_threshold = depth_edge_threshold(preset);
    float concavity_forward_offset = concavity_forward_offset_meters(preset);
    // Presentation override: sweep the concavity carve offset at runtime (no recompile).
    // Lower offset -> less carving -> box fills up; higher -> more carved/open.
    if (const char* env_off = std::getenv("RIFT_CARVE_OFFSET"))
        concavity_forward_offset = static_cast<float>(std::atof(env_off));
    const float tsdf_carve_boost = carve_boost(preset);

    TORCH_CHECK_EQ(masks.device(), transforms.device());
    TORCH_CHECK_EQ(masks.dim(), 4);
    TORCH_CHECK_EQ(depths.dim(), 4);                            
    TORCH_CHECK_EQ(transforms.dim(), 3);
    TORCH_CHECK_EQ(transforms.size(1), 4);
    TORCH_CHECK_EQ(transforms.size(2), 4);
    TORCH_CHECK_EQ(masks.size(0), transforms.size(0));
    TORCH_CHECK_EQ(masks.size(3), 1);
    TORCH_CHECK_GE(level, 0);
    TORCH_CHECK_GT(cube_length, 0.f);

    auto transforms_in_opengl = bool{};
    if (transforms_convention == "opengl")
    {
        transforms_in_opengl = true;
    }
    else if (transforms_convention == "opencv")
    {
        transforms_in_opengl = false;
    }
    else
    {
        TORCH_CHECK(false, "Unsupported transforms_convention \"" + transforms_convention + "\".");
    }

    at::cuda::CUDAGuard device_guard{ masks.device() };
    const auto stream = at::cuda::getCurrentCUDAStream();

    at::cuda::ThrustAllocator allocator;
    const auto policy = thrust::cuda::par(allocator).on(stream);

    const auto dtype_uint8 = torch::TensorOptions{}.dtype(torch::kUInt8).device(masks.device());
    const auto dtype_int64 = torch::TensorOptions{}.dtype(torch::kInt64).device(masks.device());
    const auto dtype_float = torch::TensorOptions{}.dtype(torch::kFloat32).device(masks.device());

    auto integral_masks = integral_image(masks, torch::kFloat32);

    auto candidates = torch::tensor({ 0 }, dtype_int64);
    auto candidates_octree = std::vector<torch::Tensor>{};
    candidates_octree.push_back(candidates);

    auto cube_corner_bfl_cuda = glm::vec3{ cube_corner_bfl[0], cube_corner_bfl[1], cube_corner_bfl[2] };

    auto depth_debug = torch::zeros({ 20 }, dtype_int64);
    auto depth_debug_ptr = depth_debug.data_ptr<int64_t>();
    
    for (int i = 0; i < level; ++i)
    {
        const auto N = 8 * candidates.numel(); 
        const auto resolution = glm::i64vec3{ 1 << i };
        const auto resolution_children = glm::i64vec3{ 1 << (i + 1) }; 

        auto occupied_voxel = torch::empty({ N }, dtype_uint8).contiguous(); 

        auto candidates_ = candidates.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(); 
        auto integral_masks_ = integral_masks.packed_accessor64<float, 4, torch::RestrictPtrTraits>();
        auto depths_ = depths.packed_accessor64<float, 4, torch::RestrictPtrTraits>(); 
        auto occupied_voxel_ = occupied_voxel.packed_accessor64<uint8_t, 1, torch::RestrictPtrTraits>();

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                transforms.scalar_type(),
                "classify_children",
                [&]()
                {
                    auto transforms_ = transforms.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>();

                    const int threads_per_block = 128;
                    dim3 grid_volume;
                    at::cuda::getApplyGrid(N, grid_volume, masks.device().index(), threads_per_block);
                    dim3 threads = at::cuda::getApplyBlock(threads_per_block);

                    if (masks_partial)
                    {
                        classify_children_partial<<<grid_volume, threads, 0, stream>>>(candidates_,
                                                                                       integral_masks_,
                                                                                       depths_,
                                                                                       transforms_,
                                                                                       transforms_in_opengl,
                                                                                       resolution,
                                                                                       resolution_children,
                                                                                       cube_corner_bfl_cuda,
                                                                                       cube_length,
                                                                                       i, level,
                                                                                       use_tsdf,
                                                                                       occupied_voxel_,
                                                                                       depth_debug_ptr);
                        AT_CUDA_CHECK(cudaGetLastError());
                        AT_CUDA_CHECK(cudaStreamSynchronize(stream));
                    }
                    else
                    {
                        classify_children_full<<<grid_volume, threads, 0, stream>>>(candidates_,
                                                                                    integral_masks_,
                                                                                    depths_,
                                                                                    transforms_,
                                                                                    transforms_in_opengl,
                                                                                    resolution,
                                                                                    resolution_children,
                                                                                    cube_corner_bfl_cuda,
                                                                                    cube_length,
                                                                                    i, level,
                                                                                    use_tsdf,
                                                                                    concavity_forward_offset,
                                                                                    edge_radius,
                                                                                    edge_threshold,
                                                                                    occupied_voxel_,
                                                                                    depth_debug_ptr);
                        AT_CUDA_CHECK(cudaGetLastError());
                        AT_CUDA_CHECK(cudaStreamSynchronize(stream));
                    }
                });

        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        auto num_selected_out = torch::empty({ 1 }, dtype_int64).contiguous();

        auto f = [candidates_, resolution, resolution_children] C10_HOST_DEVICE(const int64_t tid) -> int64_t
        {
#if defined(__CUDA_ARCH__)
            auto candidate_id = tid / 8;
            auto child_id = tid % 8;

            auto g = unravel_index(candidates_[candidate_id], resolution);
            auto g_child = cube_vertex(int64_t{ 2 } * g, child_id);

            return ravel_multi_index(g_child, resolution_children);
#else
            (void)tid;
            TORCH_CHECK(false,
                        "Host version of extended lambda is only defined to workaround NVCC limitations."
                        "Do not call this function on the host side.");
            return int64_t{ 0 };
#endif
        };

        auto new_candidates = torch::empty({ N }, dtype_int64).contiguous();

        TORCH_CHECK_LT(N, (static_cast<int64_t>(1) << 31));

        AT_CUDA_CHECK(
                cub::DeviceSelect::Flagged(d_temp_storage,
                                           temp_storage_bytes,
                                           thrust::make_transform_iterator(thrust::counting_iterator<int64_t>(0), f),
                                           occupied_voxel.data_ptr<uint8_t>(),
                                           new_candidates.data_ptr<int64_t>(),
                                           num_selected_out.data_ptr<int64_t>(),
                                           N,
                                           stream));
        AT_CUDA_CHECK(cudaStreamSynchronize(stream));

        auto temp_storage = torch::empty({ static_cast<int64_t>(temp_storage_bytes) }, dtype_uint8).contiguous();

        AT_CUDA_CHECK(
                cub::DeviceSelect::Flagged(temp_storage.data_ptr(),
                                           temp_storage_bytes,
                                           thrust::make_transform_iterator(thrust::counting_iterator<int64_t>(0), f),
                                           occupied_voxel.data_ptr<uint8_t>(),
                                           new_candidates.data_ptr<int64_t>(),
                                           num_selected_out.data_ptr<int64_t>(),
                                           N,
                                           stream));
        AT_CUDA_CHECK(cudaStreamSynchronize(stream));

        new_candidates.resize_({ num_selected_out.cpu().item<int64_t>() });

        candidates = new_candidates;
        candidates_octree.push_back(candidates);

        if (candidates.numel() == 0)
        {
            break;
        }
    }

    const auto resolution_cells = glm::i64vec3{ 1 << level };
    const auto resolution_grid = resolution_cells + int64_t{ 1 };

    if (candidates.numel() == 0)
    {
        auto sparse_indices = torch::empty({ 0 }, dtype_int64);
        auto sparse_values = torch::empty({ 0 }, dtype_float);

        return { RavelledSparseTensor{ sparse_indices,
                                       sparse_values,
                                       { 1, resolution_grid.z, resolution_grid.y, resolution_grid.x } },
                 candidates_octree };
    }

    integral_masks = torch::Tensor{};

    auto all_corners = torch::empty({ 8 * candidates.numel() }, dtype_int64).contiguous();
    auto sparse_indices = torch::empty({ 8 * candidates.numel() }, dtype_int64).contiguous();

    auto candidates_ = candidates.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>();

    thrust::transform(policy,
                      thrust::counting_iterator<int64_t>(0),
                      thrust::counting_iterator<int64_t>(all_corners.numel()),
                      all_corners.data_ptr<int64_t>(),
                      [candidates_, resolution_cells, resolution_grid] C10_DEVICE(const int64_t tid) -> int64_t
                      {
                          auto cell_id = tid / 8;
                          auto corner_id = tid % 8;

                          auto g = unravel_index(candidates_[cell_id], resolution_cells);
                          auto v = cube_vertex(g, corner_id);

                          return ravel_multi_index(v, resolution_grid);
                      });

    thrust::sort(policy, all_corners.data_ptr<int64_t>(), all_corners.data_ptr<int64_t>() + all_corners.numel());

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    auto num_selected_out = torch::empty({ 1 }, dtype_int64).contiguous();

    TORCH_CHECK_LT(8 * candidates.numel(), (static_cast<int64_t>(1) << 31));

    AT_CUDA_CHECK(cub::DeviceSelect::Unique(d_temp_storage,
                                            temp_storage_bytes,
                                            all_corners.data_ptr<int64_t>(),
                                            sparse_indices.data_ptr<int64_t>(),
                                            num_selected_out.data_ptr<int64_t>(),
                                            all_corners.numel(),
                                            stream));
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));

    auto temp_storage = torch::empty({ static_cast<int64_t>(temp_storage_bytes) }, dtype_uint8).contiguous();

    AT_CUDA_CHECK(cub::DeviceSelect::Unique(temp_storage.data_ptr(),
                                            temp_storage_bytes,
                                            all_corners.data_ptr<int64_t>(),
                                            sparse_indices.data_ptr<int64_t>(),
                                            num_selected_out.data_ptr<int64_t>(),
                                            all_corners.numel(),
                                            stream));
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));

    auto N = num_selected_out.cpu().item<int64_t>();
    sparse_indices.resize_({ N });
    all_corners = torch::Tensor{};

    auto sparse_indices_ = sparse_indices.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>();

    auto sparse_values = torch::Tensor{};

    if (use_tsdf) 
    {
        sparse_values = torch::ones({ N }, dtype_float);
        auto sparse_tsdf = torch::zeros({ N }, dtype_float);
        auto sparse_weights = torch::zeros({ N }, dtype_float);
        auto sparse_positive_weights = torch::zeros({ N }, dtype_float);
        auto sparse_solid_weights = torch::zeros({ N }, dtype_float);
        
        auto sparse_vh_ = sparse_values.packed_accessor64<float, 1, torch::RestrictPtrTraits>();
        auto sparse_tsdf_ = sparse_tsdf.packed_accessor64<float, 1, torch::RestrictPtrTraits>();
        auto sparse_weights_ = sparse_weights.packed_accessor64<float, 1, torch::RestrictPtrTraits>();
        auto sparse_positive_weights_ = sparse_positive_weights.packed_accessor64<float, 1, torch::RestrictPtrTraits>();
        auto sparse_solid_weights_ = sparse_solid_weights.packed_accessor64<float, 1, torch::RestrictPtrTraits>();
        auto depths_ = depths.packed_accessor64<float, 4, torch::RestrictPtrTraits>(); 
        
        AT_DISPATCH_ALL_TYPES_AND(
            torch::ScalarType::Half,
            masks.scalar_type(),
            "accumulate_tsdf",
            [&]()
            {
                auto masks_ = masks.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>();
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    transforms.scalar_type(),
                    "accumulate_tsdf",
                    [&]()
                    {
                        auto transforms_ = transforms.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>();
                        for (auto batch = int64_t{ 0 }; batch < masks.size(0); ++batch)
                        {
                            const int threads_per_block = 128;
                            dim3 grid_corners;
                            at::cuda::getApplyGrid(N, grid_corners, masks.device().index(), threads_per_block);
                            dim3 threads = at::cuda::getApplyBlock(threads_per_block);

                            if (masks_partial) {
                                accumulate_tsdf_partial<<<grid_corners, threads, 0, stream>>>(
                                        sparse_indices_, N, masks_, depths_, transforms_, transforms_in_opengl,
                                        resolution_grid, cube_corner_bfl_cuda, cube_length, batch, sparse_vh_, sparse_tsdf_, sparse_weights_, sparse_positive_weights_, sparse_solid_weights_, depth_debug_ptr,
                                        use_depth_fallback, edge_radius, edge_threshold, concavity_forward_offset);
                            } else {
                                accumulate_tsdf_full<<<grid_corners, threads, 0, stream>>>(
                                        sparse_indices_, N, masks_, depths_, transforms_, transforms_in_opengl,
                                        resolution_grid, cube_corner_bfl_cuda, cube_length, batch, sparse_vh_, sparse_tsdf_, sparse_weights_, sparse_positive_weights_, sparse_solid_weights_, depth_debug_ptr,
                                        use_depth_fallback, edge_radius, edge_threshold, concavity_forward_offset);
                            }
                            AT_CUDA_CHECK(cudaGetLastError());
                            AT_CUDA_CHECK(cudaStreamSynchronize(stream));
                        }
                    });
            });

        const int threads_per_block = 128;
        dim3 grid_norm;
        at::cuda::getApplyGrid(N, grid_norm, masks.device().index(), threads_per_block);
        dim3 threads_norm = at::cuda::getApplyBlock(threads_per_block);
        
        normalize_tsdf_field<<<grid_norm, threads_norm, 0, stream>>>(
            sparse_vh_,
            sparse_tsdf_,
            sparse_weights_,
            sparse_positive_weights_,
            sparse_solid_weights_,
            depth_debug_ptr,
            tsdf_carve_boost,
            N);
        AT_CUDA_CHECK(cudaGetLastError());
        AT_CUDA_CHECK(cudaStreamSynchronize(stream));

        auto debug = depth_debug.to(torch::kCPU);
        std::cout << "\n[TORCHHULL: EXTENDED TSDF METRICS]\n"
                  << "--- 1. OCTREE 2D SILHOUETTE GATE ---\n"
                  << "  -> [CLASSIFY] Total Voxels Evaluated: " << debug[9].item<int64_t>() + debug[10].item<int64_t>() << "\n"
                  << "  -> [CLASSIFY] Empty by 2D Mask: " << debug[9].item<int64_t>() << "\n"
                  << "  -> [CLASSIFY] Survived 2D Mask: " << debug[10].item<int64_t>() << "\n"
                  << "--- 2. OCTREE 3D DEPTH CARVING ---\n"
                  << "  -> [CARVED] Empty Air in Concavity: " << debug[0].item<int64_t>() << " (DISABLED)\n"
                  << "  -> [KEPT] Inside Surface Margin: " << debug[4].item<int64_t>() << " (DISABLED)\n"
                  << "  -> [KEPT] Background Leak Shield: " << debug[3].item<int64_t>() << " (DISABLED)\n"
                  << "  -> [KEPT] Shielded by Edge Gradient: " << debug[8].item<int64_t>() << " (DISABLED)\n"
                  << "  -> [SKIPPED] Out of Pixel Bounds: " << debug[1].item<int64_t>() << " (DISABLED)\n"
                  << "  -> [SKIPPED] Invalid Sensor Range: " << debug[2].item<int64_t>() << " (DISABLED)\n"
                  << "--- 3. TSDF ACCUMULATION ---\n"
                  << "  -> Boundary Voxels Created (N): " << N << "\n"
                  << "  -> [TSDF] Total Voxel-Cam Checks: " << debug[11].item<int64_t>() << "\n"
                  << "  -> [TSDF] Rejected by 2D Mask (<0.5): " << debug[12].item<int64_t>() << "\n"
                  << "  -> [TSDF] Rejected by Pixel Bounds: " << debug[13].item<int64_t>() << "\n"
                  << "  -> [TSDF] Rejected by Sensor Range: " << debug[14].item<int64_t>() << "\n"
                  << "  -> [TSDF] Shielded by Edge Gradient: " << debug[15].item<int64_t>() << "\n"
                  << "  -> [TSDF] Shielded by Background Ray: " << debug[16].item<int64_t>() << " (REMOVED)\n"
                  << "  -> [TSDF] Skipped (Deep Inside Object): " << debug[17].item<int64_t>() << "\n"
                  << "  -> [TSDF] Successfully Fused: " << debug[18].item<int64_t>() << "\n"
                  << "--- 4. SCALAR NORMALIZATION ---\n"
                  << "  -> [FINAL] Voxels seen by Depth: " << debug[6].item<int64_t>() << "\n"
                  << "  -> [FINAL] Hidden Voxels (VH Preserved): " << debug[7].item<int64_t>() << "\n"
                  << "--- 0. OCTREE DEPTH CARVE (finest level) ---\n"
                  << "  -> [OCTREE] Finest candidates evaluated: " << debug[5].item<int64_t>() << "\n"
                  << "  -> [OCTREE] Voxels carved (removed): " << debug[19].item<int64_t>() << "\n\n";
    }
    else 
    {
        sparse_values = torch::ones({ N }, dtype_float);
        auto sparse_values_ = sparse_values.packed_accessor64<float, 1, torch::RestrictPtrTraits>();

        AT_DISPATCH_ALL_TYPES_AND(
            torch::ScalarType::Half,
            masks.scalar_type(),
            "accumulate_hull_counts",
            [&]()
            {
                auto masks_ = masks.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>();
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    transforms.scalar_type(),
                    "accumulate_hull_counts",
                    [&]()
                    {
                        auto transforms_ = transforms.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>();
                        for (auto batch = int64_t{ 0 }; batch < masks.size(0); ++batch)
                        {
                            const int threads_per_block = 128;
                            dim3 grid_corners;
                            at::cuda::getApplyGrid(N, grid_corners, masks.device().index(), threads_per_block);
                            dim3 threads = at::cuda::getApplyBlock(threads_per_block);

                            if (masks_partial) {
                                accumulate_hull_counts_partial<<<grid_corners, threads, 0, stream>>>(
                                        sparse_indices_, N, masks_, transforms_, transforms_in_opengl,
                                        resolution_grid, cube_corner_bfl_cuda, cube_length, batch, sparse_values_);
                            } else {
                                accumulate_hull_counts_full<<<grid_corners, threads, 0, stream>>>(
                                        sparse_indices_, N, masks_, transforms_, transforms_in_opengl,
                                        resolution_grid, cube_corner_bfl_cuda, cube_length, batch, sparse_values_);
                            }
                            AT_CUDA_CHECK(cudaGetLastError());
                            AT_CUDA_CHECK(cudaStreamSynchronize(stream));
                        }
                    });
            });
    }

    return { RavelledSparseTensor{ sparse_indices,
                                   sparse_values,
                                   { 1, resolution_grid.z, resolution_grid.y, resolution_grid.x } },
             candidates_octree };
}

torch::Tensor
sparse_visual_hull_field_cuda(const torch::Tensor& masks,
                              const torch::Tensor& depths,
                              const torch::Tensor& transforms,
                              const int level,
                              const std::array<float, 3>& cube_corner_bfl,
                              const float cube_length,
                              const bool masks_partial,
                              const std::string& transforms_convention,
                              const int depth_fusion_preset)
{
    auto [sparse_volume, _] = sparse_visual_hull_field_cuda_ravelled(masks,
                                                                     depths,
                                                                     transforms,
                                                                     level,
                                                                     cube_corner_bfl,
                                                                     cube_length,
                                                                     masks_partial,
                                                                     transforms_convention,
                                                                     depth_fusion_preset);

    return to_sparse_coo_tensor(sparse_volume);
}

template <typename scalar_t>
__global__ void
to_global_coordinates(torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> verts,
                      const glm::vec3 cube_corner_bfl,
                      const float cube_length,
                      const glm::i64vec3 resolution)
{
    const auto N = verts.size(0);

    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for (auto tid = id; tid < N; tid += num_threads)
    {
        verts[tid][0] = fmaf(cube_length / static_cast<float>(resolution.x),
                             static_cast<float>(verts[tid][0]),
                             cube_corner_bfl.x);
        verts[tid][1] = fmaf(cube_length / static_cast<float>(resolution.y),
                             static_cast<float>(verts[tid][1]),
                             cube_corner_bfl.y);
        verts[tid][2] = fmaf(cube_length / static_cast<float>(resolution.z),
                             static_cast<float>(verts[tid][2]),
                             cube_corner_bfl.z);
    }
}

template <typename scalar_t>
__global__ void
flip_faces(torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> faces)
{
    const auto N = faces.size(0);

    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for (auto tid = id; tid < N; tid += num_threads)
    {
        auto temp = faces[tid][1];
        faces[tid][1] = faces[tid][2];
        faces[tid][2] = temp;
    }
}

void
to_global_coordinates_and_flip_faces_(std::tuple<torch::Tensor, torch::Tensor>& self,
                                      const std::array<float, 3>& cube_corner_bfl,
                                      const float cube_length,
                                      const glm::i64vec3& resolution)
{
    auto verts = torch::Tensor{};
    auto faces = torch::Tensor{};
    std::tie(verts, faces) = self;

    TORCH_CHECK_EQ(verts.device(), faces.device());
    TORCH_CHECK_EQ(verts.size(1), 3);
    TORCH_CHECK_EQ(faces.size(1), 3);
    TORCH_CHECK_GT(cube_length, 0.f);

    at::cuda::CUDAGuard device_guard{ verts.device() };
    const auto stream = at::cuda::getCurrentCUDAStream();

    auto cube_corner_bfl_cuda = glm::vec3{ cube_corner_bfl[0], cube_corner_bfl[1], cube_corner_bfl[2] };

    const auto N_verts = verts.size(0);
    if (N_verts > 0)
    {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                verts.scalar_type(),
                "to_global_coordinates",
                [&]()
                {
                    const int threads_per_block = 128;
                    dim3 grid;
                    at::cuda::getApplyGrid(N_verts, grid, verts.device().index(), threads_per_block);
                    dim3 threads = at::cuda::getApplyBlock(threads_per_block);

                    to_global_coordinates<<<grid, threads, 0, stream>>>(
                            verts.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                            cube_corner_bfl_cuda,
                            cube_length,
                            resolution);
                    AT_CUDA_CHECK(cudaGetLastError());
                    AT_CUDA_CHECK(cudaStreamSynchronize(stream));
                });
    }

    const auto N_faces = faces.size(0);
    if (N_faces > 0)
    {
        AT_DISPATCH_INTEGRAL_TYPES(faces.scalar_type(),
                                   "flip_faces",
                                   [&]()
                                   {
                                       const int threads_per_block = 128;
                                       dim3 grid;
                                       at::cuda::getApplyGrid(N_faces, grid, faces.device().index(), threads_per_block);
                                       dim3 threads = at::cuda::getApplyBlock(threads_per_block);

                                       flip_faces<<<grid, threads, 0, stream>>>(
                                               faces.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>());
                                       AT_CUDA_CHECK(cudaGetLastError());
                                       AT_CUDA_CHECK(cudaStreamSynchronize(stream));
                                   });
    }
}

__global__ void
to_wireframe(const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> candidates,
             const int64_t N,
             const glm::vec3 cube_corner_bfl,
             const float cube_length,
             const glm::i64vec3 resolution,
             torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> candidate_verts,
             torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> candidate_edges)
{
    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for (auto tid = id; tid < N; tid += num_threads)
    {
        auto g = unravel_index(candidates[tid], resolution);

        for (auto i = 0; i < 8; ++i)
        {
            auto v = cube_vertex(g, i);

            auto v_world = glm::vec3{ fmaf(cube_length / static_cast<float>(resolution.x), v.x, cube_corner_bfl.x),
                                      fmaf(cube_length / static_cast<float>(resolution.y), v.y, cube_corner_bfl.y),
                                      fmaf(cube_length / static_cast<float>(resolution.z), v.z, cube_corner_bfl.z) };

            auto index = 8 * tid + i;
            candidate_verts[index][0] = v_world.x;
            candidate_verts[index][1] = v_world.y;
            candidate_verts[index][2] = v_world.z;
        }

        for (auto i = 0; i < 12; ++i)
        {
            auto index = 12 * tid + i;
            candidate_edges[index][0] = 8 * tid + edge_to_vertex_table[i][0];
            candidate_edges[index][1] = 8 * tid + edge_to_vertex_table[i][1];
        }
    }
}

std::vector<std::tuple<torch::Tensor, torch::Tensor>>
candidate_voxels_to_wireframes_cuda(const std::vector<torch::Tensor>& candidate_voxels,
                                    const std::array<float, 3>& cube_corner_bfl,
                                    const float cube_length)
{
    for (auto& candidates : candidate_voxels)
    {
        TORCH_CHECK_EQ(candidates.device(), candidate_voxels[0].device());
    }

    at::cuda::CUDAGuard device_guard{ candidate_voxels[0].device() };
    const auto stream = at::cuda::getCurrentCUDAStream();

    const auto dtype_int64 = torch::TensorOptions{}.dtype(torch::kInt64).device(candidate_voxels[0].device());
    const auto dtype_float = torch::TensorOptions{}.dtype(torch::kFloat32).device(candidate_voxels[0].device());

    auto cube_corner_bfl_cuda = glm::vec3{ cube_corner_bfl[0], cube_corner_bfl[1], cube_corner_bfl[2] };

    auto wireframes = std::vector<std::tuple<torch::Tensor, torch::Tensor>>{};

    for (int64_t i = 0; i < static_cast<int64_t>(candidate_voxels.size()); ++i)
    {
        auto candidates = candidate_voxels[i];
        const auto N = candidates.numel();

        auto candidate_verts = torch::empty({ 8 * N, 3 }, dtype_float);
        auto candidate_edges = torch::empty({ 12 * N, 2 }, dtype_int64);

        if (N > 0)
        {
            const int threads_per_block = 128;
            dim3 grid;
            at::cuda::getApplyGrid(N, grid, candidate_voxels[0].device().index(), threads_per_block);
            dim3 threads = at::cuda::getApplyBlock(threads_per_block);

            to_wireframe<<<grid, threads, 0, stream>>>(
                    candidates.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                    N,
                    cube_corner_bfl_cuda,
                    cube_length,
                    glm::i64vec3{ 1 << i },
                    candidate_verts.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
                    candidate_edges.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>());
            AT_CUDA_CHECK(cudaGetLastError());
            AT_CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        wireframes.emplace_back(candidate_verts, candidate_edges);
    }

    return wireframes;
}

} // namespace torchhull