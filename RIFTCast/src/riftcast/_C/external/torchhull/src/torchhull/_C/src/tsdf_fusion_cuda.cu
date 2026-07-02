#include <ATen/Dispatch.h>
#include <ATen/cuda/ApplyGridUtils.cuh>
#include <ATen/cuda/ThrustAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <cub/device/device_select.cuh>
#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/epsilon.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <torch/nn/functional.h>
#include <torch/types.h>

#include <torchhull/image_utils.h>
#include <torchhull/marching_cubes_utils.h>
#include <torchhull/math.h>
#include <torchhull/preprocessor.h>
#include <torchhull/ravelled_sparse_tensor.h>

namespace torchhull
{

class CudaTimer
{
public:
    CudaTimer()
    {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~CudaTimer()
    {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void
    start(cudaStream_t stream = 0)
    {
        cudaEventRecord(start_, stream);
    }

    float
    stop(const std::string& label = "", cudaStream_t stream = 0)
    {
        cudaEventRecord(stop_, stream);
        cudaEventSynchronize(stop_);
        float ms = 0;
        cudaEventElapsedTime(&ms, start_, stop_);
        if (!label.empty())
        {
            // std::cout << label << ": " << ms << " ms\n";
        }
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
};

template <typename ValueT>
inline C10_DEVICE glm::vec4
bmm_4x4_transforms_COPY(const glm::vec3& v,
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

template <typename TransformT>
__global__ void
classify_children_full_COPY(const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> candidates,
                            const torch::PackedTensorAccessor64<float, 4, torch::RestrictPtrTraits> integral_masks,
                            const torch::PackedTensorAccessor64<TransformT, 3, torch::RestrictPtrTraits> transforms,
                            const bool transforms_in_opengl,
                            const glm::i64vec3 resolution,
                            const glm::i64vec3 resolution_children,
                            const glm::vec3 cube_corner_bfl,
                            const float cube_length,
                            torch::PackedTensorAccessor64<uint8_t, 1, torch::RestrictPtrTraits> occupied_voxel)
{
    const auto N = occupied_voxel.size(0);

    // Note: image has dims (N, H, W, C) instead of (N, C, H, W)
    const auto H = integral_masks.size(1);
    const auto W = integral_masks.size(2);

    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for (auto tid = id; tid < N; tid += num_threads)
    {
        auto candidate_id = tid / 8;
        auto child_id = tid % 8;

        auto g = unravel_index(candidates[candidate_id], resolution);
        auto g_child = cube_vertex(int64_t{ 2 } * g, child_id);

        auto is_empty = false;
        [[maybe_unused]] auto is_object = false;
        auto should_refine = false;
        for (auto batch = int64_t{ 0 }; batch < integral_masks.size(0); ++batch)
        {
            auto bb_min = glm::vec2{ FLT_MAX, FLT_MAX };
            auto bb_max = glm::vec2{ -FLT_MAX, -FLT_MAX };
            for (auto i = 0; i < 8; ++i)
            {
                auto v = cube_vertex(g_child, i);
                auto v_world =
                        glm::vec3{ cube_corner_bfl.x + static_cast<float>(v.x) /
                                                               static_cast<float>(resolution_children.x) * cube_length,
                                   cube_corner_bfl.y + static_cast<float>(v.y) /
                                                               static_cast<float>(resolution_children.y) * cube_length,
                                   cube_corner_bfl.z + static_cast<float>(v.z) /
                                                               static_cast<float>(resolution_children.z) *
                                                               cube_length };

                auto v_camera = bmm_4x4_transforms_COPY(v_world, transforms, batch);

                auto v_pixel = glm::vec2{};
                if (transforms_in_opengl)
                {
                    auto v_camera_ndc = glm::vec2{ v_camera.x / v_camera.w, v_camera.y / v_camera.w };
                    v_pixel = glm::vec2{ unnormalize_ndc_false(v_camera_ndc.x, W),
                                         unnormalize_ndc_false(v_camera_ndc.y, H) };
                }
                else
                {
                    auto v_camera_cv = glm::vec2{ v_camera.x / v_camera.z, v_camera.y / v_camera.z };
                    v_pixel = glm::vec2{ align_cv_false(v_camera_cv.x), align_cv_false(v_camera_cv.y) };
                }

                bb_min.x = fminf(bb_min.x, v_pixel.x);
                bb_min.y = fminf(bb_min.y, v_pixel.y);
                bb_max.x = fmaxf(bb_max.x, v_pixel.x);
                bb_max.y = fmaxf(bb_max.y, v_pixel.y);
            }

            const auto ROUND_DOWN = -0.5f;
            const auto ROUND_UP = 0.5f;

            auto bb_min_rounded = glm::i64vec2{ roundf(bb_min.x + ROUND_DOWN), roundf(bb_min.y + ROUND_DOWN) };
            auto bb_max_rounded = glm::i64vec2{ roundf(bb_max.x + ROUND_UP), roundf(bb_max.y + ROUND_UP) };

            auto bb_min_border = glm::i64vec2{ glm::clamp<int64_t>(bb_min_rounded.x, 0, W - 1),
                                               glm::clamp<int64_t>(bb_min_rounded.y, 0, H - 1) };
            auto bb_max_border = glm::i64vec2{ glm::clamp<int64_t>(bb_max_rounded.x, 0, W - 1),
                                               glm::clamp<int64_t>(bb_max_rounded.y, 0, H - 1) };

            auto area_bb = (bb_max_border.y - bb_min_border.y + 1) * (bb_max_border.x - bb_min_border.x + 1);

            auto integral_mask_00 =
                    sample_zeros_padding(integral_masks, bb_min_border.y - 1, bb_min_border.x - 1, batch, 0);
            auto integral_mask_10 =
                    sample_zeros_padding(integral_masks, bb_max_border.y, bb_min_border.x - 1, batch, 0);
            auto integral_mask_01 =
                    sample_zeros_padding(integral_masks, bb_min_border.y - 1, bb_max_border.x, batch, 0);
            auto integral_mask_11 = sample_zeros_padding(integral_masks, bb_max_border.y, bb_max_border.x, batch, 0);

            auto integral_bb = integral_mask_11 + integral_mask_00 - integral_mask_10 - integral_mask_01;

            // NOTE: Due to the large range of sizes, numerical errors may quickly build up
            const auto epsilon = 1e-1f;

            CUDA_DEVICE_CHECK(integral_bb >= 0.f - epsilon);
            CUDA_DEVICE_CHECK(integral_bb <= static_cast<float>(area_bb) + epsilon);

            // Take the (image) isolevel into account when evaluating the accumulated mask values
            const auto isolevel = 0.5f;
            const float margin_isosurface = isolevel - epsilon;

            if (integral_bb <= 0.f + margin_isosurface)
            {
                is_empty = true;
            }
            else if (integral_bb >= static_cast<float>(area_bb) - margin_isosurface)
            {
                is_object = true;
            }
            else
            {
                should_refine = true;
            }
        }

        occupied_voxel[tid] = (should_refine && !is_empty);
    }
}

template <typename TransformT>
__global__ void
classify_children_partial_COPY(const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> candidates,
                               const torch::PackedTensorAccessor64<float, 4, torch::RestrictPtrTraits> integral_masks,
                               const torch::PackedTensorAccessor64<TransformT, 3, torch::RestrictPtrTraits> transforms,
                               const bool transforms_in_opengl,
                               const glm::i64vec3 resolution,
                               const glm::i64vec3 resolution_children,
                               const glm::vec3 cube_corner_bfl,
                               const float cube_length,
                               const bool last_children,
                               torch::PackedTensorAccessor64<uint8_t, 1, torch::RestrictPtrTraits> occupied_voxel)
{
    const auto N = occupied_voxel.size(0);

    // Note: image has dims (N, H, W, C) instead of (N, C, H, W)
    const auto H = integral_masks.size(1);
    const auto W = integral_masks.size(2);

    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for (auto tid = id; tid < N; tid += num_threads)
    {
        auto candidate_id = tid / 8;
        auto child_id = tid % 8;

        auto g = unravel_index(candidates[candidate_id], resolution);
        auto g_child = cube_vertex(int64_t{ 2 } * g, child_id);

        auto is_empty = false;
        [[maybe_unused]] auto is_object = false;
        auto should_refine = false;
        // For partial masks, we assume an overlap at the boundaries so all voxels fully lie within at least one image
        auto fully_inside_one_frame = false;
        for (auto batch = int64_t{ 0 }; batch < integral_masks.size(0); ++batch)
        {
            auto bb_min = glm::vec2{ FLT_MAX, FLT_MAX };
            auto bb_max = glm::vec2{ -FLT_MAX, -FLT_MAX };
            auto fully_inside = true;
            for (auto i = 0; i < 8; ++i)
            {
                auto v = cube_vertex(g_child, i);
                auto v_world =
                        glm::vec3{ cube_corner_bfl.x + static_cast<float>(v.x) /
                                                               static_cast<float>(resolution_children.x) * cube_length,
                                   cube_corner_bfl.y + static_cast<float>(v.y) /
                                                               static_cast<float>(resolution_children.y) * cube_length,
                                   cube_corner_bfl.z + static_cast<float>(v.z) /
                                                               static_cast<float>(resolution_children.z) *
                                                               cube_length };

                auto v_camera = bmm_4x4_transforms_COPY(v_world, transforms, batch);

                auto v_pixel = glm::vec2{};
                if (transforms_in_opengl)
                {
                    auto v_camera_ndc = glm::vec2{ v_camera.x / v_camera.w, v_camera.y / v_camera.w };
                    v_pixel = glm::vec2{ unnormalize_ndc_false(v_camera_ndc.x, W),
                                         unnormalize_ndc_false(v_camera_ndc.y, H) };
                }
                else
                {
                    auto v_camera_cv = glm::vec2{ v_camera.x / v_camera.z, v_camera.y / v_camera.z };
                    v_pixel = glm::vec2{ align_cv_false(v_camera_cv.x), align_cv_false(v_camera_cv.y) };
                }

                bb_min.x = fminf(bb_min.x, v_pixel.x);
                bb_min.y = fminf(bb_min.y, v_pixel.y);
                bb_max.x = fmaxf(bb_max.x, v_pixel.x);
                bb_max.y = fmaxf(bb_max.y, v_pixel.y);

                auto v_pixel_rounded = glm::i64vec2{ roundf(v_pixel.x), roundf(v_pixel.y) };
                if (!in_image(v_pixel_rounded.y, v_pixel_rounded.x, H, W, 1))
                {
                    fully_inside = false;
                }
            }

            const auto ROUND_DOWN = -0.5f;
            const auto ROUND_UP = 0.5f;

            auto bb_min_rounded = glm::i64vec2{ roundf(bb_min.x + ROUND_DOWN), roundf(bb_min.y + ROUND_DOWN) };
            auto bb_max_rounded = glm::i64vec2{ roundf(bb_max.x + ROUND_UP), roundf(bb_max.y + ROUND_UP) };

            auto bb_min_border = glm::i64vec2{ glm::clamp<int64_t>(bb_min_rounded.x, 0, W - 1),
                                               glm::clamp<int64_t>(bb_min_rounded.y, 0, H - 1) };
            auto bb_max_border = glm::i64vec2{ glm::clamp<int64_t>(bb_max_rounded.x, 0, W - 1),
                                               glm::clamp<int64_t>(bb_max_rounded.y, 0, H - 1) };

            auto area_bb = (bb_max_border.y - bb_min_border.y + 1) * (bb_max_border.x - bb_min_border.x + 1);
            auto full_area_bb = (bb_max_rounded.y - bb_min_rounded.y + 1) * (bb_max_rounded.x - bb_min_rounded.x + 1);

            auto integral_mask_00 =
                    sample_zeros_padding(integral_masks, bb_min_border.y - 1, bb_min_border.x - 1, batch, 0);
            auto integral_mask_10 =
                    sample_zeros_padding(integral_masks, bb_max_border.y, bb_min_border.x - 1, batch, 0);
            auto integral_mask_01 =
                    sample_zeros_padding(integral_masks, bb_min_border.y - 1, bb_max_border.x, batch, 0);
            auto integral_mask_11 = sample_zeros_padding(integral_masks, bb_max_border.y, bb_max_border.x, batch, 0);

            auto integral_bb = integral_mask_11 + integral_mask_00 - integral_mask_10 - integral_mask_01;

            // NOTE: Due to the large range of sizes, numerical errors may quickly build up
            const auto epsilon = 1e-1f;

            CUDA_DEVICE_CHECK(integral_bb >= 0.f - epsilon);
            CUDA_DEVICE_CHECK(integral_bb <= static_cast<float>(area_bb) + epsilon);

            // Take the (image) isolevel into account when evaluating the accumulated mask values
            const auto isolevel = 0.5f;
            const float margin_isosurface = isolevel - epsilon;

            if (integral_bb <= 0.f + margin_isosurface && area_bb == full_area_bb)
            {
                is_empty = true;
            }
            else if (integral_bb >= static_cast<float>(area_bb) - margin_isosurface && area_bb == full_area_bb)
            {
                is_object = true;
            }
            else if (bb_max_border.y - bb_min_border.y > 1 && bb_max_border.x - bb_min_border.x > 1)
            {
                should_refine = true;
            }

            fully_inside_one_frame |= fully_inside;
        }

        occupied_voxel[tid] = (should_refine && !is_empty && (!last_children || fully_inside_one_frame));
    }
}

template <typename T>
inline C10_DEVICE bool
intervals_intersect(const T x_start, const T x_end, const T y_start, const T y_end)
{
    return x_start <= y_end     // x not behind y
           && y_start <= x_end; // y not before x
}

template <typename TransformT>
__global__ void
append_classify_children_depths(
        const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> candidates,
        const torch::PackedTensorAccessor64<float, 4, torch::RestrictPtrTraits> depths,
        const torch::PackedTensorAccessor64<float, 4, torch::RestrictPtrTraits> integral_depths,
        const torch::PackedTensorAccessor64<float, 4, torch::RestrictPtrTraits> integral_depth_masks,
        const torch::PackedTensorAccessor64<TransformT, 3, torch::RestrictPtrTraits> transforms,
        const bool transforms_in_opengl,
        const glm::i64vec3 resolution,
        const glm::i64vec3 resolution_children,
        const glm::vec3 cube_corner_bfl,
        const float cube_length,
        const float truncation_region,
        const glm::i64vec3 max_resolution,
        torch::PackedTensorAccessor64<uint8_t, 1, torch::RestrictPtrTraits> occupied_voxel)
{
    const auto N = occupied_voxel.size(0);

    // Note: image has dims (N, H, W, C) instead of (N, C, H, W)
    const auto H = depths.size(1);
    const auto W = depths.size(2);

    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for (auto tid = id; tid < N; tid += num_threads)
    {
        auto candidate_id = tid / 8;
        auto child_id = tid % 8;

        auto g = unravel_index(candidates[candidate_id], resolution);
        auto g_child = cube_vertex(int64_t{ 2 } * g, child_id);

        auto should_refine = false;
        for (auto batch = int64_t{ 0 }; batch < depths.size(0); ++batch)
        {
            auto bb_min = glm::vec3{ FLT_MAX, FLT_MAX, FLT_MAX };
            auto bb_max = glm::vec3{ -FLT_MAX, -FLT_MAX, -FLT_MAX };
            for (auto i = 0; i < 8; ++i)
            {
                auto v = cube_vertex(g_child, i);
                auto v_world =
                        glm::vec3{ cube_corner_bfl.x + static_cast<float>(v.x) /
                                                               static_cast<float>(resolution_children.x) * cube_length,
                                   cube_corner_bfl.y + static_cast<float>(v.y) /
                                                               static_cast<float>(resolution_children.y) * cube_length,
                                   cube_corner_bfl.z + static_cast<float>(v.z) /
                                                               static_cast<float>(resolution_children.z) *
                                                               cube_length };

                auto v_camera = bmm_4x4_transforms_COPY(v_world, transforms, batch);

                auto v_pixel = glm::vec2{};
                auto v_depth = 0.f;
                if (transforms_in_opengl)
                {
                    auto v_camera_ndc = glm::vec2{ v_camera.x / v_camera.w, v_camera.y / v_camera.w };
                    v_pixel = glm::vec2{ unnormalize_ndc_false(v_camera_ndc.x, W),
                                         unnormalize_ndc_false(v_camera_ndc.y, H) };
                    v_depth = v_camera.w;
                }
                else
                {
                    auto v_camera_cv = glm::vec2{ v_camera.x / v_camera.z, v_camera.y / v_camera.z };
                    v_pixel = glm::vec2{ align_cv_false(v_camera_cv.x), align_cv_false(v_camera_cv.y) };
                    v_depth = v_camera.z;
                }

                bb_min.x = fminf(bb_min.x, v_pixel.x);
                bb_min.y = fminf(bb_min.y, v_pixel.y);
                bb_min.z = fminf(bb_min.z, v_depth);
                bb_max.x = fmaxf(bb_max.x, v_pixel.x);
                bb_max.y = fmaxf(bb_max.y, v_pixel.y);
                bb_max.z = fmaxf(bb_max.z, v_depth);
            }

            const auto ROUND_DOWN = -0.5f;
            const auto ROUND_UP = 0.5f;

            auto bb_min_rounded = glm::i64vec2{ roundf(bb_min.x + ROUND_DOWN), roundf(bb_min.y + ROUND_DOWN) };
            auto bb_max_rounded = glm::i64vec2{ roundf(bb_max.x + ROUND_UP), roundf(bb_max.y + ROUND_UP) };

            auto bb_min_border = glm::i64vec2{ glm::clamp<int64_t>(bb_min_rounded.x, 0, W - 1),
                                               glm::clamp<int64_t>(bb_min_rounded.y, 0, H - 1) };
            auto bb_max_border = glm::i64vec2{ glm::clamp<int64_t>(bb_max_rounded.x, 0, W - 1),
                                               glm::clamp<int64_t>(bb_max_rounded.y, 0, H - 1) };

            auto integral_depth_00 =
                    sample_zeros_padding(integral_depths, bb_min_border.y - 1, bb_min_border.x - 1, batch, 0);
            auto integral_depth_10 =
                    sample_zeros_padding(integral_depths, bb_max_border.y, bb_min_border.x - 1, batch, 0);
            auto integral_depth_01 =
                    sample_zeros_padding(integral_depths, bb_min_border.y - 1, bb_max_border.x, batch, 0);
            auto integral_depth_11 = sample_zeros_padding(integral_depths, bb_max_border.y, bb_max_border.x, batch, 0);

            auto integral_depth = integral_depth_11 + integral_depth_00 - integral_depth_10 - integral_depth_01;

            auto integral_depth_mask_00 =
                    sample_zeros_padding(integral_depth_masks, bb_min_border.y - 1, bb_min_border.x - 1, batch, 0);
            auto integral_depth_mask_10 =
                    sample_zeros_padding(integral_depth_masks, bb_max_border.y, bb_min_border.x - 1, batch, 0);
            auto integral_depth_mask_01 =
                    sample_zeros_padding(integral_depth_masks, bb_min_border.y - 1, bb_max_border.x, batch, 0);
            auto integral_depth_mask_11 =
                    sample_zeros_padding(integral_depth_masks, bb_max_border.y, bb_max_border.x, batch, 0);

            auto integral_depth_mask =
                    integral_depth_mask_11 + integral_depth_mask_00 - integral_depth_mask_10 - integral_depth_mask_01;

            auto mean_depth = integral_depth / fmaxf(integral_depth_mask, 1e-5);

            // NOTE: Due to the large range of sizes, numerical errors may quickly build up
            const auto epsilon = 1e-1f;

            if (mean_depth > 0.f + epsilon)
            {
                auto level_scale = static_cast<float>(max_resolution.x / resolution.x);

                auto band_min = mean_depth - truncation_region * level_scale;
                auto band_max = mean_depth + truncation_region * level_scale;

                if (intervals_intersect(bb_min.z, bb_max.z, band_min, band_max))
                {
                    should_refine = true;
                }
            }
        }

        occupied_voxel[tid] |= should_refine; // Enable more indices if necessary
    }
}

template <typename MaskT, typename TransformT>
__global__ void
accumulate_hull_counts_full_COPY(
        const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> sparse_indices,
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

    // Note: image has dims (N, H, W, C) instead of (N, C, H, W)
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

        auto g_camera = bmm_4x4_transforms_COPY(g_world, transforms, batch);

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
accumulate_hull_counts_partial_COPY(
        const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> sparse_indices,
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

    // Note: image has dims (N, H, W, C) instead of (N, C, H, W)
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

        auto g_camera = bmm_4x4_transforms_COPY(g_world, transforms, batch);

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

        // For partial masks, only accumulate valid values (no interpolation across the boundary)
        auto g_pixel_rounded = glm::i64vec2{ roundf(g_pixel.x), roundf(g_pixel.y) };
        if (in_image(g_pixel_rounded.y, g_pixel_rounded.x, H, W, 1))
        {
            sparse_values[tid] *= sample_bilinear_mode_ones_padding(masks, g_pixel.y, g_pixel.x, batch, 0);
        }
    }
}

template <typename TransformT>
__global__ void
accumulate_constrained_tsdf(const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> sparse_indices,
                            const torch::PackedTensorAccessor64<float, 1, torch::RestrictPtrTraits> sparse_values,
                            const int64_t N,
                            const torch::PackedTensorAccessor64<float, 4, torch::RestrictPtrTraits> depths,
                            const torch::PackedTensorAccessor64<TransformT, 3, torch::RestrictPtrTraits> transforms,
                            const bool transforms_in_opengl,
                            const glm::i64vec3 resolution_grid,
                            const glm::vec3 cube_corner_bfl,
                            const float cube_length,
                            const float truncation_region,
                            const int64_t batch,
                            torch::PackedTensorAccessor64<float, 1, torch::RestrictPtrTraits> final_sparse_values,
                            torch::PackedTensorAccessor64<float, 1, torch::RestrictPtrTraits> final_sparse_weights)
{
    const auto resolution_cells = glm::i64vec3{ resolution_grid.x - 1, resolution_grid.y - 1, resolution_grid.z - 1 };

    // Note: image has dims (N, H, W, C) instead of (N, C, H, W)
    const auto H = depths.size(1);
    const auto W = depths.size(2);

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

        auto g_camera = bmm_4x4_transforms_COPY(g_world, transforms, batch);

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

        /*
        if (tid % 1000 == 0)
        {
            printf("(%f %f %f), (%f %f), (%ld %ld)\n",
                   g_camera.x,
                   g_camera.y,
                   g_camera.z,
                   g_pixel.x,
                   g_pixel.y,
                   g_pixel_rounded.x,
                   g_pixel_rounded.y);
        }
        */

        if (in_image(g_pixel_rounded.y, g_pixel_rounded.x, H, W) && g_camera.z > 0.f)
        {
            // This also interpolates across holes
            auto depth = sample_bilinear_mode_zeros_padding(depths, g_pixel.y, g_pixel.x, batch, 0);
            // auto depth = depths[batch][g_pixel_rounded.y][g_pixel_rounded.x][0];
            auto sdf = depth - g_camera.z;

            auto hull_value = sparse_values[tid];

            auto tsdf = float{ 0.f };
            auto w = float{ 0.f };

            bool violated = false;
            // violated = "behind depth surface" and "in front of visual hull"
            if (sdf < 0.f && hull_value < 0.5f)
            {
                violated = true;
            }

            if (depth > 0.f && sdf > -truncation_region && !violated)
            {
                tsdf = glm::clamp<float>(sdf / truncation_region, -1.f, 1.f);
                w = glm::clamp(
                        1.f + tsdf,
                        0.f,
                        1.f); // tsdf in [0, inf) -> 1, tsdf in [-1, 0] -> linear decreasing, tsdf in (-inf, -1] -> 0
            }
            else
            {
                // Remap [0, 1] to [-1, 1] by: 0 -> 1, 1 -> -1
                tsdf = -2.f * hull_value + 1.f;
                w = 0.1f;
            }

            auto TSDF = final_sparse_values[tid];
            auto W = final_sparse_weights[tid];

            final_sparse_values[tid] = (W * TSDF + w * tsdf) / fmaxf(W + w, 1e-5);
            final_sparse_weights[tid] = W + w;
        }
    }
}

template <typename TransformT>
__global__ void
accumulate_tsdf(const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> sparse_indices,
                const int64_t N,
                const torch::PackedTensorAccessor64<float, 4, torch::RestrictPtrTraits> depths,
                const torch::PackedTensorAccessor64<TransformT, 3, torch::RestrictPtrTraits> transforms,
                const bool transforms_in_opengl,
                const glm::i64vec3 resolution_grid,
                const glm::vec3 cube_corner_bfl,
                const float cube_length,
                const float truncation_region,
                const int64_t batch,
                torch::PackedTensorAccessor64<float, 1, torch::RestrictPtrTraits> tsdf_sparse_values,
                torch::PackedTensorAccessor64<float, 1, torch::RestrictPtrTraits> tsdf_sparse_weights)
{
    const auto resolution_cells = glm::i64vec3{ resolution_grid.x - 1, resolution_grid.y - 1, resolution_grid.z - 1 };

    // Note: image has dims (N, H, W, C) instead of (N, C, H, W)
    const auto H = depths.size(1);
    const auto W = depths.size(2);

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

        auto g_camera = bmm_4x4_transforms_COPY(g_world, transforms, batch);

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

        /*
        if (tid % 1000 == 0)
        {
            printf("(%f %f %f), (%f %f), (%ld %ld)\n",
                   g_camera.x,
                   g_camera.y,
                   g_camera.z,
                   g_pixel.x,
                   g_pixel.y,
                   g_pixel_rounded.x,
                   g_pixel_rounded.y);
        }
        */

        if (in_image(g_pixel_rounded.y, g_pixel_rounded.x, H, W) && g_camera.z > 0.f)
        {
            // This also interpolates across holes
            auto depth = sample_bilinear_mode_zeros_padding(depths, g_pixel.y, g_pixel.x, batch, 0);
            // auto depth = depths[batch][g_pixel_rounded.y][g_pixel_rounded.x][0];
            auto sdf = depth - g_camera.z;

            if (depth > 0.f && sdf > -truncation_region)
            {
                auto tsdf = glm::clamp<float>(sdf / truncation_region, -1.f, 1.f);
                auto w = glm::clamp(
                        1.f + tsdf,
                        0.f,
                        1.f); // tsdf in [0, inf) -> 1, tsdf in [-1, 0] -> linear decreasing, tsdf in (-inf, -1] -> 0

                auto TSDF = tsdf_sparse_values[tid];
                auto W = tsdf_sparse_weights[tid];

                /*
                if (tid % 10000 == 0)
                {
                    printf("%d: (%f %f), (%f %f)\n", batch, sdf, tsdf, TSDF, W);
                }
                */

                tsdf_sparse_values[tid] = (W * TSDF + w * tsdf) / fmaxf(W + w, 1e-5);
                tsdf_sparse_weights[tid] = W + w;
            }
        }
    }
}

template <typename TransformT>
inline glm::vec3 C10_DEVICE
unproject(const float x,
          const float y,
          const float depth,
          const torch::PackedTensorAccessor64<TransformT, 3, torch::RestrictPtrTraits> transforms_inv,
          const bool transforms_in_opengl,
          const int batch)
{
    auto vertex = glm::vec4{};
    if (transforms_in_opengl)
    {
        assert(false && "Not doable with transforms_inv only");
    }
    else
    {
        vertex = bmm_4x4_transforms_COPY(glm::vec4{ x * depth, y * depth, depth, 1.f }, transforms_inv, batch);
    }

    return glm::vec3{ vertex };
}

template <typename TransformT>
__global__ void
normal_map_kernel(const torch::PackedTensorAccessor64<float, 4, torch::RestrictPtrTraits> depths,
                  const torch::PackedTensorAccessor64<TransformT, 3, torch::RestrictPtrTraits> transforms_inv,
                  const bool transforms_in_opengl,
                  torch::PackedTensorAccessor64<float, 4, torch::RestrictPtrTraits> normals)
{

    // Note: image has dims (N, H, W, C) instead of (N, C, H, W)
    const auto sizes = glm::i64vec3{ depths.size(2), depths.size(1), depths.size(0) };
    const auto N = numel(sizes);

    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for (auto tid = id; tid < N; tid += num_threads)
    {
        auto p = unravel_index(tid, sizes);

        auto d_x_m1 = sample_zeros_padding(depths, p.y, p.x - 1, p.z, 0);
        auto d_x_p1 = sample_zeros_padding(depths, p.y, p.x + 1, p.z, 0);
        auto d_y_m1 = sample_zeros_padding(depths, p.y - 1, p.x, p.z, 0);
        auto d_y_p1 = sample_zeros_padding(depths, p.y + 1, p.x, p.z, 0);

        if (d_x_m1 > 0.f && d_x_p1 > 0.f && d_y_m1 > 0.f && d_y_p1 > 0.f)
        {
            auto v_x_m1 = unproject(p.x - 1, p.y, d_x_m1, transforms_inv, transforms_in_opengl, p.z);
            auto v_x_p1 = unproject(p.x + 1, p.y, d_x_p1, transforms_inv, transforms_in_opengl, p.z);
            auto v_y_m1 = unproject(p.x, p.y - 1, d_y_m1, transforms_inv, transforms_in_opengl, p.z);
            auto v_y_p1 = unproject(p.x, p.y + 1, d_y_p1, transforms_inv, transforms_in_opengl, p.z);

            auto normal = glm::normalize(glm::cross(v_x_p1 - v_x_m1, v_y_p1 - v_y_m1));

            normals[p.z][p.y][p.x][0] = normal.x;
            normals[p.z][p.y][p.x][1] = normal.y;
            normals[p.z][p.y][p.x][2] = normal.z;
        }
        else
        {
            normals[p.z][p.y][p.x][0] = 0.f;
            normals[p.z][p.y][p.x][1] = 0.f;
            normals[p.z][p.y][p.x][2] = 0.f;
        }
    }
}

torch::Tensor
normal_map(const torch::Tensor& depths, const torch::Tensor& transforms, const bool transforms_in_opengl)
{
    at::cuda::CUDAGuard device_guard{ depths.device() };
    const auto stream = at::cuda::getCurrentCUDAStream();

    const auto dtype_float = torch::TensorOptions{}.dtype(torch::kFloat32).device(depths.device());

    auto normals = torch::empty({ depths.size(0), depths.size(1), depths.size(2), 3 }, dtype_float);
    auto transforms_inv = transforms.inverse();

    // std::cout << transforms.sizes() << transforms_inv.sizes() << std::endl;

    auto depths_ = depths.packed_accessor64<float, 4, torch::RestrictPtrTraits>();
    auto normals_ = normals.packed_accessor64<float, 4, torch::RestrictPtrTraits>();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            transforms_inv.scalar_type(),
            "normal_map_kernel",
            [&]()
            {
                auto transforms_inv_ = transforms_inv.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>();

                const int threads_per_block = 128;
                dim3 grid;
                at::cuda::getApplyGrid(numel({ depths.size(2), depths.size(1), depths.size(0) }),
                                       grid,
                                       depths.device().index(),
                                       threads_per_block);
                dim3 threads = at::cuda::getApplyBlock(threads_per_block);

                normal_map_kernel<<<grid, threads, 0, stream>>>(depths_,
                                                                transforms_inv_,
                                                                transforms_in_opengl,
                                                                normals_);
                AT_CUDA_CHECK(cudaGetLastError());
                AT_CUDA_CHECK(cudaStreamSynchronize(stream));
            });

    return normals;
}

// Forward declare instead of copy-pasting it
torch::Tensor
integral_image(const torch::Tensor& self, c10::ScalarType dtype);

torch::Tensor
to_sparse_coo_tensor(const RavelledSparseTensor& ravelled_tensor);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::vector<torch::Tensor>>
tsdf_fusion_cuda(const torch::Tensor& masks,
                 const torch::Tensor& transforms_masks_cv,
                 const torch::Tensor& depths,
                 const torch::Tensor& transforms_depths_cv,
                 const int level,
                 const std::array<float, 3>& cube_corner_bfl,
                 const float cube_length,
                 const bool masks_partial,
                 const float truncation_region)
{
    TORCH_CHECK_EQ(depths.device(), transforms_depths_cv.device());
    TORCH_CHECK_EQ(depths.dim(), 4);
    TORCH_CHECK_EQ(transforms_depths_cv.dim(), 3);
    TORCH_CHECK_EQ(transforms_depths_cv.size(1), 4);
    TORCH_CHECK_EQ(transforms_depths_cv.size(2), 4);
    TORCH_CHECK_EQ(depths.size(0), depths.size(0));
    TORCH_CHECK_EQ(depths.size(3), 1);
    TORCH_CHECK_GE(level, 0);
    TORCH_CHECK_GT(cube_length, 0.f);

    CudaTimer timer;

    const auto transforms_in_opengl = false;

    at::cuda::CUDAGuard device_guard{ depths.device() };
    const auto stream = at::cuda::getCurrentCUDAStream();

    at::cuda::ThrustAllocator allocator;
    const auto policy = thrust::cuda::par(allocator).on(stream);

    const auto dtype_uint8 = torch::TensorOptions{}.dtype(torch::kUInt8).device(depths.device());
    const auto dtype_int64 = torch::TensorOptions{}.dtype(torch::kInt64).device(depths.device());
    const auto dtype_float = torch::TensorOptions{}.dtype(torch::kFloat32).device(depths.device());

    timer.start(stream);

    auto integral_masks = integral_image(masks, torch::kFloat32);
    auto integral_depths = integral_image(depths, torch::kFloat32);
    auto integral_depth_masks = integral_image((depths != 0).to(torch::kFloat32), torch::kFloat32);

    timer.stop("Integral images", stream);

    timer.start(stream);

    auto normals = normal_map(depths, transforms_depths_cv, transforms_in_opengl);

    timer.stop("Normal Map     ", stream);

    timer.start(stream);

    auto candidates = torch::tensor({ 0 }, dtype_int64);
    auto candidates_octree = std::vector<torch::Tensor>{};
    candidates_octree.push_back(candidates);

    auto cube_corner_bfl_cuda = glm::vec3{ cube_corner_bfl[0], cube_corner_bfl[1], cube_corner_bfl[2] };

    // 1. Hierarchically compute sparse cells
    for (int i = 0; i < level; ++i)
    {
        const auto N = 8 * candidates.numel();
        const auto resolution = glm::i64vec3{ 1 << i };
        const auto resolution_children = glm::i64vec3{ 1 << (i + 1) };

        auto occupied_voxel = torch::zeros({ N }, dtype_uint8).contiguous();

        auto candidates_ = candidates.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>();
        auto integral_masks_ = integral_masks.packed_accessor64<float, 4, torch::RestrictPtrTraits>();
        auto occupied_voxel_ = occupied_voxel.packed_accessor64<uint8_t, 1, torch::RestrictPtrTraits>();

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                transforms_masks_cv.scalar_type(),
                "classify_children",
                [&]()
                {
                    auto transforms_masks_cv_ =
                            transforms_masks_cv.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>();

                    const int threads_per_block = 128;
                    dim3 grid_volume;
                    at::cuda::getApplyGrid(N, grid_volume, masks.device().index(), threads_per_block);
                    dim3 threads = at::cuda::getApplyBlock(threads_per_block);

                    if (masks_partial)
                    {
                        classify_children_partial_COPY<<<grid_volume, threads, 0, stream>>>(candidates_,
                                                                                            integral_masks_,
                                                                                            transforms_masks_cv_,
                                                                                            transforms_in_opengl,
                                                                                            resolution,
                                                                                            resolution_children,
                                                                                            cube_corner_bfl_cuda,
                                                                                            cube_length,
                                                                                            i == level - 1,
                                                                                            occupied_voxel_);
                        AT_CUDA_CHECK(cudaGetLastError());
                        AT_CUDA_CHECK(cudaStreamSynchronize(stream));
                    }
                    else
                    {
                        classify_children_full_COPY<<<grid_volume, threads, 0, stream>>>(candidates_,
                                                                                         integral_masks_,
                                                                                         transforms_masks_cv_,
                                                                                         transforms_in_opengl,
                                                                                         resolution,
                                                                                         resolution_children,
                                                                                         cube_corner_bfl_cuda,
                                                                                         cube_length,
                                                                                         occupied_voxel_);
                        AT_CUDA_CHECK(cudaGetLastError());
                        AT_CUDA_CHECK(cudaStreamSynchronize(stream));
                    }
                });

        // TODO Should we dispatch this ???
        auto depths_ = depths.packed_accessor64<float, 4, torch::RestrictPtrTraits>();
        auto integral_depths_ = integral_depths.packed_accessor64<float, 4, torch::RestrictPtrTraits>();
        auto integral_depth_masks_ = integral_depth_masks.packed_accessor64<float, 4, torch::RestrictPtrTraits>();

        auto max_resolution = glm::i64vec3{ 1 << (level - 1) };

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                transforms_depths_cv.scalar_type(),
                "classify_children",
                [&]()
                {
                    auto transforms_depths_cv_ =
                            transforms_depths_cv.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>();

                    const int threads_per_block = 128;
                    dim3 grid_volume;
                    at::cuda::getApplyGrid(N, grid_volume, depths.device().index(), threads_per_block);
                    dim3 threads = at::cuda::getApplyBlock(threads_per_block);

                    append_classify_children_depths<<<grid_volume, threads, 0, stream>>>(candidates_,
                                                                                         depths_,
                                                                                         integral_depths_,
                                                                                         integral_depth_masks_,
                                                                                         transforms_depths_cv_,
                                                                                         transforms_in_opengl,
                                                                                         resolution,
                                                                                         resolution_children,
                                                                                         cube_corner_bfl_cuda,
                                                                                         cube_length,
                                                                                         truncation_region,
                                                                                         max_resolution,
                                                                                         occupied_voxel_);
                    AT_CUDA_CHECK(cudaGetLastError());
                    AT_CUDA_CHECK(cudaStreamSynchronize(stream));
                });

        auto new_candidates = torch::empty({ N }, dtype_int64).contiguous();
        {
            void* d_temp_storage = nullptr;
            size_t temp_storage_bytes = 0;
            auto num_selected_out = torch::empty({ 1 }, dtype_int64).contiguous();

            // cuda::proclaim_return_type may require a higher CUDA version on Windows, so use this hacky version
            // instead
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

            // Flagged is limited to 32-bit indices at least up to cub 2.6
            TORCH_CHECK_LT(N, (static_cast<int64_t>(1) << 31));

            AT_CUDA_CHECK(cub::DeviceSelect::Flagged(
                    d_temp_storage,
                    temp_storage_bytes,
                    thrust::make_transform_iterator(thrust::counting_iterator<int64_t>(0), f),
                    occupied_voxel.data_ptr<uint8_t>(),
                    new_candidates.data_ptr<int64_t>(),
                    num_selected_out.data_ptr<int64_t>(),
                    N,
                    stream));
            AT_CUDA_CHECK(cudaStreamSynchronize(stream));

            auto temp_storage = torch::empty({ static_cast<int64_t>(temp_storage_bytes) }, dtype_uint8).contiguous();

            AT_CUDA_CHECK(cub::DeviceSelect::Flagged(
                    temp_storage.data_ptr(),
                    temp_storage_bytes,
                    thrust::make_transform_iterator(thrust::counting_iterator<int64_t>(0), f),
                    occupied_voxel.data_ptr<uint8_t>(),
                    new_candidates.data_ptr<int64_t>(),
                    num_selected_out.data_ptr<int64_t>(),
                    N,
                    stream));
            AT_CUDA_CHECK(cudaStreamSynchronize(stream));

            new_candidates.resize_({ num_selected_out.cpu().item<int64_t>() });
        }

        candidates = new_candidates;
        candidates_octree.push_back(candidates);

        // std::cout << i << ": " << N / 8 << " " << new_candidates.numel() << std::endl;

        if (candidates.numel() == 0)
        {
            break;
        }
    }

    const auto resolution_cells = glm::i64vec3{ 1 << level };
    const auto resolution_grid = resolution_cells + int64_t{ 1 };

    if (candidates.numel() == 0)
    {
        auto sparse_indices = torch::empty({ 4, 0 }, dtype_int64);
        auto sparse_values = torch::empty({ 0 }, dtype_float);

        auto field = torch::sparse_coo_tensor(sparse_indices,
                                              sparse_values,
                                              { 1, resolution_grid.x, resolution_grid.y, resolution_grid.z },
                                              dtype_float)
                             .coalesce();

        return { normals, field, field, candidates_octree };
    }

    timer.stop("Candidates     ", stream);

    timer.start(stream);

    // 2. Convert sparse cells to sparse grid indices
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

    auto N = int64_t{ 0 };
    {
        // Unique is limited to 32-bit indices, at least up to cub 2.6
        TORCH_CHECK_LT(8 * candidates.numel(), (static_cast<int64_t>(1) << 31));

        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        auto num_selected_out = torch::empty({ 1 }, dtype_int64).contiguous();

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

        N = num_selected_out.cpu().item<int64_t>();
    }

    sparse_indices.resize_({ N });

    // Release no longer needed tensors early to reduce memory pressure
    all_corners = torch::Tensor{};

    timer.stop("Indices        ", stream);

    timer.start(stream);

    // 3. Compute sparse hull counts
    auto sparse_values = torch::ones({ N }, dtype_float);

    auto sparse_indices_ = sparse_indices.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>();
    auto sparse_values_ = sparse_values.packed_accessor64<float, 1, torch::RestrictPtrTraits>();
    AT_DISPATCH_ALL_TYPES_AND(
            torch::ScalarType::Half,
            masks.scalar_type(),
            "accumulate_hull_counts",
            [&]()
            {
                auto masks_ = masks.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>();

                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                        transforms_masks_cv.scalar_type(),
                        "accumulate_hull_counts",
                        [&]()
                        {
                            auto transforms_masks_cv_ =
                                    transforms_masks_cv.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>();

                            for (auto batch = int64_t{ 0 }; batch < masks.size(0); ++batch)
                            {
                                const int threads_per_block = 128;
                                dim3 grid_corners;
                                at::cuda::getApplyGrid(N, grid_corners, masks.device().index(), threads_per_block);
                                dim3 threads = at::cuda::getApplyBlock(threads_per_block);

                                if (masks_partial)
                                {
                                    accumulate_hull_counts_partial_COPY<<<grid_corners, threads, 0, stream>>>(
                                            sparse_indices_,
                                            N,
                                            masks_,
                                            transforms_masks_cv_,
                                            transforms_in_opengl,
                                            resolution_grid,
                                            cube_corner_bfl_cuda,
                                            cube_length,
                                            batch,
                                            sparse_values_);
                                    AT_CUDA_CHECK(cudaGetLastError());
                                    AT_CUDA_CHECK(cudaStreamSynchronize(stream));
                                }
                                else
                                {
                                    accumulate_hull_counts_full_COPY<<<grid_corners, threads, 0, stream>>>(
                                            sparse_indices_,
                                            N,
                                            masks_,
                                            transforms_masks_cv_,
                                            transforms_in_opengl,
                                            resolution_grid,
                                            cube_corner_bfl_cuda,
                                            cube_length,
                                            batch,
                                            sparse_values_);
                                    AT_CUDA_CHECK(cudaGetLastError());
                                    AT_CUDA_CHECK(cudaStreamSynchronize(stream));
                                }
                            }
                        });
            });

    timer.stop("Hull values    ", stream);

    timer.start(stream);

    auto final_values = torch::empty({ N }, dtype_float);
    auto final_weights = torch::zeros({ N }, dtype_float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            transforms_depths_cv.scalar_type(),
            "accumulate_constrained_tsdf",
            [&]()
            {
                auto transforms_depths_cv_ =
                        transforms_depths_cv.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>();

                auto final_values_ = final_values.packed_accessor64<float, 1, torch::RestrictPtrTraits>();
                auto final_weights_ = final_weights.packed_accessor64<float, 1, torch::RestrictPtrTraits>();

                auto depths_ = depths.packed_accessor64<float, 4, torch::RestrictPtrTraits>();

                for (auto batch = int64_t{ 0 }; batch < depths.size(0); ++batch)
                {
                    const int threads_per_block = 128;
                    dim3 grid_corners;
                    at::cuda::getApplyGrid(N, grid_corners, depths.device().index(), threads_per_block);
                    dim3 threads = at::cuda::getApplyBlock(threads_per_block);

                    accumulate_constrained_tsdf<<<grid_corners, threads, 0, stream>>>(sparse_indices_,
                                                                                      sparse_values_,
                                                                                      N,
                                                                                      depths_,
                                                                                      transforms_depths_cv_,
                                                                                      transforms_in_opengl,
                                                                                      resolution_grid,
                                                                                      cube_corner_bfl_cuda,
                                                                                      cube_length,
                                                                                      truncation_region,
                                                                                      batch,
                                                                                      final_values_,
                                                                                      final_weights_);
                    AT_CUDA_CHECK(cudaGetLastError());
                    AT_CUDA_CHECK(cudaStreamSynchronize(stream));
                }
            });

    timer.stop("Merge fields   ", stream);

    timer.start(stream);

    // 3. Fuse TSDF values
    auto tsdf_sparse_values = torch::ones({ N }, dtype_float).contiguous();
    auto tsdf_sparse_weights = torch::zeros({ N }, dtype_float).contiguous();

    auto tsdf_sparse_values_ = tsdf_sparse_values.packed_accessor64<float, 1, torch::RestrictPtrTraits>();
    auto tsdf_sparse_weights_ = tsdf_sparse_weights.packed_accessor64<float, 1, torch::RestrictPtrTraits>();

    auto depths_ = depths.packed_accessor64<float, 4, torch::RestrictPtrTraits>();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            transforms_depths_cv.scalar_type(),
            "accumulate_tsdf",
            [&]()
            {
                auto transforms_depths_cv_ =
                        transforms_depths_cv.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>();

                for (auto batch = int64_t{ 0 }; batch < depths.size(0); ++batch)
                {
                    const int threads_per_block = 128;
                    dim3 grid_corners;
                    at::cuda::getApplyGrid(N, grid_corners, depths.device().index(), threads_per_block);
                    dim3 threads = at::cuda::getApplyBlock(threads_per_block);

                    accumulate_tsdf<<<grid_corners, threads, 0, stream>>>(sparse_indices_,
                                                                          N,
                                                                          depths_,
                                                                          transforms_depths_cv_,
                                                                          transforms_in_opengl,
                                                                          resolution_grid,
                                                                          cube_corner_bfl_cuda,
                                                                          cube_length,
                                                                          truncation_region,
                                                                          batch,
                                                                          tsdf_sparse_values_,
                                                                          tsdf_sparse_weights_);
                    AT_CUDA_CHECK(cudaGetLastError());
                    AT_CUDA_CHECK(cudaStreamSynchronize(stream));
                }
            });

    auto tsdf_sparse_indices_filtered = sparse_indices.clone();
    auto tsdf_sparse_values_filtered = tsdf_sparse_values.clone();
    {
        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        auto num_selected_out = torch::empty({ 1 }, dtype_int64).contiguous();

        auto valid_weight = [] C10_HOST_DEVICE(const float w) -> bool { return w > 0.f; };

        // FlaggedIf is limited to 32-bit indices at least up to cub 2.6
        TORCH_CHECK_LT(N, (static_cast<int64_t>(1) << 31));

        AT_CUDA_CHECK(cub::DeviceSelect::Flagged(
                d_temp_storage,
                temp_storage_bytes,
                thrust::make_zip_iterator(
                        thrust::make_tuple(sparse_indices.data_ptr<int64_t>(), tsdf_sparse_values.data_ptr<float>())),
                thrust::make_transform_iterator(tsdf_sparse_weights.data_ptr<float>(), valid_weight),
                thrust::make_zip_iterator(thrust::make_tuple(tsdf_sparse_indices_filtered.data_ptr<int64_t>(),
                                                             tsdf_sparse_values_filtered.data_ptr<float>())),
                num_selected_out.data_ptr<int64_t>(),
                N,
                stream));
        AT_CUDA_CHECK(cudaStreamSynchronize(stream));

        auto temp_storage = torch::empty({ static_cast<int64_t>(temp_storage_bytes) }, dtype_uint8).contiguous();

        AT_CUDA_CHECK(cub::DeviceSelect::Flagged(
                temp_storage.data_ptr(),
                temp_storage_bytes,
                thrust::make_zip_iterator(
                        thrust::make_tuple(sparse_indices.data_ptr<int64_t>(), tsdf_sparse_values.data_ptr<float>())),
                thrust::make_transform_iterator(tsdf_sparse_weights.data_ptr<float>(), valid_weight),
                thrust::make_zip_iterator(thrust::make_tuple(tsdf_sparse_indices_filtered.data_ptr<int64_t>(),
                                                             tsdf_sparse_values_filtered.data_ptr<float>())),
                num_selected_out.data_ptr<int64_t>(),
                N,
                stream));
        AT_CUDA_CHECK(cudaStreamSynchronize(stream));

        tsdf_sparse_indices_filtered.resize_({ num_selected_out.cpu().item<int64_t>() });
        tsdf_sparse_values_filtered.resize_({ num_selected_out.cpu().item<int64_t>() });

        N = num_selected_out.cpu().item<int64_t>();
    }

    timer.stop("[O] TSDF fusion", stream);

    auto tsdf_sparse_field = to_sparse_coo_tensor(
            RavelledSparseTensor{ tsdf_sparse_indices_filtered,
                                  tsdf_sparse_values_filtered,
                                  { 1, resolution_grid.z, resolution_grid.y, resolution_grid.x } });

    auto final_field = to_sparse_coo_tensor(
            RavelledSparseTensor{ sparse_indices,
                                  final_values,
                                  { 1, resolution_grid.z, resolution_grid.y, resolution_grid.x } });

    return { normals, tsdf_sparse_field, final_field, candidates_octree };
}

} // namespace torchhull
