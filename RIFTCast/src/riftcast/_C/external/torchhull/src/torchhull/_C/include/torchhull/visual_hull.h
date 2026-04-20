#pragma once

#include <array>
#include <tuple>

#include <torch/types.h>

namespace torchhull
{

enum class DepthFusionPreset : int
{
    NoDepth = 0,
    RealData = 1,
    SyntheticData = 2,
};

std::tuple<torch::Tensor, torch::Tensor>
visual_hull(const torch::Tensor& masks,
            const torch::Tensor& depths,
            const torch::Tensor& transforms,
            const int level,
            const std::array<float, 3>& cube_corner_bfl,
            const float cube_length,
            const bool masks_partial,
            const std::string& transforms_convention,
            const bool unique_verts,
            const int depth_fusion_preset = static_cast<int>(DepthFusionPreset::RealData));

std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor>>
visual_hull_with_candidate_voxels(const torch::Tensor& masks,
                                  const torch::Tensor& depths,
                                  const torch::Tensor& transforms,
                                  const int level,
                                  const std::array<float, 3>& cube_corner_bfl,
                                  const float cube_length,
                                  const bool masks_partial,
                                  const std::string& transforms_convention,
                                  const bool unique_verts,
                                  const int depth_fusion_preset = static_cast<int>(DepthFusionPreset::RealData));

std::vector<std::tuple<torch::Tensor, torch::Tensor>>
candidate_voxels_to_wireframes(const std::vector<torch::Tensor>& candidate_voxels,
                               const std::array<float, 3>& cube_corner_bfl,
                               const float cube_length);

torch::Tensor
sparse_visual_hull_field(const torch::Tensor& masks,
                         const torch::Tensor& depths,
                         const torch::Tensor& transforms,
                         const int level,
                         const std::array<float, 3>& cube_corner_bfl,
                         const float cube_length,
                         const bool masks_partial,
                         const std::string& transforms_convention,
                         const int depth_fusion_preset = static_cast<int>(DepthFusionPreset::RealData));

} // namespace torchhull
