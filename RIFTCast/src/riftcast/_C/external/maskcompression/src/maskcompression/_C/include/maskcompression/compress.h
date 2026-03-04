#pragma once

#include <torch/types.h>

namespace maskcompression
{
/**
 * @brief Compress a batch of masks.
 *
 * @param masks (B, H, W) tensor of masks
 * @return A list of linear compressed masks
 */
std::vector<torch::Tensor> compress(const torch::Tensor& masks);
}    // namespace maskcompression