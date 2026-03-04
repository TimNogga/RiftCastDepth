#pragma once

#include <array>

#include <torch/types.h>

namespace maskcompression
{
/**
 * @brief Decompress a list of compressed masks.
 * Only works on cuda devices. The compressed tensors are expected to be cuda tensors.
 *
 * @param compressed List of linear tensors representing the masks
 * @param resolution The resolution of the mask. Has to be constant for all input masks
 * @param vertical_flip Optional vertical flip of the decompressed image
 * @param dtype The output type of the decompressed masks
 *
 * @return (B, H, W) tensor with decoded masks
 */
torch::Tensor decompress(const std::vector<torch::Tensor>& compressed,
                         const std::array<int, 2>& resolution,
                         const bool vertical_flip       = false,
                         const std::optional<torch::ScalarType> = std::nullopt);
}    // namespace maskcompression