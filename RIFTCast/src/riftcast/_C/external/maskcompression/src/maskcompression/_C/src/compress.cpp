#include <maskcompression/compress.h>

namespace maskcompression
{
std::vector<torch::Tensor> compress(const torch::Tensor& masks)
{
    if(masks.ndimension() != 3)
    {
        std::stringstream ss;
        ss << "Tensor has wrong size of dimensions. Expected ndim==3 (B,H,W) but received ndim=" << masks.ndimension()
           << " with shape=" << masks.sizes();
        throw std::runtime_error(ss.str());
    }

    uint32_t batch_size = masks.size(0);
    std::vector<torch::Tensor> result(batch_size);

    for(int i = 0; i < batch_size; ++i)
    {
        int32_t leading_one = (int32_t)(masks.index({0, 0, 0}).item<bool>());
        auto compressed     = std::get<2>(
            torch::unique_consecutive(masks.index({i, torch::indexing::Slice(), torch::indexing::Slice()}).flatten(),
                                      /*return_inverse = */ false,
                                      /*return_counts = */ true));

        result[i] = torch::cat(
            {torch::tensor({leading_one}, torch::TensorOptions {}.dtype(torch::kInt32).device(compressed.device())),
             torch::cumsum(compressed, 0).to(torch::kInt32)});
    }

    return result;
}
}    // namespace maskcompression