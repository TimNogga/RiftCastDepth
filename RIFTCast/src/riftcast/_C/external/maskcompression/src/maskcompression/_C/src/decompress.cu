#include <maskcompression/decompress.h>

#include <ATen/cuda/ApplyGridUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <iostream>

namespace maskcompression
{

namespace detail
{

inline __device__ uint32_t
binary_search(const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits>& sorted_array, int32_t value)
{
    // Find first element in sorted_array that is larger than value.
    uint32_t left  = 1;    // Start at 1 because index 0 encodes if the mask starts with 0 or 1
    uint32_t right = sorted_array.size(0) - 1;
    while(left < right)
    {
        uint32_t mid = (left + right) / 2;
        if(sorted_array[mid] < value)
            left = mid + 1;
        else
            right = mid;
    }
    return left;
}

template<typename scalar_t>
__global__ void decompressImage(const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> cumsum,
                                const uint32_t batch_id,
                                const uint32_t width,
                                const uint32_t height,
                                const bool vertical_flip,
                                torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> output)
{
    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for(int32_t tid = id; tid < width * height; tid += num_threads)
    {
        int pixel_x = tid % width;
        int pixel_y = vertical_flip ? height - tid / width - 1 : tid / width;

        uint32_t bin_index =
            binary_search(cumsum, tid + 1) - 1;    // -1 because index 0 encodes if the mask starts with 0 or 1

        int32_t leading_one = cumsum[0];

        output[batch_id][pixel_y][pixel_x] = ((bin_index + leading_one) & 1) ? scalar_t(1) : scalar_t(0);
    }
}
}    // namespace detail

torch::Tensor decompress(const std::vector<torch::Tensor>& compressed,
                         const std::array<int, 2>& resolution,
                         const bool vertical_flip,
                         const std::optional<torch::ScalarType> dtype_)
{
    if(resolution[0] <= 0 || resolution[1] <= 1)
    {
        std::stringstream ss;
        ss << "Got invalid input resolution: " << resolution[0] << ", " << resolution[1];
        throw std::runtime_error(ss.str());
    }

    auto dtype = dtype_ ? *dtype_ : torch::kUInt8;

    int batch_size       = compressed.size();
    torch::Tensor output = torch::zeros({batch_size, resolution[0], resolution[1]},
                                        torch::TensorOptions {}.dtype(dtype).device(torch::kCUDA));

    auto device = output.device();

    at::cuda::CUDAGuard device_guard {device};
    const auto stream = at::cuda::getCurrentCUDAStream();

    const int threads_per_block = 128;
    dim3 grid;
    at::cuda::getApplyGrid(resolution[0] * resolution[1], grid, device.index(), threads_per_block);
    dim3 threads = at::cuda::getApplyBlock(threads_per_block);

    for(int batch_id = 0; batch_id < batch_size; ++batch_id)
    {
        auto cumsum = compressed[batch_id];

        if(!cumsum.device().is_cuda())
        {
            throw std::runtime_error("maskcompression only works for CUDA capable devices");
        }

        if(cumsum.device() != device)
        {
            std::stringstream ss;
            ss << "All compressed masks have to be on the same device. Got: " << cumsum.device() << " and " << device;
            throw std::runtime_error(ss.str());
        }

        if(cumsum.scalar_type() != torch::kInt32)
        {
            std::stringstream ss;
            ss << "Got unexpected datatype for maskcompression. Expected torch.int32, got " << cumsum.scalar_type();
            throw std::runtime_error(ss.str());
        }

        AT_DISPATCH_ALL_TYPES(output.scalar_type(),
                              "decompress_image",
                              [&]()
                              {
                                  auto output_ = output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();

                                  detail::decompressImage<<<grid, threads, 0, stream>>>(
                                      cumsum.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                                      batch_id,
                                      resolution[1],
                                      resolution[0],
                                      vertical_flip,
                                      output_);
                              });
    }

    AT_CUDA_CHECK(cudaGetLastError());
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));

    return output;
}
}    // namespace maskcompression