#include <torch/python.h>

#include <maskcompression/decompress.h>
#include <maskcompression/compress.h>

using namespace pybind11::literals;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("decompress",
          &maskcompression::decompress,
          "compressed"_a,
          "resolution"_a,
          "vertical_flip"_a = false,
          "dtype"_a         = py::none(),
          R"(
    Decompress a list of compressed masks.
    Only works on cuda devices. The compressed tensors are expected to be cuda tensors.
    
    Parameters
    ----------
    compressed 
        List of linear tensors representing the masks
    resolution 
        The resolution of the mask. Has to be constant for all input masks
    vertical_flip
        Optional vertical flip of the decompressed image
    dtype
        The dtype of the output masks

    Returns
    -------
        masks: (B, H, W) tensor with decoded masks
        )");

    m.def("compress", &maskcompression::compress, "masks"_a, R"(
    Compress a batch of masks.
        
    Parameters
    ----------
    masks 
        (B, H, W) tensor of masks

    Returns
    -------
        compressed: A list of linear compressed masks
    )");
}