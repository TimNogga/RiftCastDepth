<h1 align="center">maskcompression</h1>

A simple library to encode and decode run length compressed binary mask images.
A batch of foreground masks (B,H,W, dtype=np.float32) is converted to a list of tensors with `len(compressed) == B`. Each tensor represents a run-length encoded mask (dtype=np.uint32), where the first entry is either 0 or 1 if the first pixel is 0 or 1 and the renaining entries represent a cumulative histogram of number of consecutive foreground or background pixels.

## Installation

This package only works with CUDA enabled.

Requires 
* Python >= 3.8, 
* [charonload](https://github.com/vc-bonn/charonload), 
* [pytorch](https://pytorch.org), 
* a C++ and CUDA compiler. 

Can be installed via:

```
python -m pip install --editable .
```

## Quick Start

* `masks` (B, H, W) tensor of masks
* `compressed` List of compressed masks of `dtype=np.int32`

```python
import maskcompression

masks = generate_masks() # (B,H,W), device=cuda, dtype=np.float32

compressed = maskcompression.compress(masks) # list(torch.Tensor, dtype=np.int32)
decompressed = maskcompression.decompress(compressed, 
                                          resolution, # (H, W)
                                          vertical_flip=False,
                                          dtype=torch.float32) # (B,H,W), device=cuda, foreground=1
```

## Licnese

MIT. See [`LICENSE`](LICENSE).

## Contact

Domenic Zingsheim - zingsheim@cs.uni-bonn.de