import torch
import maskcompression
import random

import pytest


def create_mask():
    width = 1920
    height = 1080
    min_size = 20
    max_size = 200
    mask = torch.zeros((height, width), dtype=torch.uint8, device="cuda:0")

    num_boxes = 10
    for _ in range(num_boxes):
        # Randomly choose the top-left corner of the box
        x1 = random.randint(0, width - 1)
        y1 = random.randint(0, height - 1)

        # Randomly determine the width and height of the box
        box_width = random.randint(min_size, max_size)
        box_height = random.randint(min_size, max_size)

        # Ensure the box stays within the image bounds
        x2 = min(x1 + box_width, width)
        y2 = min(y1 + box_height, height)

        mask[y1:y2, x1:x2] = 1

    return mask


def test_compress_empty():
    empty = torch.zeros((300, 100), device="cuda:0", dtype=torch.uint8)

    compressed = maskcompression.compress(empty.unsqueeze(0))

    assert len(compressed) == 1
    assert len(compressed[0]) == 2
    assert compressed[0][0] == 0
    assert compressed[0][1] == empty.numel()


def test_compress_full():
    empty = torch.full((300, 100), 255, device="cuda:0", dtype=torch.uint8)

    compressed = maskcompression.compress(empty.unsqueeze(0))

    assert len(compressed) == 1
    assert len(compressed[0]) == 2
    assert compressed[0][0] == 1
    assert compressed[0][1] == empty.numel()


def test_compress_full_float():
    empty = torch.full((300, 100), 1.0, device="cuda:0", dtype=torch.float32)

    compressed = maskcompression.compress(empty.unsqueeze(0))

    assert len(compressed) == 1
    assert len(compressed[0]) == 2
    assert compressed[0][0] == 1
    assert compressed[0][1] == empty.numel()


def test_roundtrip():
    mask = create_mask()

    compressed = maskcompression.compress(mask.unsqueeze(0))

    decompressed = maskcompression.decompress(
        compressed, (mask.shape[0], mask.shape[1])
    )

    assert torch.allclose(decompressed[0], mask)


def test_roundtrip_batch():
    num_masks = 15
    masks = []
    for _ in range(num_masks):
        masks.append(create_mask())

    masks = torch.stack(masks, dim=0)

    compressed = maskcompression.compress(masks)

    decompressed = maskcompression.decompress(
        compressed, (masks.shape[1], masks.shape[2])
    )

    assert torch.allclose(decompressed, masks)


def test_roundtrip_flip():
    mask = create_mask()

    compressed = maskcompression.compress(mask.unsqueeze(0))

    decompressed = maskcompression.decompress(
        compressed, (mask.shape[0], mask.shape[1]), True
    )

    assert torch.allclose(decompressed[0], mask.flip(0))


def test_roundtrip_batch_flip():
    num_masks = 15
    masks = []
    for _ in range(num_masks):
        masks.append(create_mask())

    masks = torch.stack(masks, dim=0)

    compressed = maskcompression.compress(masks)

    decompressed = maskcompression.decompress(
        compressed, (masks.shape[1], masks.shape[2]), True
    )

    assert torch.allclose(decompressed, masks.flip(1))


def test_roundtrip_int32():
    mask = create_mask()

    compressed = maskcompression.compress(mask.unsqueeze(0))

    decompressed = maskcompression.decompress(
        compressed, (mask.shape[0], mask.shape[1]), dtype=torch.int32
    )

    assert torch.allclose(decompressed[0], mask.to(decompressed.dtype))


def test_roundtrip_batch_int32():
    num_masks = 15
    masks = []
    for _ in range(num_masks):
        masks.append(create_mask())

    masks = torch.stack(masks, dim=0)

    compressed = maskcompression.compress(masks)

    decompressed = maskcompression.decompress(
        compressed, (masks.shape[1], masks.shape[2]), dtype=torch.int32
    )

    assert torch.allclose(decompressed, masks.to(decompressed.dtype))


def test_roundtrip_flip_int32():
    mask = create_mask()

    compressed = maskcompression.compress(mask.unsqueeze(0))

    decompressed = maskcompression.decompress(
        compressed, (mask.shape[0], mask.shape[1]), True, dtype=torch.int32
    )

    assert torch.allclose(decompressed[0], mask.flip(0).to(decompressed.dtype))


def test_roundtrip_batch_flip_int32():
    num_masks = 15
    masks = []
    for _ in range(num_masks):
        masks.append(create_mask())

    masks = torch.stack(masks, dim=0)

    compressed = maskcompression.compress(masks)

    decompressed = maskcompression.decompress(
        compressed, (masks.shape[1], masks.shape[2]), True, dtype=torch.int32
    )

    assert torch.allclose(decompressed, masks.flip(1).to(decompressed.dtype))


def test_roundtrip_int32():
    mask = create_mask()

    compressed = maskcompression.compress(mask.unsqueeze(0))

    decompressed = maskcompression.decompress(
        compressed, (mask.shape[0], mask.shape[1]), dtype=torch.int32
    )

    assert torch.allclose(decompressed[0], mask.to(decompressed.dtype))


def test_roundtrip_batch_int32():
    num_masks = 15
    masks = []
    for _ in range(num_masks):
        masks.append(create_mask())

    masks = torch.stack(masks, dim=0)

    compressed = maskcompression.compress(masks)

    decompressed = maskcompression.decompress(
        compressed, (masks.shape[1], masks.shape[2]), dtype=torch.int32
    )

    assert torch.allclose(decompressed, masks.to(decompressed.dtype))


def test_roundtrip_flip_int32():
    mask = create_mask()

    compressed = maskcompression.compress(mask.unsqueeze(0))

    decompressed = maskcompression.decompress(
        compressed, (mask.shape[0], mask.shape[1]), True, dtype=torch.int32
    )

    assert torch.allclose(decompressed[0], mask.flip(0).to(decompressed.dtype))


def test_roundtrip_batch_flip_float():
    num_masks = 15
    masks = []
    for _ in range(num_masks):
        masks.append(create_mask())

    masks = torch.stack(masks, dim=0)

    compressed = maskcompression.compress(masks)

    decompressed = maskcompression.decompress(
        compressed, (masks.shape[1], masks.shape[2]), True, dtype=torch.float32
    )

    assert torch.allclose(decompressed, masks.flip(1).to(decompressed.dtype))


def test_wrong_shape():
    empty = torch.zeros((300, 100), device="cuda:0", dtype=torch.uint8)

    with pytest.raises(RuntimeError):
        _ = maskcompression.compress(empty)


def test_wrong_device():
    num_masks = 15
    masks = []
    for _ in range(num_masks):
        masks.append(create_mask())

    masks = torch.stack(masks, dim=0)

    compressed = maskcompression.compress(masks)

    for i in range(len(compressed)):
        compressed[i] = compressed[i].cpu()
    with pytest.raises(RuntimeError):
        _ = maskcompression.decompress(compressed, (masks.shape[1], masks.shape[2]))


def test_wrong_dtype():
    num_masks = 15
    masks = []
    for _ in range(num_masks):
        masks.append(create_mask())

    masks = torch.stack(masks, dim=0)

    compressed = maskcompression.compress(masks)

    for i in range(len(compressed)):
        compressed[i] = compressed[i].to(torch.float32)
    with pytest.raises(RuntimeError):
        _ = maskcompression.decompress(compressed, (masks.shape[1], masks.shape[2]))


def test_wrong_resolution():
    num_masks = 15
    masks = []
    for _ in range(num_masks):
        masks.append(create_mask())

    masks = torch.stack(masks, dim=0)

    compressed = maskcompression.compress(masks)

    with pytest.raises(RuntimeError):
        _ = maskcompression.decompress(compressed, (1, 0))

    with pytest.raises(RuntimeError):
        _ = maskcompression.decompress(compressed, (0, 1))

    with pytest.raises(RuntimeError):
        _ = maskcompression.decompress(compressed, (-1, 1))

    with pytest.raises(RuntimeError):
        _ = maskcompression.decompress(compressed, (1, -1))
