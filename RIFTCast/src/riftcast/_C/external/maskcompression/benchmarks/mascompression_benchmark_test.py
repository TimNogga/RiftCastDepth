import torch
import random

import maskcompression


def create_mask():
    width = 1920
    height = 1080
    min_size = 20
    max_size = 200
    mask = torch.zeros((height, width), dtype=torch.float32, device="cuda:0")

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

        mask[y1:y2, x1:x2] = 1.0

    return mask


def create_batch(num_masks):
    masks = []
    for _ in range(num_masks):
        masks.append(create_mask())

    masks = torch.stack(masks, dim=0)

    return masks


def run_compression(masks):
    torch.cuda.synchronize()
    _ = maskcompression.compress(masks)
    torch.cuda.synchronize()


def run_decompression(compressed):
    torch.cuda.synchronize()
    _ = maskcompression.decompress(compressed, (1080, 1920))
    torch.cuda.synchronize()


def test_single_mask_compression_speed(benchmark):
    masks = create_mask().unsqueeze(0)

    benchmark(run_compression, masks)


def test_batch_mask_compression_speed_05(benchmark):
    masks = create_batch(5)

    benchmark(run_compression, masks)


def test_batch_mask_compression_speed_10(benchmark):
    masks = create_batch(10)

    benchmark(run_compression, masks)


def test_batch_mask_compression_speed_15(benchmark):
    masks = create_batch(15)

    benchmark(run_compression, masks)


def test_batch_mask_compression_speed_20(benchmark):
    masks = create_batch(20)

    benchmark(run_compression, masks)


def test_batch_mask_compression_speed_30(benchmark):
    masks = create_batch(30)

    benchmark(run_compression, masks)


def test_batch_mask_compression_speed_40(benchmark):
    masks = create_batch(40)

    benchmark(run_compression, masks)


def test_batch_mask_compression_speed_60(benchmark):
    masks = create_batch(60)

    benchmark(run_compression, masks)


def test_single_mask_decompression_speed(benchmark):
    masks = create_mask().unsqueeze(0)

    compressed = maskcompression.compress(masks)
    benchmark(run_decompression, compressed)


def test_batch_mask_decompression_speed_05(benchmark):
    masks = create_batch(5)

    compressed = maskcompression.compress(masks)
    benchmark(run_decompression, compressed)


def test_batch_mask_decompression_speed_10(benchmark):
    masks = create_batch(10)

    compressed = maskcompression.compress(masks)
    benchmark(run_decompression, compressed)


def test_batch_mask_decompression_speed_15(benchmark):
    masks = create_batch(15)

    compressed = maskcompression.compress(masks)
    benchmark(run_decompression, compressed)


def test_batch_mask_decompression_speed_20(benchmark):
    masks = create_batch(20)

    compressed = maskcompression.compress(masks)
    benchmark(run_decompression, compressed)


def test_batch_mask_decompression_speed_30(benchmark):
    masks = create_batch(30)

    compressed = maskcompression.compress(masks)
    benchmark(run_decompression, compressed)


def test_batch_mask_decompression_speed_40(benchmark):
    masks = create_batch(40)

    compressed = maskcompression.compress(masks)
    benchmark(run_decompression, compressed)


def test_batch_mask_decompression_speed_60(benchmark):
    masks = create_batch(60)

    compressed = maskcompression.compress(masks)
    benchmark(run_decompression, compressed)
