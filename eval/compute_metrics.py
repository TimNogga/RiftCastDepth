#!/usr/bin/env python3
"""Compute PSNR / SSIM / LPIPS between rendered views and GT images."""

import argparse
import json
import math
import struct
from pathlib import Path

import cv2
import numpy as np

# ------------------------------------------------------------------ deps --
try:
    import lpips
    import torch
    _LPIPS_NET = None

    def _get_lpips():
        global _LPIPS_NET
        if _LPIPS_NET is None:
            _LPIPS_NET = lpips.LPIPS(net="alex")
            if torch.cuda.is_available():
                _LPIPS_NET = _LPIPS_NET.cuda()
        return _LPIPS_NET

    def _lpips(img_a: np.ndarray, img_b: np.ndarray) -> float:
        """img_a/b: HxWx3 uint8"""
        net = _get_lpips()
        def to_tensor(x):
            t = torch.from_numpy(x).float() / 127.5 - 1.0  # [-1,1]
            t = t.permute(2, 0, 1).unsqueeze(0)
            if torch.cuda.is_available():
                t = t.cuda()
            return t
        with torch.no_grad():
            d = net(to_tensor(img_a), to_tensor(img_b))
        return float(d.item())

    HAS_LPIPS = True
except ModuleNotFoundError:
    HAS_LPIPS = False
    print("[warn] lpips not installed. Run: pip install lpips")
    print("       LPIPS scores will be reported as NaN.")

try:
    from skimage.metrics import structural_similarity as _skimage_ssim
    def _ssim(img_a: np.ndarray, img_b: np.ndarray) -> float:
        return float(_skimage_ssim(img_a, img_b, channel_axis=2, data_range=255))
    HAS_SKIMAGE = True
except ModuleNotFoundError:
    HAS_SKIMAGE = False
    try:
        # torchvision fallback
        import torch
        from torchvision.transforms.functional import to_tensor as _tv_to_tensor
        from torchvision.ops import box_iou  # just to test import
    except Exception:
        pass

    def _ssim(img_a: np.ndarray, img_b: np.ndarray) -> float:
        # minimal SSIM approximation via OpenCV filter
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        a = img_a.astype(np.float64)
        b = img_b.astype(np.float64)
        k = 11
        mu_a = cv2.GaussianBlur(a, (k, k), 1.5)
        mu_b = cv2.GaussianBlur(b, (k, k), 1.5)
        mu_a2 = mu_a * mu_a
        mu_b2 = mu_b * mu_b
        mu_ab = mu_a * mu_b
        sig_a2 = cv2.GaussianBlur(a * a, (k, k), 1.5) - mu_a2
        sig_b2 = cv2.GaussianBlur(b * b, (k, k), 1.5) - mu_b2
        sig_ab = cv2.GaussianBlur(a * b, (k, k), 1.5) - mu_ab
        ssim_map = ((2 * mu_ab + c1) * (2 * sig_ab + c2)) / (
            (mu_a2 + mu_b2 + c1) * (sig_a2 + sig_b2 + c2)
        )
        return float(ssim_map.mean())


# ------------------------------------------------------------------ io --
CAM_ID_TO_NAME = {
    # filled from calibration order; override via --id-map if needed
}

H_COLOR, W_COLOR = 5328, 4608
LONG_EDGE = 1024  # resize to this for LPIPS (avoids OOM)


def load_mask_bin(path: Path, h: int = H_COLOR, w: int = W_COLOR, flip: bool = True) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size == h * w:
        mask = raw.reshape(h, w)
    else:
        raise ValueError(f"Unexpected mask size {raw.size} in {path}")
    if flip:
        mask = np.flipud(mask)
    return mask > 127


def load_image(path: Path) -> np.ndarray:
    """Returns HxWx3 uint8 RGB."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_long_edge(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= size:
        return img
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


# ------------------------------------------------------------------ core --
def evaluate_pair(
    pred_path: Path,
    gt_path: Path,
    mask_path: Path | None,
    long_edge: int = LONG_EDGE,
) -> dict:
    pred = load_image(pred_path)
    gt   = load_image(gt_path)

    # rendered images are RGBA — drop alpha if present
    if pred.shape[2] == 4:
        pred = pred[:, :, :3]

    # GT may be larger; if pred was rendered at a different res, resize GT to match
    if pred.shape[:2] != gt.shape[:2]:
        gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_AREA)

    mask = None
    if mask_path is not None and mask_path.exists():
        m = load_mask_bin(mask_path)
        if m.shape[:2] != pred.shape[:2]:
            m_img = m.astype(np.uint8) * 255
            m_img = cv2.resize(m_img, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
            m = m_img > 127
        mask = m

    if mask is not None:
        pred_fg = pred[mask]
        gt_fg   = gt[mask]
        # rebuild masked arrays for SSIM/LPIPS (black out background)
        pred_m = np.zeros_like(pred)
        gt_m   = np.zeros_like(gt)
        pred_m[mask] = pred_fg
        gt_m[mask]   = gt_fg
    else:
        pred_m, gt_m = pred, gt

    result = {}
    result["psnr"] = psnr(pred_m, gt_m)
    result["ssim"] = _ssim(pred_m, gt_m)

    if HAS_LPIPS:
        p_small = resize_long_edge(pred_m, long_edge)
        g_small = resize_long_edge(gt_m,   long_edge)
        result["lpips"] = _lpips(p_small, g_small)
    else:
        result["lpips"] = float("nan")

    return result


# ------------------------------------------------------------------ main --
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred",  type=Path, required=True, help="Dir with rgb_<id>.png")
    parser.add_argument("--gt",    type=Path, required=True, help="Dir with C<name>.png")
    parser.add_argument("--mask",  type=Path, default=None,  help="Dir with mask_C<name>.bin")
    parser.add_argument("--cams",  type=int, nargs="+", required=True, help="Camera integer IDs")
    parser.add_argument("--cam-names", nargs="+", default=None, metavar="NAME",
                        help="Camera names matching --cams (e.g. C0005 C0026 C0037)")
    parser.add_argument("--label", default="run", help="Label for output")
    parser.add_argument("--out",   type=Path, default=None, help="Write results JSON here")
    parser.add_argument("--long-edge", type=int, default=LONG_EDGE)
    args = parser.parse_args()

    # build id -> name map
    if args.cam_names:
        assert len(args.cam_names) == len(args.cams)
        id_to_name = dict(zip(args.cams, args.cam_names))
    else:
        # derive name from id: id 5 -> "C0005", id 1005 -> "C1005"
        id_to_name = {i: f"C{i:04d}" for i in args.cams}

    results = []
    for cam_id, cam_name in id_to_name.items():
        pred_file = args.pred / f"rgb_{cam_id}.png"
        gt_file   = args.gt   / f"{cam_name}.png"
        mask_file = (args.mask / f"mask_{cam_name}.bin") if args.mask else None

        if not pred_file.exists():
            print(f"  [skip] pred not found: {pred_file}")
            continue
        if not gt_file.exists():
            print(f"  [skip] GT not found: {gt_file}")
            continue

        metrics = evaluate_pair(pred_file, gt_file, mask_file, args.long_edge)
        metrics["cam_id"]   = cam_id
        metrics["cam_name"] = cam_name
        results.append(metrics)

        print(f"  {cam_name:6s}  PSNR={metrics['psnr']:6.2f}  "
              f"SSIM={metrics['ssim']:.4f}  LPIPS={metrics['lpips']:.4f}")

    if not results:
        print("No results computed.")
        return

    psnrs  = [r["psnr"]  for r in results]
    ssims  = [r["ssim"]  for r in results]
    lpipss = [r["lpips"] for r in results if not math.isnan(r["lpips"])]

    print(f"\n  {'MEAN':6s}  PSNR={np.mean(psnrs):6.2f}  "
          f"SSIM={np.mean(ssims):.4f}  "
          f"LPIPS={np.mean(lpipss) if lpipss else float('nan'):.4f}")

    summary = {
        "label": args.label,
        "cameras": results,
        "mean": {
            "psnr":  float(np.mean(psnrs)),
            "ssim":  float(np.mean(ssims)),
            "lpips": float(np.mean(lpipss)) if lpipss else None,
        },
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Results written to {args.out}")


if __name__ == "__main__":
    main()
