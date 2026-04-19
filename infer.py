"""infer.py — Cycle-Crack Inference Pipeline (v3, paper-accurate)

Usage:
    python infer.py --checkpoint checkpoints/GE_latest.pth \
                    --image test.jpg --output result.png

Pipeline [paper §III-B, Eq.1]:
  1. Sliding-window 256×256 tiling with stride=128
  2. G_E(patch) → error map |patch - G_E(patch)|
  3. Stitch via average of overlapping regions
  4. Bilateral filter (d=9, σ=75)
  5. Otsu threshold → binary mask
"""
import os, argparse
import numpy as np
import cv2
import torch
from PIL import Image

import config as C
from model import UNet256


def segment(image_path: str, G_E: UNet256, device: str = 'cuda',
            stride: int = 128) -> tuple:
    """Full-resolution crack segmentation via sliding window.

    Returns:
        mask      (np.ndarray uint8): Binary mask, 255=crack.
        error_map (np.ndarray uint8): Normalized error heatmap [0,255].
    """
    G_E.eval()
    ps = C.IMG_SIZE  # 256

    img_pil = Image.open(image_path).convert('RGB')
    W, H    = img_pil.size
    img_np  = np.array(img_pil, dtype=np.float32) / 127.5 - 1.0  # → [-1,1]

    # Reflect-pad so image tiles evenly
    ph = (ps - H % ps) % ps
    pw = (ps - W % ps) % ps
    img_pad = np.pad(img_np, ((0,ph),(0,pw),(0,0)), mode='reflect')
    Hp, Wp  = img_pad.shape[:2]

    err_acc = np.zeros((Hp, Wp), dtype=np.float32)
    cnt_map = np.zeros((Hp, Wp), dtype=np.float32)

    ys = list(range(0, Hp - ps + 1, stride))
    xs = list(range(0, Wp - ps + 1, stride))
    if not ys or ys[-1]+ps < Hp: ys.append(Hp-ps)
    if not xs or xs[-1]+ps < Wp: xs.append(Wp-ps)

    with torch.no_grad():
        for y in ys:
            for x in xs:
                patch = img_pad[y:y+ps, x:x+ps]               # (256,256,3)
                t = torch.tensor(patch).permute(2,0,1).unsqueeze(0).float().to(device)
                fake = G_E(t)
                err  = torch.abs(t - fake).mean(dim=1).squeeze().cpu().numpy()
                err_acc[y:y+ps, x:x+ps] += err
                cnt_map[y:y+ps, x:x+ps] += 1.0

    # Crop, normalize, filter, threshold
    err = (err_acc / np.maximum(cnt_map, 1e-8))[:H, :W]
    err = np.clip(err, 0, None)
    if err.max() > 0:
        err = (err / err.max() * 255).astype(np.uint8)
    else:
        err = err.astype(np.uint8)

    filtered = cv2.bilateralFilter(err,
                                   C.BILATERAL_D,
                                   C.BILATERAL_SIGMA_COLOR,
                                   C.BILATERAL_SIGMA_SPACE)
    _, mask = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # [not in paper]: morphological closing is NOT applied

    return mask, filtered


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--image',      required=True)
    parser.add_argument('--output',     default='result.png')
    parser.add_argument('--stride',     type=int, default=128)
    args   = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    G_E = UNet256()
    G_E.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    G_E.to(device)

    mask, err = segment(args.image, G_E, device=device, stride=args.stride)
    cv2.imwrite(args.output, mask)
    cv2.imwrite(args.output.replace('.png', '_error.png'), err)
    print(f"Saved: {args.output}")
