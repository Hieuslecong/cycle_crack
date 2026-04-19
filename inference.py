import torch
import cv2
import numpy as np
import os
import argparse
from PIL import Image
from models.generator import UNetGenerator
from data.transforms import get_transforms


def segment_crack(image_path: str, generator_GE: UNetGenerator,
                  device: str = 'cuda', stride: int = 128) -> tuple:
    """
    Segment pavement cracks using the Cycle-Crack G_E generator.
    
    Full-resolution inference via sliding-window tiling with average stitching.
    [paper §III-B, Eq.1]

    Steps:
    1. Partition full-resolution image into overlapping 256×256 patches
    2. For each patch: normalize → G_E → error map |p - p'|
    3. Stitch patches back (average overlapping regions)
    4. Rescale error map to [0, 255]
    5. Apply bilateral filter
    6. Apply Otsu thresholding → binary crack mask
    # [not in paper]: morphological closing is NOT applied

    Args:
        image_path: Path to input image.
        generator_GE: Trained G_E generator.
        device: Computation device.
        stride: Patch stride for sliding window (128 for 50% overlap).

    Returns:
        mask (np.ndarray): Binary crack mask (255 = crack, 0 = background).
        error_map (np.ndarray): Normalized error map [0, 255] uint8.
    """
    generator_GE.eval()
    patch_size = 256
    transform = get_transforms(image_size=patch_size, is_train=False)

    # Load full-resolution image
    img_pil = Image.open(image_path).convert('RGB')
    W, H = img_pil.size
    img_np = np.array(img_pil).astype(np.float32) / 127.5 - 1.0  # [-1, 1]

    # Pad image so it tiles evenly
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    img_padded = np.pad(img_np, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    H_pad, W_pad = img_padded.shape[:2]

    error_accum = np.zeros((H_pad, W_pad), dtype=np.float32)
    count_map   = np.zeros((H_pad, W_pad), dtype=np.float32)

    # Sliding window
    ys = list(range(0, H_pad - patch_size + 1, stride))
    xs = list(range(0, W_pad - patch_size + 1, stride))
    if not ys or ys[-1] + patch_size < H_pad:
        ys.append(H_pad - patch_size)
    if not xs or xs[-1] + patch_size < W_pad:
        xs.append(W_pad - patch_size)

    with torch.no_grad():
        for y in ys:
            for x in xs:
                patch = img_padded[y:y + patch_size, x:x + patch_size]  # (H, W, 3)
                patch_tensor = torch.tensor(patch).permute(2, 0, 1).unsqueeze(0).float().to(device)

                fake = generator_GE(patch_tensor)

                # Error map: |original - G_E(original)|, averaged over channels
                error = torch.abs(patch_tensor - fake).mean(dim=1).squeeze().cpu().numpy()
                error_accum[y:y + patch_size, x:x + patch_size] += error
                count_map  [y:y + patch_size, x:x + patch_size] += 1.0

    # Average overlapping regions and crop back
    error_map = error_accum / np.maximum(count_map, 1e-8)
    error_map = error_map[:H, :W]

    # Rescale to [0, 255]
    error_map = np.clip(error_map, 0, None)
    if error_map.max() > 0:
        error_map = (error_map / error_map.max() * 255).astype(np.uint8)
    else:
        error_map = error_map.astype(np.uint8)

    # Bilateral filter
    error_filtered = cv2.bilateralFilter(error_map, d=9, sigmaColor=75, sigmaSpace=75)

    # Otsu thresholding
    _, mask = cv2.threshold(error_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # [not in paper]: morphological closing is NOT added here

    return mask, error_filtered


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path',  type=str, required=True)
    parser.add_argument('--checkpoint',  type=str, required=True, help='Path to G_E weights (.pth)')
    parser.add_argument('--output_dir',  type=str, default='outputs')
    parser.add_argument('--stride',      type=int, default=128, help='Sliding window stride')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    G_E = UNetGenerator(in_channels=3, out_channels=3)
    G_E.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    G_E.to(device)

    os.makedirs(args.output_dir, exist_ok=True)
    mask, error_map = segment_crack(args.image_path, G_E, device=device, stride=args.stride)

    base_name = os.path.basename(args.image_path)
    cv2.imwrite(os.path.join(args.output_dir, f'mask_{base_name}'),  mask)
    cv2.imwrite(os.path.join(args.output_dir, f'error_{base_name}'), error_map)
    print(f"Saved results to {args.output_dir}")
