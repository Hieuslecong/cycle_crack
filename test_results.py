"""
test_results.py — Cycle-Crack Test & Visualization Script

Usage:
  # Test all cracked images in data/cracked/ using latest checkpoint
  python test_results.py

  # Test with a specific checkpoint
  python test_results.py --checkpoint checkpoints/GE_epoch_100.pth

  # Test with ground truth masks for quantitative evaluation
  python test_results.py --gt_dir /path/to/gt_masks/

  # Test on a custom folder of images
  python test_results.py --image_dir /path/to/test/images/

This script produces:
  outputs/test_YYYYMMDD_HHMMSS/
  ├── grids/          # Side-by-side: Original | Error Map | Binary Mask
  ├── masks/          # Binary crack masks only
  ├── error_maps/     # Normalized error maps only
  └── metrics.txt     # Quantitative results (if --gt_dir provided)
"""

import os
import argparse
import datetime
import numpy as np
import cv2
import torch
from PIL import Image

from models.generator import UNetGenerator
from inference import segment_crack
from evaluate import calculate_metrics


# ── Argument Parsing ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Cycle-Crack Test & Visualization')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/GE_latest.pth',
                        help='Path to G_E weights (.pth)')
    parser.add_argument('--image_dir', type=str,
                        default='data/cracked/',
                        help='Directory of input images to test')
    parser.add_argument('--gt_dir', type=str, default=None,
                        help='Directory of GT binary masks (optional, for quantitative eval)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: outputs/test_TIMESTAMP/)')
    parser.add_argument('--stride', type=int, default=128,
                        help='Sliding window stride for inference')
    parser.add_argument('--max_images', type=int, default=50,
                        help='Max number of images to test (0 = all)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


# ── Utilities ────────────────────────────────────────────────────────────────

def load_generator(checkpoint_path: str, device: str) -> UNetGenerator:
    """Load the trained G_E generator from checkpoint.
    Always loads state_dict to CPU first to avoid CUDA OOM during deserialization.
    """
    # Free any cached GPU memory before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    G_E = UNetGenerator(in_channels=3, out_channels=3)
    # Load to CPU first, then move to device (avoids OOM at deserialization time)
    state = torch.load(checkpoint_path, map_location='cpu')
    G_E.load_state_dict(state)
    G_E.to(device)
    G_E.eval()
    print(f"Loaded G_E from: {checkpoint_path}")
    return G_E



def get_images(image_dir: str, max_images: int = 0):
    """Collect all valid image paths from a directory."""
    exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    paths = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if os.path.splitext(f.lower())[1] in exts
    ])
    if max_images > 0:
        paths = paths[:max_images]
    return paths


def make_grid(original: np.ndarray, error_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create a side-by-side comparison grid: Original | Error Map | Mask overlay."""
    h, w = original.shape[:2]

    # Error map → colormap heatmap (COLORMAP_JET: blue=low, red=high error)
    error_color = cv2.applyColorMap(error_map, cv2.COLORMAP_JET)
    error_color = cv2.resize(error_color, (w, h))

    # Mask overlay on original (crack pixels highlighted in red)
    overlay = original.copy()
    crack_pixels = mask > 127
    overlay[crack_pixels] = [0, 0, 255]  # Red for cracks (BGR)
    alpha = 0.5
    blend = cv2.addWeighted(original, 1 - alpha, overlay, alpha, 0)

    # Add text labels
    def label(img, text):
        img = img.copy()
        cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 1, cv2.LINE_AA)
        return img

    panels = [
        label(cv2.cvtColor(original, cv2.COLOR_RGB2BGR), 'Input'),
        label(error_color, 'Error Map'),
        label(blend, 'Crack Overlay'),
    ]
    return np.concatenate(panels, axis=1)


# ── Main Test Loop ───────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Output directories
    if args.output_dir is None:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = os.path.join('outputs', f'test_{ts}')

    grids_dir      = os.path.join(args.output_dir, 'grids')
    masks_dir      = os.path.join(args.output_dir, 'masks')
    error_maps_dir = os.path.join(args.output_dir, 'error_maps')
    for d in [grids_dir, masks_dir, error_maps_dir]:
        os.makedirs(d, exist_ok=True)

    # Load model
    G_E = load_generator(args.checkpoint, args.device)

    # Collect images
    image_paths = get_images(args.image_dir, args.max_images)
    if not image_paths:
        print(f"No images found in: {args.image_dir}")
        return
    print(f"Testing {len(image_paths)} images → {args.output_dir}")

    # Optionally collect GT masks
    gt_available = args.gt_dir is not None and os.path.isdir(args.gt_dir)
    all_metrics = []

    for i, img_path in enumerate(image_paths):
        basename = os.path.basename(img_path)
        stem     = os.path.splitext(basename)[0]

        print(f"[{i+1:3d}/{len(image_paths)}] {basename}", end='  ')

        # ── Inference ─────────────
        mask, error_map = segment_crack(img_path, G_E,
                                        device=args.device,
                                        stride=args.stride)

        # ── Save outputs ──────────
        original_rgb = np.array(Image.open(img_path).convert('RGB'))
        h, w = original_rgb.shape[:2]

        # Resize mask/error_map to match original size (in case of resize differences)
        mask_resized      = cv2.resize(mask,      (w, h), interpolation=cv2.INTER_NEAREST)
        error_map_resized = cv2.resize(error_map, (w, h), interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(os.path.join(masks_dir,      f'{stem}_mask.png'),      mask_resized)
        cv2.imwrite(os.path.join(error_maps_dir, f'{stem}_error.png'),     error_map_resized)

        grid = make_grid(original_rgb, error_map_resized, mask_resized)
        cv2.imwrite(os.path.join(grids_dir, f'{stem}_grid.png'), grid)

        # ── Quantitative Metrics (if GT available) ──
        if gt_available:
            # Try matching GT mask by same stem
            for ext in ['.png', '.jpg', '.bmp']:
                gt_path = os.path.join(args.gt_dir, stem + ext)
                if os.path.exists(gt_path):
                    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                    gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    metrics = calculate_metrics(mask_resized, gt_mask)
                    metrics['image'] = basename
                    all_metrics.append(metrics)
                    print(f"F1={metrics['F1']:.3f}  IoU={metrics['IoU']:.3f}  mBF1={metrics['mBF1']:.3f}", end='')
                    break
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    metrics_path = os.path.join(args.output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        if all_metrics:
            keys = ['Precision', 'Recall', 'F1', 'IoU', 'mBF1']
            # Per-image table
            f.write(f"{'Image':<30} {'Precision':>10} {'Recall':>8} {'F1':>8} {'IoU':>8} {'mBF1':>8}\n")
            f.write('-' * 76 + '\n')
            for m in all_metrics:
                f.write(f"{m['image']:<30} {m['Precision']:>10.4f} {m['Recall']:>8.4f} "
                        f"{m['F1']:>8.4f} {m['IoU']:>8.4f} {m['mBF1']:>8.4f}\n")
            f.write('-' * 76 + '\n')
            # Averages
            avgs = {k: np.mean([m[k] for m in all_metrics if k in m]) for k in keys}
            f.write(f"{'AVERAGE':<30} {avgs['Precision']:>10.4f} {avgs['Recall']:>8.4f} "
                    f"{avgs['F1']:>8.4f} {avgs['IoU']:>8.4f} {avgs['mBF1']:>8.4f}\n")

            print("\n=== Quantitative Results ===")
            for k in keys:
                print(f"  {k:12s}: {avgs[k]:.4f}")
            print(f"\nSaved metrics to: {metrics_path}")
        else:
            f.write("No ground truth masks provided — no quantitative metrics computed.\n")
            f.write(f"Tested {len(image_paths)} images.\n")
            f.write(f"Output saved to: {args.output_dir}\n")

    print(f"\n✓ Done. Results saved to: {args.output_dir}")
    print(f"  grids/       {len(image_paths)} side-by-side comparison images")
    print(f"  masks/       {len(image_paths)} binary crack masks")
    print(f"  error_maps/  {len(image_paths)} error heatmaps")
    if all_metrics:
        print(f"  metrics.txt  per-image + average scores")


if __name__ == '__main__':
    main()
