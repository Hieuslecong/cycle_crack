import numpy as np
import cv2


def calculate_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """
    Calculate evaluation metrics comparing predicted mask and ground truth mask.
    
    Metrics: Precision, Recall, F1, IoU, mBF1.
    mBF1 is computed at boundary tolerance d ∈ {1, 3, 5} and averaged. [paper §IV-B, Eq.13]
    
    Args:
        pred_mask (np.ndarray): Predicted binary mask (0 or 255).
        gt_mask (np.ndarray): Ground truth binary mask (0 or 255).
    
    Returns:
        dict with Precision, Recall, F1, IoU, mBF1 (all float in [0, 1]).
    """
    pred = (pred_mask > 127).astype(np.uint8)
    gt   = (gt_mask   > 127).astype(np.uint8)

    TP = np.sum((pred == 1) & (gt == 1))
    FP = np.sum((pred == 1) & (gt == 0))
    FN = np.sum((pred == 0) & (gt == 1))

    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    iou       = TP / (TP + FP + FN + 1e-8)

    # Boundary F1 — average over d ∈ {1, 3, 5} [paper §IV-B, Eq.13]
    bf1_scores = []
    for d in [1, 3, 5]:
        bf1_scores.append(_boundary_f1(pred, gt, tolerance=d))
    mbf1 = float(np.mean(bf1_scores))

    return {
        'Precision': float(precision),
        'Recall':    float(recall),
        'F1':        float(f1),
        'IoU':       float(iou),
        'mBF1':      mbf1,
    }


def _boundary_f1(pred: np.ndarray, gt: np.ndarray, tolerance: int = 3) -> float:
    """Compute boundary F1 at a given tolerance (dilation radius).
    
    Extract boundaries using morphological gradient, then check if
    predicted boundary pixels fall within 'tolerance' pixels of the GT boundary.
    
    Args:
        pred: Binary predicted mask (0/1).
        gt: Binary ground truth mask (0/1).
        tolerance: Dilation radius in pixels.
    
    Returns:
        Boundary F1 score.
    """
    kernel = np.ones((tolerance * 2 + 1, tolerance * 2 + 1), np.uint8)

    # Extract boundaries via morphological gradient (dil - ero)
    gt_boundary   = cv2.dilate(gt,   kernel) - cv2.erode(gt,   kernel)
    pred_boundary = cv2.dilate(pred, kernel) - cv2.erode(pred, kernel)

    # Dilate boundaries to create tolerance zones
    gt_dil   = cv2.dilate(gt_boundary,   kernel)
    pred_dil = cv2.dilate(pred_boundary, kernel)

    # Boundary precision: how many predicted boundary px are near GT boundary
    TP_p = np.sum((pred_boundary == 1) & (gt_dil == 1))
    FP_p = np.sum((pred_boundary == 1) & (gt_dil == 0))

    # Boundary recall: how many GT boundary px are near predicted boundary
    TP_r = np.sum((gt_boundary == 1) & (pred_dil == 1))
    FN_r = np.sum((gt_boundary == 1) & (pred_dil == 0))

    precision_b = TP_p / (TP_p + FP_p + 1e-8)
    recall_b    = TP_r / (TP_r + FN_r + 1e-8)
    bf1 = 2 * precision_b * recall_b / (precision_b + recall_b + 1e-8)
    return float(bf1)
