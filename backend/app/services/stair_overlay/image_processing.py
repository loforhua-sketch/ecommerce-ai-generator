"""Image loading and product alpha preparation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: Path, unchanged: bool = False, description: str = "image") -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED if unchanged else cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read {description}: {path}")
    return image


def _validate_byte_threshold(value: int, name: str) -> None:
    if not 0 <= value <= 255:
        raise ValueError(f"{name} must be between 0 and 255")


def _keep_product_components(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    minimum_area = max(1, int(mask.size * 0.005))
    keep = set((np.flatnonzero(areas >= minimum_area) + 1).tolist())
    keep.add(int(np.argmax(areas)) + 1)
    return np.where(np.isin(labels, list(keep)), 255, 0).astype(np.uint8)


def _border_connected_background(candidates: np.ndarray) -> np.ndarray:
    """Keep only pale regions connected to the canvas border, not white motifs."""
    count, labels = cv2.connectedComponents(candidates, connectivity=8)
    if count <= 1:
        return np.zeros_like(candidates)
    border_labels = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    border_labels = border_labels[border_labels != 0]
    return np.where(np.isin(labels, border_labels), 255, 0).astype(np.uint8)


def _decontaminate_rgb(bgr: np.ndarray, alpha: np.ndarray, alpha_cutoff: int) -> np.ndarray:
    clean = bgr.copy()
    # Only opaque product pixels seed edge colours; feather pixels must not retain white.
    known = alpha >= max(250, alpha_cutoff)
    clean[~known] = 0
    kernel = np.ones((3, 3), np.uint8)
    # The default 33px feather has a radius of 16px.
    for _ in range(18):
        expanded = cv2.dilate(clean, kernel)
        neighbour = cv2.dilate(known.astype(np.uint8), kernel).astype(bool)
        fill = ~known & neighbour
        clean[fill] = expanded[fill]
        known |= fill
    return clean


def prepare_product_rgba(
    product: np.ndarray,
    white_threshold: int = 235,
    saturation_threshold: int = 45,
    alpha_blur: int = 33,
    alpha_cutoff: int = 18,
) -> np.ndarray:
    """Prepare BGRA data and remove bright, low-saturation backgrounds."""
    _validate_byte_threshold(white_threshold, "white_threshold")
    _validate_byte_threshold(saturation_threshold, "saturation_threshold")
    _validate_byte_threshold(alpha_cutoff, "alpha_cutoff")
    if product.ndim == 2:
        bgr = cv2.cvtColor(product, cv2.COLOR_GRAY2BGR)
        has_alpha = False
    elif product.ndim == 3 and product.shape[2] == 4:
        bgr = product[:, :, :3].copy()
        alpha = product[:, :, 3].copy()
        has_alpha = True
    elif product.ndim == 3 and product.shape[2] == 3:
        bgr = product.copy()
        has_alpha = False
    else:
        raise ValueError("Product image must be grayscale, BGR, or BGRA")

    if has_alpha:
        # Preserve authored colour/alpha; remove only effectively invisible alpha noise.
        alpha[alpha <= 1] = 0
    else:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        background = np.where(
            (hsv[:, :, 2] >= white_threshold) & (hsv[:, :, 1] <= saturation_threshold), 255, 0
        ).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        background = cv2.morphologyEx(background, cv2.MORPH_CLOSE, kernel)
        background = _border_connected_background(background)
        alpha = _keep_product_components(cv2.bitwise_not(background))
        if alpha_blur > 0:
            blur_kernel = alpha_blur if alpha_blur % 2 else alpha_blur + 1
            alpha = cv2.GaussianBlur(alpha, (blur_kernel, blur_kernel), 0)

    alpha[alpha < alpha_cutoff] = 0
    if not has_alpha:
        bgr = _decontaminate_rgb(bgr, alpha, alpha_cutoff)
    return np.dstack((bgr, alpha))
