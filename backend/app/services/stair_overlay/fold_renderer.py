"""Texture-preserving front fold rendering."""

from __future__ import annotations

import cv2
import numpy as np

from .geometry_model import MatDimensions, build_warped_geometry


def validate_fold_ratio(value: float) -> float:
    if not 0.03 <= value <= 0.25:
        raise ValueError("fold_ratio 必须在 0.03 到 0.25 之间")
    return value


def fold_quad(destination: np.ndarray, fold_ratio: float = 0.125,
              dimensions: MatDimensions | None = None) -> np.ndarray:
    validate_fold_ratio(fold_ratio)
    return build_warped_geometry(destination, fold_ratio, dimensions).fold_quad


def render_fold(product_rgba: np.ndarray, destination: np.ndarray, output_size: tuple[int, int],
                fold_ratio: float = 0.125, texture_ratio: float = 0.10,
                darkening: float = 0.15, bottom_corner_ratio: float = 0.25,
                dimensions: MatDimensions | None = None) -> np.ndarray:
    validate_fold_ratio(fold_ratio)
    if not 0.08 <= texture_ratio <= 0.15:
        raise ValueError("fold_texture_ratio 必须在 0.08 到 0.15 之间")
    if not 0 <= darkening <= 1:
        raise ValueError("fold_darkening 必须在 0 到 1 之间")
    h, w = product_rgba.shape[:2]
    foreground_rows = np.flatnonzero(np.any(product_rgba[:, :, 3] > 0, axis=1))
    if foreground_rows.size == 0:
        return np.zeros((output_size[1], output_size[0], 4), np.uint8)
    content_top, content_bottom = int(foreground_rows[0]), int(foreground_rows[-1]) + 1
    strip_h = max(1, int(round((content_bottom-content_top) * texture_ratio)))
    strip = product_rgba[content_bottom-strip_h:content_bottom].copy()
    strip[:, :, :3] = np.clip(strip[:, :, :3].astype(np.float32) * (1-darkening), 0, 255).astype(np.uint8)
    # A one-pixel strip is valid for tiny synthetic/products; use a virtual unit height.
    source_bottom = max(1, strip_h-1)
    if strip_h == 1:
        strip = np.repeat(strip, 2, axis=0)
    src = np.float32([[0, 0], [w-1, 0], [w-1, source_bottom], [0, source_bottom]])
    target = fold_quad(destination, fold_ratio, dimensions)
    matrix = cv2.getPerspectiveTransform(src, target)
    out = cv2.warpPerspective(strip, matrix, output_size, flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    # Soft hinge and a subtle lower thickness/contact edge, both perspective-scaled.
    alpha = out[:, :, 3]
    fold_h = max(1.0, (np.linalg.norm(target[3]-target[0])+np.linalg.norm(target[2]-target[1]))/2)
    fold_w = max(1.0, (np.linalg.norm(target[1]-target[0])+np.linalg.norm(target[2]-target[3]))/2)
    # Radius is a small 20-35% portion of fold height, never a semicircle.
    radius_ratio = float(np.clip(bottom_corner_ratio, 0.20, 0.35))
    radius = int(max(1, round(fold_h*radius_ratio)))
    if radius:
        local_mask = np.full((strip.shape[0], strip.shape[1]), 255, np.uint8)
        local_r = min(radius, local_mask.shape[0]-1, local_mask.shape[1]//2)
        if local_r > 0:
            cv2.rectangle(local_mask, (0, local_mask.shape[0]-local_r),
                          (local_mask.shape[1]-1, local_mask.shape[0]-1), 0, -1)
            cv2.rectangle(local_mask, (local_r, local_mask.shape[0]-local_r),
                          (local_mask.shape[1]-local_r-1, local_mask.shape[0]-1), 255, -1)
            cv2.ellipse(local_mask, (local_r, local_mask.shape[0]-local_r-1),
                        (local_r, local_r), 0, 0, 90, 255, -1)
            cv2.ellipse(local_mask, (local_mask.shape[1]-local_r-1, local_mask.shape[0]-local_r-1),
                        (local_r, local_r), 0, 90, 180, 255, -1)
            warped_mask = cv2.warpPerspective(local_mask, matrix, output_size, flags=cv2.INTER_LINEAR)
            alpha = np.minimum(alpha, warped_mask)
    out[:, :, 3] = cv2.GaussianBlur(alpha, (3, 3), 0)
    lower = cv2.subtract(cv2.dilate(alpha, np.ones((3, 3), np.uint8)), alpha)
    out[:, :, :3] = np.where(lower[:, :, None] > 0,
                              (out[:, :, :3].astype(np.float32) * 0.75).astype(np.uint8), out[:, :, :3])
    out[:, :, 3] = np.maximum(out[:, :, 3], (lower.astype(np.float32)*0.22).astype(np.uint8))
    return out
