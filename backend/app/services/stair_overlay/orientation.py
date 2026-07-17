"""Canonical orientation: arch at REAR/top and straight edge at FRONT/bottom."""

from __future__ import annotations

import cv2
import numpy as np

ORIENTATIONS = ("auto", "normal", "rotate-180")


def _edge_line_error(alpha: np.ndarray, top: bool) -> float:
    mask = alpha >= 32
    xs, values = [], []
    for x in range(mask.shape[1]):
        ys = np.flatnonzero(mask[:, x])
        if ys.size:
            xs.append(x)
            values.append(ys[0] if top else ys[-1])
    if len(xs) < 8:
        return float("inf")
    x, y = np.asarray(xs, np.float32), np.asarray(values, np.float32)
    lo, hi = np.quantile(x, [0.05, 0.95])
    keep = (x >= lo) & (x <= hi)
    residual = y[keep]-np.polyval(np.polyfit(x[keep], y[keep], 1), x[keep])
    return float(np.sqrt(np.mean(residual**2)))


def detect_orientation(alpha: np.ndarray) -> tuple[str, float]:
    """Return only normal/rotate-180 plus confidence; reflection is impossible."""
    top_error = _edge_line_error(alpha, True)
    bottom_error = _edge_line_error(alpha, False)
    if not np.isfinite(top_error+bottom_error):
        return "normal", 0.0
    scale = max(top_error, bottom_error, 1.0)
    confidence = abs(top_error-bottom_error)/scale
    # The straighter edge (smaller residual) must become the bottom/front.
    return ("rotate-180" if top_error < bottom_error else "normal"), confidence


def orient_product(product_rgba: np.ndarray, mode: str = "auto") -> tuple[np.ndarray, str, bool]:
    if mode not in ORIENTATIONS:
        raise ValueError("orientation must be auto, normal, or rotate-180")
    if mode == "normal":
        return product_rgba.copy(), "产品方向：normal（圆弧在上，直边在下）", False
    if mode == "rotate-180":
        return cv2.rotate(product_rgba, cv2.ROTATE_180), "产品方向：强制旋转180度", False
    detected, confidence = detect_orientation(product_rgba[:, :, 3])
    if confidence < 0.15:
        return (product_rgba.copy(),
                "警告：自动方向置信度低，保持 normal；可用 --orientation rotate-180 覆盖",
                True)
    if detected == "rotate-180":
        return cv2.rotate(product_rgba, cv2.ROTATE_180), "产品方向：自动旋转180度", False
    return product_rgba.copy(), "产品方向：自动识别正常（normal）", False
