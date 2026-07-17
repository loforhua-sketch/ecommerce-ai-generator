"""Perspective warping, realistic folds, shadows, and alpha compositing."""

from __future__ import annotations

import cv2
import numpy as np

from .fold_renderer import render_fold
from .geometry import get_tread_edges, inset_quad
from .geometry_model import (
    MatDimensions, StairMatRenderConfig, build_warped_geometry, dimensioned_destination,
    quality_check_warped_geometry,
)


def alpha_blend(background: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    if background.ndim != 3 or background.shape[2] != 3:
        raise ValueError("Background must be a BGR image")
    if overlay.ndim != 3 or overlay.shape[2] != 4 or overlay.shape[:2] != background.shape[:2]:
        raise ValueError("Overlay must be a same-sized BGRA image")
    alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
    value = overlay[:, :, :3].astype(np.float32)*alpha + background.astype(np.float32)*(1-alpha)
    return np.clip(value, 0, 255).astype(np.uint8)


def warp_product(product_rgba: np.ndarray, destination: np.ndarray,
                 output_size: tuple[int, int]) -> np.ndarray:
    height, width = product_rgba.shape[:2]
    source_rear_left = (0, 0)
    source_rear_right = (width-1, 0)
    source_front_right = (width-1, height-1)
    source_front_left = (0, height-1)
    source_rear_front_quad = np.float32([
        source_rear_left, source_rear_right, source_front_right, source_front_left,
    ])
    target_rear_front_quad = np.asarray(destination, np.float32)
    matrix = cv2.getPerspectiveTransform(source_rear_front_quad, target_rear_front_quad)
    alpha = product_rgba[:, :, 3:4].astype(np.float32)/255
    premul = np.dstack((product_rgba[:, :, :3].astype(np.float32)*alpha, product_rgba[:, :, 3]))
    warped = cv2.warpPerspective(premul, matrix, output_size, flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    wa = warped[:, :, 3]
    rgb = np.where(wa[:, :, None] > 0, warped[:, :, :3]/np.maximum(wa[:, :, None]/255, 1e-6), 0)
    return np.dstack((np.clip(rgb, 0, 255), wa)).astype(np.uint8)


def perspective_semantic_samples(product_rgba: np.ndarray,
                                 target_rear_front_quad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map sampled curved-rear and straight-front product boundaries by physical meaning."""
    height, width = product_rgba.shape[:2]
    source_rear_front_quad = np.float32([
        (0, 0), (width-1, 0), (width-1, height-1), (0, height-1),
    ])
    matrix = cv2.getPerspectiveTransform(source_rear_front_quad,
                                         np.asarray(target_rear_front_quad, np.float32))
    alpha = product_rgba[:, :, 3] >= 24
    columns = np.linspace(0, width-1, 9).round().astype(int)
    rear, front = [], []
    for x in columns:
        ys = np.flatnonzero(alpha[:, x])
        if ys.size:
            rear.append((x, int(ys[0])))
            front.append((x, int(ys[-1])))
    if len(rear) < 2:
        raise ValueError("standard product has insufficient boundary samples")
    rear_mapped = cv2.perspectiveTransform(np.float32([rear]), matrix)[0]
    front_mapped = cv2.perspectiveTransform(np.float32([front]), matrix)[0]
    return rear_mapped, front_mapped


def _dynamic_shadow_kernel(shape: tuple[int, int]) -> int:
    kernel = min(21, max(7, int(round(min(shape)/80))))
    return kernel+1 if kernel % 2 == 0 else kernel


def shadow_overlay(alpha: np.ndarray, offset: int, opacity: float, contact: bool = False) -> np.ndarray:
    h, w = alpha.shape
    if contact:
        shifted = cv2.subtract(cv2.dilate(alpha, np.ones((3, 3), np.uint8)), alpha)
        shifted = cv2.warpAffine(shifted, np.float32([[1, 0, 0], [0, 1, 1]]), (w, h))
        blurred = cv2.GaussianBlur(shifted, (3, 3), 0)
        opacity = min(opacity, 0.18)
    else:
        shifted = cv2.warpAffine(alpha, np.float32([[1, 0, 0], [0, 1, offset]]), (w, h),
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        k = _dynamic_shadow_kernel(alpha.shape)
        blurred = cv2.GaussianBlur(shifted, (k, k), 0)
    layer = np.zeros((h, w, 4), np.uint8)
    layer[:, :, :3] = (34, 38, 42)  # deep warm brown-grey in BGR, never pure black
    layer[:, :, 3] = np.clip(blurred.astype(np.float32)*opacity, 0, 255).astype(np.uint8)
    return layer


def _edge_thickness(top: np.ndarray) -> np.ndarray:
    """Expand the authored edge by one pixel using nearby product colours."""
    kernel = np.ones((3, 3), np.uint8)
    expanded = cv2.dilate(top[:, :, 3], kernel)
    ring = cv2.subtract(expanded, top[:, :, 3])
    layer = np.zeros_like(top)
    # Channel dilation samples the immediately adjacent binding instead of painting black.
    sampled = cv2.dilate(top[:, :, :3], kernel)
    layer[:, :, :3] = np.clip(sampled.astype(np.float32)*0.82, 0, 255).astype(np.uint8)
    layer[:, :, 3] = (ring.astype(np.float32)*0.55).astype(np.uint8)
    return layer


def compose_stair_overlay(
    stair_bgr: np.ndarray,
    product_rgba: np.ndarray,
    treads: list[list[list[int]]],
    margin: float | None = None,
    shadow: bool = True,
    shadow_offset: int = 6,
    shadow_opacity: float = 0.22,
    contact_shadow: bool = True,
    left_margin: float | None = None,
    right_margin: float | None = None,
    rear_margin: float | None = None,
    front_margin: float = 0.0,
    fold: bool = True,
    fold_ratio: float = 0.125,
    fold_height_cm: float | None = 3.0,
    fold_darkening: float = 0.15,
    fold_texture_ratio: float = 0.10,
    bottom_corner_ratio: float = 0.25,
    debug_layers: dict[str, np.ndarray] | None = None,
    config: StairMatRenderConfig | None = None,
) -> np.ndarray:
    """Compose all treads. Explicit side/rear margins override legacy ``margin``."""
    config = config or StairMatRenderConfig()
    base = 0.08 if margin is None else margin
    left = config.left_margin if left_margin is None else left_margin
    right = config.right_margin if right_margin is None else right_margin
    rear = config.rear_margin if rear_margin is None else rear_margin
    if margin is not None:
        left = base if left_margin is None else left_margin
        right = base if right_margin is None else right_margin
        rear = base if rear_margin is None else rear_margin
    dimensions = MatDimensions(config.width_cm, config.depth_cm,
                               config.fold_height_cm if fold_height_cm is None else fold_height_cm)
    if not 0 <= shadow_opacity <= 1 or shadow_offset < 0:
        raise ValueError("阴影参数超出合法范围")
    if not treads:
        raise ValueError("At least one tread is required")
    result = stair_bgr.copy()
    size = (stair_bgr.shape[1], stair_bgr.shape[0])
    combined_top = np.zeros((*stair_bgr.shape[:2], 4), np.uint8)
    combined_fold = np.zeros_like(combined_top)
    combined_hinge = np.zeros_like(combined_top)
    debug_top = cv2.cvtColor(stair_bgr, cv2.COLOR_BGR2BGRA)
    debug_fold = debug_top.copy()
    debug_hinge = debug_top.copy()
    for points in treads:
        edges = get_tread_edges(points)
        destination = inset_quad(edges.as_quad(), left, right, rear, front_margin)
        destination = dimensioned_destination(destination, dimensions)
        geometry = build_warped_geometry(destination, fold_ratio, dimensions)
        failed = [name for name, passed in quality_check_warped_geometry(
            geometry, dimensions).items() if not passed]
        if failed:
            raise ValueError("warped geometry quality check failed: " + ", ".join(failed))
        rear_samples, front_samples = perspective_semantic_samples(product_rgba, destination)
        _validate_boundary_semantics(rear_samples, front_samples, geometry)
        top = warp_product(product_rgba, destination, size)
        fold_layer = render_fold(product_rgba, destination, size, fold_ratio,
                                 fold_texture_ratio, fold_darkening,
                                 bottom_corner_ratio, dimensions) if fold else np.zeros_like(top)
        if fold:
            _validate_fold_alpha(fold_layer[:, :, 3], geometry)
        total_alpha = np.maximum(top[:, :, 3], fold_layer[:, :, 3])
        if shadow:
            result = alpha_blend(result, shadow_overlay(total_alpha, shadow_offset, shadow_opacity))
        # The vertical face goes down first; the top face then closes/softens the hinge.
        if fold:
            result = alpha_blend(result, fold_layer)
            combined_fold = _merge_rgba(combined_fold, fold_layer)
        result = alpha_blend(result, _edge_thickness(top))
        result = alpha_blend(result, top)
        combined_top = _merge_rgba(combined_top, top)
        if contact_shadow:
            result = alpha_blend(result, shadow_overlay(total_alpha, 1, 0.14, contact=True))
        if debug_layers is not None:
            rear_left, rear_right, front_right, front_left = destination
            rl, rr = tuple(np.rint(rear_left).astype(int)), tuple(np.rint(rear_right).astype(int))
            fl, fr = tuple(np.rint(front_left).astype(int)), tuple(np.rint(front_right).astype(int))
            cv2.line(debug_top, rl, rr, (0, 190, 0, 255), 5, cv2.LINE_AA)
            cv2.line(debug_top, fl, fr, (255, 80, 0, 255), 5, cv2.LINE_AA)
            for rear_sample in rear_samples:
                cv2.circle(debug_top, tuple(np.rint(rear_sample).astype(int)), 7,
                           (0, 255, 0, 255), -1, cv2.LINE_AA)
            for front_sample in front_samples:
                cv2.circle(debug_top, tuple(np.rint(front_sample).astype(int)), 7,
                           (255, 80, 0, 255), -1, cv2.LINE_AA)

            tinted = np.zeros_like(fold_layer)
            tinted[:, :, 2] = 255
            tinted[:, :, 3] = (fold_layer[:, :, 3].astype(np.float32)*.55).astype(np.uint8)
            debug_fold = _merge_rgba(debug_fold, tinted)
            cv2.line(debug_fold, rl, rr, (0, 190, 0, 255), 5, cv2.LINE_AA)
            cv2.line(debug_fold, fl, fr, (0, 255, 255, 255), 5, cv2.LINE_AA)
            cv2.putText(debug_fold, "REAR - NO FOLD", rl, cv2.FONT_HERSHEY_SIMPLEX,
                        .8, (0, 190, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(debug_fold, "FRONT - FOLD DOWN 3CM", fl, cv2.FONT_HERSHEY_SIMPLEX,
                        .8, (0, 255, 255, 255), 2, cv2.LINE_AA)

            hinge = np.zeros_like(combined_hinge)
            p1 = tuple(np.rint(geometry.hinge_left).astype(int))
            p2 = tuple(np.rint(geometry.hinge_right).astype(int))
            cv2.line(hinge, p1, p2, (0, 255, 0, 255), 2, cv2.LINE_AA)
            fold_line = np.zeros_like(hinge)
            q1 = tuple(np.rint(geometry.fold_quad[0]).astype(int))
            q2 = tuple(np.rint(geometry.fold_quad[1]).astype(int))
            cv2.line(fold_line, q1, q2, (0, 0, 255, 255), 2, cv2.LINE_AA)
            overlap = cv2.add(hinge, fold_line)
            for point in (p1, p2):
                cv2.circle(overlap, point, 4, (0, 255, 255, 255), -1, cv2.LINE_AA)
            combined_hinge = cv2.max(combined_hinge, overlap)
            cv2.line(debug_hinge, rl, rr, (0, 190, 0, 255), 5, cv2.LINE_AA)
            cv2.line(debug_hinge, fl, fr, (255, 80, 0, 255), 5, cv2.LINE_AA)
            cv2.line(debug_hinge, p1, p2, (0, 255, 255, 255), 3, cv2.LINE_AA)
            fold_mid = tuple(np.rint((geometry.fold_quad[2]+geometry.fold_quad[3])/2).astype(int))
            hinge_mid = tuple(np.rint((geometry.hinge_left+geometry.hinge_right)/2).astype(int))
            cv2.arrowedLine(debug_hinge, hinge_mid, fold_mid, (0, 0, 255, 255), 4,
                            cv2.LINE_AA, tipLength=.25)
    if debug_layers is not None:
        debug_layers["top"] = debug_top
        debug_layers["fold"] = debug_fold
        debug_layers["hinge"] = debug_hinge
        debug_layers["fold_height_px"] = np.asarray([geometry.fold_height_px], np.float32)
    return result


def _validate_fold_alpha(alpha: np.ndarray, geometry: object) -> None:
    """Block a detached/rear fold before it can reach the final composite."""
    fold_points = np.column_stack(np.nonzero(alpha >= 24))[:, ::-1].astype(np.float32)
    if not fold_points.size:
        raise ValueError("front edge has no fold alpha")
    rear_left, rear_right = geometry.top_quad[:2]
    rear_edge = rear_right-rear_left
    rear_distances = np.abs(
        rear_edge[0]*(fold_points[:, 1]-rear_left[1]) -
        rear_edge[1]*(fold_points[:, 0]-rear_left[0])
    ) / max(float(np.linalg.norm(rear_edge)), 1e-6)
    if float(np.quantile(rear_distances, .05)) <= geometry.projected_depth_px*.5:
        raise ValueError("fold alpha appears near rear edge or behind top plane")


def _validate_boundary_semantics(rear_samples: np.ndarray, front_samples: np.ndarray,
                                 geometry: object) -> None:
    rear_mid = np.mean(geometry.top_quad[:2], axis=0)
    front_mid = (geometry.front_hinge_left + geometry.front_hinge_right) / 2
    axis = front_mid-rear_mid
    depth2 = max(float(np.dot(axis, axis)), 1e-6)
    rear_position = float(np.mean((rear_samples-rear_mid) @ axis) / depth2)
    front_position = float(np.mean((front_samples-rear_mid) @ axis) / depth2)
    if rear_position >= .5:
        raise ValueError("product arch is closer to front edge than rear edge")
    if front_position <= .5:
        raise ValueError("product straight edge is closer to rear edge than front edge")


def _merge_rgba(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    alpha = top[:, :, 3:4].astype(np.float32)/255
    out = bottom.copy()
    out[:, :, :3] = np.clip(top[:, :, :3]*alpha + bottom[:, :, :3]*(1-alpha), 0, 255)
    out[:, :, 3] = np.clip(top[:, :, 3] + bottom[:, :, 3]*(1-alpha[:, :, 0]), 0, 255)
    return out.astype(np.uint8)
