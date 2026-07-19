"""Perspective warping, realistic folds, shadows, and alpha compositing."""

from __future__ import annotations

import cv2
import numpy as np

from .geometry import get_tread_edges, inset_quad
from .geometry_model import (
    MatDimensions, StairMatRenderConfig, build_warped_geometry,
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


def _clean_master_alpha(master: np.ndarray) -> np.ndarray:
    """Remove opaque black pixels leaked from a transparent source background."""
    clean = master.copy()
    leaked = (np.max(clean[:, :, :3], axis=2) <= 12).astype(np.uint8)
    count, labels = cv2.connectedComponents(leaked, connectivity=8)
    if count <= 1:
        return clean
    border_labels = np.unique(np.concatenate((
        labels[0], labels[-1], labels[:, 0], labels[:, -1],
    )))
    border_labels = border_labels[border_labels != 0]
    if border_labels.size:
        exterior = np.isin(labels, border_labels)
        clean[exterior] = 0
    clean[clean[:, :, 3] < 2] = 0
    return clean


def _warp_premultiplied(source: np.ndarray, target: np.ndarray,
                        output_size: tuple[int, int], alpha_cutoff: int) -> np.ndarray:
    """Warp one face of a product mesh; callers composite the product only once."""
    h, w = source.shape[:2]
    src = np.float32([(0, 0), (w-1, 0), (w-1, h-1), (0, h-1)])
    clean = source.copy()
    clean[clean[:, :, 3] == 0] = 0
    alpha = clean[:, :, 3:4].astype(np.float32) / 255.0
    premul = np.dstack((clean[:, :, :3].astype(np.float32)*alpha,
                        clean[:, :, 3].astype(np.float32)))
    matrix = cv2.getPerspectiveTransform(src, np.asarray(target, np.float32))
    warped = cv2.warpPerspective(premul, matrix, output_size, flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))
    warped[warped[:, :, 3] < alpha_cutoff] = 0
    return warped


def _quad_mask(quad: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    mask = np.zeros((output_size[1], output_size[0]), np.uint8)
    cv2.fillConvexPoly(mask, np.rint(quad).astype(np.int32), 255, lineType=cv2.LINE_8)
    return mask


def warp_product_mesh(canvas_rgba: np.ndarray, top_height: int, geometry: object,
                      output_size: tuple[int, int], alpha_cutoff: int = 18,
                      fold_enabled: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render the continuous top/fold mesh and return one product instance.

    Source and destination face order is consistently TL, TR, BR, BL.  The two
    warps are face rasterization, not separate product layers; they are joined
    here and the returned instance is composited exactly once by the caller.
    """
    if not 1 < top_height < canvas_rgba.shape[0]:
        raise ValueError("invalid product mesh top height")
    top_source = canvas_rgba[:top_height]
    # The first fold source row is the same authored hinge row as top[-1].
    fold_source = np.concatenate((top_source[-1:], canvas_rgba[top_height:]), axis=0)
    top = _warp_premultiplied(top_source, geometry.top_quad, output_size, alpha_cutoff)
    fold = np.zeros_like(top)
    if fold_enabled:
        fold = _warp_premultiplied(fold_source, geometry.fold_quad, output_size, alpha_cutoff)
        riser_mask = _quad_mask(geometry.fold_quad, output_size)
        fold[riser_mask == 0] = 0
        # Perspective interpolation can leak an antialiased fold sample back
        # into the tread by more than the permitted hinge pixel.  Remove that
        # raster fringe before validation; this also prevents lateral spikes.
        yy, xx = np.indices(fold.shape[:2])
        a, b = geometry.hinge_left, geometry.hinge_right
        edge = b-a
        signed = (edge[0]*(yy-a[1])-edge[1]*(xx-a[0])) / max(float(np.linalg.norm(edge)), 1e-6)
        fold[(top[:, :, 3] >= 24) & (np.abs(signed) > 1.0)] = 0
        _validate_product_faces(top[:, :, 3], fold[:, :, 3], riser_mask, geometry)
    # Fold owns the shared one-pixel hinge; remove it from top before joining.
    joined = top.copy()
    joined[fold[:, :, 3] > 0] = fold[fold[:, :, 3] > 0]
    return joined, top, fold


def _validate_product_faces(top_alpha: np.ndarray, fold_alpha: np.ndarray,
                            riser_mask: np.ndarray, geometry: object) -> None:
    top_y = float(np.mean(geometry.fold_quad[:2, 1]))
    bottom_y = float(np.mean(geometry.fold_quad[2:, 1]))
    if bottom_y <= top_y:
        raise ValueError("fold bottom must be below fold top")
    hinge_error = max(float(np.linalg.norm(geometry.fold_quad[0]-geometry.hinge_left)),
                      float(np.linalg.norm(geometry.fold_quad[1]-geometry.hinge_right)))
    if hinge_error > 1.0:
        raise ValueError("fold top and tread front edge differ by more than 1px")
    if np.any((fold_alpha >= 24) & (riser_mask == 0)):
        raise ValueError("fold mask exceeds riser quad")
    binary = (fold_alpha >= 24).astype(np.uint8)
    components = cv2.connectedComponents(binary, connectivity=8)[0]-1
    if components != 1:
        raise ValueError("fold alpha must contain exactly one connected region")
    # Faces may share only the rasterized hinge band (one pixel plus AA tolerance).
    overlap = (top_alpha >= 24) & (fold_alpha >= 24)
    if np.any(overlap):
        ys, xs = np.nonzero(overlap)
        a, b = geometry.hinge_left, geometry.hinge_right
        edge = b-a
        distance = np.abs(edge[0]*(ys-a[1])-edge[1]*(xs-a[0])) / max(float(np.linalg.norm(edge)), 1e-6)
        if float(distance.max()) > 1.0:
            raise ValueError("top and fold overlap outside the 1px hinge")


def alpha_composite_premultiplied(background: np.ndarray,
                                  warped: np.ndarray) -> np.ndarray:
    """Composite one premultiplied warped canvas without edge unpremultiplication."""
    alpha = warped[:, :, 3:4] / 255.0
    value = warped[:, :, :3] + background.astype(np.float32) * (1.0 - alpha)
    return np.clip(value, 0, 255).astype(np.uint8)


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
    master_top_height: int | None = None,
    alpha_cutoff: int = 18,
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
    if master_top_height is None:
        raise ValueError("master_top_height is required for the frozen top+fold canvas")
    master_canvas = _clean_master_alpha(product_rgba)
    final_mask = np.zeros(stair_bgr.shape[:2], np.uint8)
    combined_hinge = np.zeros((*stair_bgr.shape[:2], 4), np.uint8)
    debug_top = cv2.cvtColor(stair_bgr, cv2.COLOR_BGR2BGRA)
    debug_fold = debug_top.copy()
    debug_hinge = debug_top.copy()
    fold_render_passes = 0
    fold_only = np.zeros((*stair_bgr.shape[:2], 4), np.uint8)
    debug_geometry = cv2.cvtColor(stair_bgr, cv2.COLOR_BGR2BGRA)
    for points in treads:
        edges = get_tread_edges(points)
        destination = inset_quad(edges.as_quad(), left, right, rear, front_margin)
        geometry = build_warped_geometry(destination, fold_ratio, dimensions)
        failed = [name for name, passed in quality_check_warped_geometry(
            geometry, dimensions).items() if not passed]
        if failed:
            raise ValueError("warped geometry quality check failed: " + ", ".join(failed))
        top_master = master_canvas[:master_top_height]
        rear_samples, front_samples = perspective_semantic_samples(top_master, destination)
        _validate_boundary_semantics(rear_samples, front_samples, geometry)
        warped, warped_top, warped_fold = warp_product_mesh(
            master_canvas, master_top_height, geometry, size, alpha_cutoff, fold,
        )
        fold_render_passes += 1
        final_mask = np.maximum(final_mask, warped[:, :, 3].astype(np.uint8))
        fold_only = cv2.max(fold_only, warped_fold.astype(np.uint8))
        result = alpha_composite_premultiplied(result, warped)
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
            bottom_left = tuple(np.rint(geometry.fold_quad[3]).astype(int))
            bottom_right = tuple(np.rint(geometry.fold_quad[2]).astype(int))
            cv2.line(debug_geometry, fl, fr, (255, 0, 0, 255), 3, cv2.LINE_AA)
            cv2.line(debug_geometry, p1, p2, (0, 255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(debug_geometry, bottom_left, bottom_right, (0, 0, 255, 255), 3, cv2.LINE_AA)
            cv2.arrowedLine(debug_geometry, hinge_mid, fold_mid, (0, 0, 255, 255), 3,
                            cv2.LINE_AA, tipLength=.25)
    if fold_render_passes != len(treads):
        raise ValueError("fold_render_passes must equal one per tread")
    if debug_layers is not None:
        debug_layers["top"] = debug_top
        debug_layers["fold"] = debug_fold
        debug_layers["hinge"] = debug_hinge
        debug_layers["final_mask"] = final_mask
        debug_layers["fold_height_px"] = np.asarray([geometry.fold_height_px], np.float32)
        debug_layers["tread_fold_geometry"] = debug_geometry
        debug_layers["fold_only"] = fold_only
        debug_layers["fold_render_passes"] = np.asarray([fold_render_passes], np.int32)
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
