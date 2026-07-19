"""MVP-05 polished two-face renderer with halo-safe alpha handling."""

from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

from .geometry_v2 import TreadGeometry, product_target_geometry


@dataclass(frozen=True)
class FaceWarps:
    top: np.ndarray
    fold: np.ndarray
    product_instance: np.ndarray
    top_homography: np.ndarray
    fold_homography: np.ndarray
    top_quad: np.ndarray
    fold_quad: np.ndarray


def _quad_mask(quad: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    mask = np.zeros((size[1], size[0]), np.uint8)
    cv2.fillConvexPoly(mask, np.rint(quad).astype(np.int32), 255, lineType=cv2.LINE_8)
    return mask


def _warp(source: np.ndarray, destination: np.ndarray,
          size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = source.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    matrix = cv2.getPerspectiveTransform(src, destination.astype(np.float32))
    alpha = source[:, :, 3].astype(np.float32) / 255.0
    premul_rgb = source[:, :, :3].astype(np.float32) * alpha[:, :, None]
    warped_rgb = cv2.warpPerspective(
        premul_rgb, matrix, size, flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    warped_alpha = cv2.warpPerspective(
        alpha, matrix, size, flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    result = np.zeros((*warped_alpha.shape, 4), np.float32)
    present = warped_alpha > (1.0 / 255.0)
    result[present, :3] = warped_rgb[present] / warped_alpha[present, None]
    result[:, :, 3] = warped_alpha * 255.0
    result[~present] = 0
    return result, matrix


def _feather_alpha(face: np.ndarray, clip_mask: np.ndarray) -> np.ndarray:
    """Soften only the silhouette alpha; preserve texture and target geometry."""
    original = face[:, :, 3].copy()
    feathered = cv2.GaussianBlur(original, (3, 3), 0)
    feathered[clip_mask == 0] = 0
    # Supply nearest edge colour to pixels introduced by outward alpha feathering.
    added = (feathered > 0.5) & (original <= 0.5)
    if np.any(added):
        kernel = np.ones((3, 3), np.uint8)
        for channel in range(3):
            dilated = cv2.dilate(face[:, :, channel], kernel)
            face[:, :, channel][added] = dilated[added]
    face[:, :, 3] = feathered
    face[feathered <= 0.5] = 0
    return face


def _alpha_over(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    """Standard straight-alpha source-over returning straight BGRA."""
    a_top = top[:, :, 3:4] / 255.0
    a_bottom = bottom[:, :, 3:4] / 255.0
    out_a = a_top + a_bottom * (1.0 - a_top)
    premul = (top[:, :, :3] * a_top +
              bottom[:, :, :3] * a_bottom * (1.0 - a_top))
    out = np.zeros_like(top, dtype=np.float32)
    np.divide(premul, out_a, out=out[:, :, :3], where=out_a > 1e-6)
    out[:, :, 3:4] = out_a * 255.0
    out[out[:, :, 3] <= 0.5] = 0
    return out


def _line_distance(xs: np.ndarray, ys: np.ndarray,
                   a: np.ndarray, b: np.ndarray) -> np.ndarray:
    edge = b - a
    return np.abs(edge[0] * (ys - a[1]) - edge[1] * (xs - a[0])) / max(
        float(np.linalg.norm(edge)), 1e-6)


def render_product_instance(top_source: np.ndarray, fold_source: np.ndarray,
                            geometry: TreadGeometry, width_ratio: float,
                            output_size: tuple[int, int],
                            depth_ratio: float = 1.0) -> FaceWarps:
    geometry.validate()
    top_quad, fold_quad = product_target_geometry(geometry, width_ratio, depth_ratio)
    if not np.array_equal(top_quad[[3, 2]], fold_quad[:2]):
        raise ValueError("top front and fold top must share exact hinge coordinates")
    top, h_top = _warp(top_source, top_quad, output_size)
    shaded_fold = fold_source.copy()
    shaded_fold[:, :, :3] = np.clip(
        shaded_fold[:, :, :3].astype(np.float32) * .92, 0, 255).astype(np.uint8)
    # A low-opacity hinge occlusion shadow, confined to the fold texture.
    shadow_rows = min(3, shaded_fold.shape[0])
    if shadow_rows:
        shade = np.linspace(.94, 1.0, shadow_rows, dtype=np.float32)[:, None, None]
        shaded_fold[:shadow_rows, :, :3] = np.clip(
            shaded_fold[:shadow_rows, :, :3].astype(np.float32) * shade, 0, 255)
    fold, h_fold = _warp(shaded_fold, fold_quad, output_size)
    top_mask, fold_mask = _quad_mask(top_quad, output_size), _quad_mask(fold_quad, output_size)
    top[top_mask == 0] = 0
    fold[fold_mask == 0] = 0
    top = _feather_alpha(top, top_mask)
    fold = _feather_alpha(fold, fold_mask)

    # Clip antialias spill so the faces can overlap only in the one-pixel hinge band.
    overlap = (top[:, :, 3] >= 18) & (fold[:, :, 3] >= 18)
    ys, xs = np.nonzero(overlap)
    if xs.size:
        distances = _line_distance(xs, ys, top_quad[3], top_quad[2])
        fold[ys[distances > 1], xs[distances > 1]] = 0
    _quality_check_faces(top, fold, top_quad, fold_quad, fold_mask)

    # Fold first, then top. Top owns the hinge and covers exactly its raster edge.
    instance = _alpha_over(fold, top)
    return FaceWarps(top, fold, instance, h_top, h_fold, top_quad, fold_quad)


def _quality_check_faces(top: np.ndarray, fold: np.ndarray, top_quad: np.ndarray,
                         fold_quad: np.ndarray, fold_mask: np.ndarray) -> None:
    if float(np.mean(fold_quad[2:, 1])) <= float(np.mean(fold_quad[:2, 1])):
        raise ValueError("fold is inverted")
    hinge_error = float(np.max(np.linalg.norm(top_quad[[3, 2]] - fold_quad[:2], axis=1)))
    if hinge_error > 1:
        raise ValueError("top front and fold top error exceeds 1px")
    fold_binary = (fold[:, :, 3] >= 18).astype(np.uint8)
    if cv2.connectedComponents(fold_binary, connectivity=8)[0] - 1 != 1:
        raise ValueError("fold must have exactly one connected region")
    if np.any((fold[:, :, 3] >= 18) & (fold_mask == 0)):
        raise ValueError("fold alpha exceeds the fold target quadrilateral")
    overlap = (top[:, :, 3] >= 18) & (fold[:, :, 3] >= 18)
    ys, xs = np.nonzero(overlap)
    if xs.size and float(_line_distance(xs, ys, top_quad[3], top_quad[2]).max()) > 1:
        raise ValueError("top and fold overlap outside the hinge")


def _composite(background: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    alpha = overlay[:, :, 3:4] / 255.0
    return np.clip(overlay[:, :, :3] * alpha +
                   background.astype(np.float32) * (1 - alpha),
                   0, 255).astype(np.uint8)


def _assert_transparent_rgb_clear(rgba: np.ndarray) -> None:
    transparent = rgba[:, :, 3] <= 0.5
    if np.any(rgba[:, :, :3][transparent] > 1.0):
        raise ValueError("transparent pixels must have zero RGB")


def _edge_check(top: np.ndarray, fold: np.ndarray, composite: np.ndarray) -> np.ndarray:
    canvas = np.full((*composite.shape[:2], 3), 224, np.uint8)
    checker = ((np.indices(composite.shape[:2]).sum(axis=0) // 8) % 2) == 0
    canvas[checker] = 176
    alpha = composite[:, :, 3:4] / 255.0
    canvas = np.clip(composite[:, :, :3] * alpha + canvas * (1.0 - alpha),
                     0, 255).astype(np.uint8)
    top_edge = cv2.Canny(np.clip(top[:, :, 3], 0, 255).astype(np.uint8), 20, 80)
    fold_edge = cv2.Canny(np.clip(fold[:, :, 3], 0, 255).astype(np.uint8), 20, 80)
    canvas[top_edge > 0] = (0, 255, 0)
    canvas[fold_edge > 0] = (0, 128, 255)
    return canvas


def compose_overlay_v2(stair_bgr: np.ndarray, top_source: np.ndarray,
                       fold_source: np.ndarray, treads: tuple[TreadGeometry, ...],
                       tread_reference_width_cm: float,
                       depth_reference_cm: float | None = None,
                       debug: dict[str, np.ndarray] | None = None) -> np.ndarray:
    if tread_reference_width_cm <= 65:
        raise ValueError("tread_reference_width_cm must be greater than 65cm")
    width_ratio = 65.0 / tread_reference_width_cm
    # Without a depth calibration the complete annotated rear/front interval is
    # the one global model. It is never re-fit from an individual tread.
    depth_ratio = min(1.0, 24.0 / depth_reference_cm) if depth_reference_cm else 1.0
    size = (stair_bgr.shape[1], stair_bgr.shape[0])
    result = stair_bgr.copy()
    top_debug = np.zeros((*stair_bgr.shape[:2], 4), np.float32)
    fold_debug = np.zeros_like(top_debug)
    instance_debug = np.zeros_like(top_debug)
    final_mask = np.zeros(stair_bgr.shape[:2], np.uint8)
    geometry_debug = stair_bgr.copy()
    composite_count = 0
    ratios = []
    for index, tread in enumerate(treads, 1):
        faces = render_product_instance(top_source, fold_source, tread, width_ratio,
                                        size, depth_ratio)
        ratios.append(width_ratio)
        result = _composite(result, faces.product_instance)
        composite_count += 1
        final_mask = np.maximum(final_mask, faces.product_instance[:, :, 3].astype(np.uint8))
        top_debug = np.maximum(top_debug, faces.top)
        fold_debug = np.maximum(fold_debug, faces.fold)
        instance_debug = np.maximum(instance_debug, faces.product_instance)
        _draw_geometry(geometry_debug, tread, faces.top_quad, faces.fold_quad, index)
    if composite_count != len(treads):
        raise ValueError("each tread must be composited exactly once")
    if not ratios or max(ratios) - min(ratios) > 1e-9:
        raise ValueError("product width ratio differs between treads")
    if debug is not None:
        _assert_transparent_rgb_clear(top_debug)
        _assert_transparent_rgb_clear(fold_debug)
        _assert_transparent_rgb_clear(instance_debug)
        debug.update(geometry=geometry_debug,
                     top_warp=np.clip(top_debug, 0, 255).astype(np.uint8),
                     fold_warp=np.clip(fold_debug, 0, 255).astype(np.uint8),
                     product_instance=np.clip(instance_debug, 0, 255).astype(np.uint8),
                     final_mask=final_mask,
                     top_alpha=np.clip(top_debug[:, :, 3], 0, 255).astype(np.uint8),
                     fold_alpha=np.clip(fold_debug[:, :, 3], 0, 255).astype(np.uint8),
                     composite_alpha=np.clip(instance_debug[:, :, 3], 0, 255).astype(np.uint8),
                     edge_check=_edge_check(top_debug, fold_debug, instance_debug),
                     width_ratios=np.asarray(ratios),
                     composite_count=np.asarray([composite_count]))
    return result


def _draw_geometry(canvas: np.ndarray, tread: TreadGeometry, top: np.ndarray,
                   fold: np.ndarray, index: int) -> None:
    overlay = canvas.copy()
    cv2.fillConvexPoly(overlay, np.rint(top).astype(np.int32), (30, 170, 30))
    cv2.addWeighted(overlay, .20, canvas, .80, 0, canvas)
    overlay = canvas.copy()
    cv2.fillConvexPoly(overlay, np.rint(fold).astype(np.int32), (20, 20, 220))
    cv2.addWeighted(overlay, .20, canvas, .80, 0, canvas)
    cv2.line(canvas, tuple(tread.rear_left.astype(int)), tuple(tread.rear_right.astype(int)),
             (0, 210, 0), 3, cv2.LINE_AA)
    cv2.line(canvas, tuple(tread.front_left.astype(int)), tuple(tread.front_right.astype(int)),
             (255, 80, 0), 3, cv2.LINE_AA)
    cv2.line(canvas, tuple(tread.fold_bottom_left.astype(int)),
             tuple(tread.fold_bottom_right.astype(int)), (0, 0, 255), 3, cv2.LINE_AA)
    for a, b in ((top[0], top[3]), (top[1], top[2])):
        cv2.line(canvas, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)),
                 (0, 255, 255), 2, cv2.LINE_AA)
    hinge_mid = np.mean(fold[:2], axis=0).astype(int)
    bottom_mid = np.mean(fold[2:], axis=0).astype(int)
    cv2.arrowedLine(canvas, tuple(hinge_mid), tuple(bottom_mid), (0, 0, 255), 3,
                    cv2.LINE_AA, tipLength=.25)
    cv2.putText(canvas, str(index), tuple(np.rint(top[0]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2, cv2.LINE_AA)
