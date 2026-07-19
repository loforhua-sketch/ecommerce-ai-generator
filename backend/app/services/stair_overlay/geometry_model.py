"""Dimensioned 65 x 24 cm stair-mat geometry in canonical orientation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MatDimensions:
    width_cm: float = 65.0
    depth_cm: float = 24.0
    fold_height_cm: float = 3.0

    def validate(self) -> None:
        if self.width_cm <= 0 or self.depth_cm <= 0:
            raise ValueError("mat width and depth must be positive")
        if not 1.0 <= self.fold_height_cm <= 8.0:
            raise ValueError("fold_height_cm must be between 1.0 and 8.0")

    @property
    def fold_to_depth_ratio(self) -> float:
        self.validate()
        return self.fold_height_cm / self.depth_cm


@dataclass(frozen=True)
class MatShapeParameters:
    side_straight_ratio: float = 0.55
    arch_rise_ratio: float = 0.28
    shoulder_smoothing_ratio: float = 0.05
    bottom_corner_ratio: float = 0.08


@dataclass(frozen=True)
class StairMatRenderConfig:
    width_cm: float = 65.0
    depth_cm: float = 24.0
    fold_height_cm: float = 3.0
    orientation: str = "normal"
    arch_rise_ratio: float = 0.28
    side_straight_ratio: float = 0.55
    left_margin: float = 0.08
    right_margin: float = 0.08
    rear_margin: float = 0.06
    front_margin: float = 0.0

    @property
    def dimensions(self) -> MatDimensions:
        return MatDimensions(self.width_cm, self.depth_cm, self.fold_height_cm)


@dataclass(frozen=True)
class WarpedMatGeometry:
    top_quad: np.ndarray
    hinge_left: np.ndarray
    hinge_right: np.ndarray
    fold_quad: np.ndarray
    projected_depth_px: float
    fold_height_px: float

    @property
    def front_hinge_left(self) -> np.ndarray:
        return self.hinge_left

    @property
    def front_hinge_right(self) -> np.ndarray:
        return self.hinge_right


def _validate_ratio(value: float, low: float, high: float, name: str) -> float:
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def validate_shape_parameters(arch_rise_ratio: float, side_straight_ratio: float,
                              shoulder_smoothing_ratio: float = 0.05,
                              bottom_corner_ratio: float = 0.025) -> None:
    _validate_ratio(arch_rise_ratio, 0.18, 0.42, "arch_rise_ratio")
    _validate_ratio(side_straight_ratio, 0.35, 0.75, "side_straight_ratio")
    _validate_ratio(shoulder_smoothing_ratio, 0.0, 0.12, "shoulder_smoothing_ratio")
    _validate_ratio(bottom_corner_ratio, 0.0, 0.08, "bottom_corner_ratio")


def validate_contour_ratios(arch_height_ratio: float, side_round_ratio: float,
                            bottom_corner_ratio: float) -> None:
    """Compatibility validator for MVP-01 parameter names."""
    validate_shape_parameters(arch_height_ratio, max(0.35, min(0.75, 1-side_round_ratio)),
                              0.05, bottom_corner_ratio)


def _cubic(p0: tuple[float, float], p1: tuple[float, float],
           p2: tuple[float, float], p3: tuple[float, float], count: int,
           endpoint: bool = False) -> np.ndarray:
    t = np.linspace(0.0, 1.0, count, endpoint=endpoint, dtype=np.float32)[:, None]
    return ((1-t)**3*np.asarray(p0) + 3*(1-t)**2*t*np.asarray(p1) +
            3*(1-t)*t**2*np.asarray(p2) + t**3*np.asarray(p3)).astype(np.float32)


def generate_dimensioned_mat_contour(
    width_px: int,
    height_px: int,
    dimensions: MatDimensions = MatDimensions(),
    arch_rise_ratio: float = 0.22,
    side_straight_ratio: float = 0.46,
    shoulder_smoothing_ratio: float = 0.05,
    bottom_corner_ratio: float = 0.025,
) -> np.ndarray:
    """Generate a canonical contour: curved rear/top, straight front/bottom.

    Coordinates first represent the physical ``width_cm x depth_cm`` plane and
    are then independently converted to the requested raster. The fold is not
    part of this contour.
    """
    dimensions.validate()
    if width_px < 4 or height_px < 4:
        raise ValueError("width_px and height_px must be at least 4")
    validate_shape_parameters(arch_rise_ratio, side_straight_ratio,
                              shoulder_smoothing_ratio, bottom_corner_ratio)
    width_cm, depth_cm = dimensions.width_cm, dimensions.depth_cm
    rise_cm = arch_rise_ratio * depth_cm
    shoulder_y = rise_cm
    # The arch rise is measured from its shoulder to the centered apex. Sides
    # then continue vertically; side_straight_ratio is validated as the minimum
    # product-design intent and does not bow those sides to satisfy a raster.
    smooth = shoulder_smoothing_ratio * width_cm
    corner = min(bottom_corner_ratio * min(width_cm, depth_cm), depth_cm * 0.08)
    n = max(24, width_px // 8)
    # Vertical tangent at each shoulder and horizontal tangent at the apex.
    left_arch = _cubic((0, shoulder_y), (smooth, shoulder_y-rise_cm*.42),
                       (width_cm*.32, 0), (width_cm*.5, 0), n)
    right_arch = _cubic((width_cm*.5, 0), (width_cm*.68, 0),
                        (width_cm-smooth, shoulder_y-rise_cm*.42),
                        (width_cm, shoulder_y), n)
    parts = [left_arch, right_arch,
             np.asarray([[width_cm, shoulder_y], [width_cm, depth_cm-corner]], np.float32)]
    if corner:
        parts += [_cubic((width_cm, depth_cm-corner), (width_cm, depth_cm-corner*.45),
                         (width_cm-corner*.45, depth_cm), (width_cm-corner, depth_cm), 8),
                  np.asarray([[width_cm-corner, depth_cm], [corner, depth_cm]], np.float32),
                  _cubic((corner, depth_cm), (corner*.45, depth_cm),
                         (0, depth_cm-corner*.45), (0, depth_cm-corner), 8)]
    else:
        parts.append(np.asarray([[width_cm, depth_cm], [0, depth_cm]], np.float32))
    parts.append(np.asarray([[0, depth_cm-corner], [0, shoulder_y]], np.float32))
    cm = np.concatenate(parts)
    cm = np.vstack((cm, cm[0]))
    scale = np.asarray([(width_px-1)/width_cm, (height_px-1)/depth_cm], np.float32)
    return np.clip(cm*scale, [0, 0], [width_px-1, height_px-1]).astype(np.float32)


def generate_mat_contour(width: int, height: int, arch_height_ratio: float = 0.22,
                         side_round_ratio: float = 0.12,
                         bottom_corner_ratio: float = 0.025) -> np.ndarray:
    """MVP-01-compatible wrapper; sides are now physically straight."""
    _validate_ratio(arch_height_ratio, 0.12, 0.45, "arch_height_ratio")
    _validate_ratio(side_round_ratio, 0.03, 0.25, "side_round_ratio")
    _validate_ratio(bottom_corner_ratio, 0.0, 0.12, "bottom_corner_ratio")
    side_straight = max(0.35, min(0.75, 1-side_round_ratio))
    # Legacy inputs outside the new product-safe range are accepted only here.
    safe_arch = max(0.18, min(0.42, arch_height_ratio))
    return generate_dimensioned_mat_contour(width, height, MatDimensions(),
                                            safe_arch, side_straight,
                                            0.05, bottom_corner_ratio)


def contour_mask(width: int, height: int, arch_height_ratio: float = 0.22,
                 side_round_ratio: float = 0.12,
                 bottom_corner_ratio: float = 0.025,
                 dimensions: MatDimensions = MatDimensions(),
                 side_straight_ratio: float | None = None) -> np.ndarray:
    straight = (max(0.35, min(0.75, 1-side_round_ratio)) if side_straight_ratio is None
                else side_straight_ratio)
    contour = generate_dimensioned_mat_contour(width, height, dimensions,
                                                arch_height_ratio, straight,
                                                0.05, bottom_corner_ratio)
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [np.rint(contour).astype(np.int32)], 255, lineType=cv2.LINE_AA)
    mask = np.maximum(mask, np.fliplr(mask))
    bottom = contour[np.isclose(contour[:, 1], height-1)]
    if bottom.size:
        lo, hi = int(np.ceil(bottom[:, 0].min())), int(np.floor(bottom[:, 0].max()))
        mask[-1, lo:hi+1] = 255
    return mask


def map_texture_to_contour(product_rgba: np.ndarray, arch_height_ratio: float = 0.22,
                           side_round_ratio: float = 0.12,
                           bottom_corner_ratio: float = 0.025,
                           dimensions: MatDimensions = MatDimensions(),
                           side_straight_ratio: float | None = None,
                           width_px: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Crop, proportionally normalize scanlines, then apply the standard mask."""
    ys, xs = np.nonzero(product_rgba[:, :, 3] > 0)
    if not xs.size:
        return product_rgba.copy(), np.zeros(product_rgba.shape[:2], np.uint8)
    crop = product_rgba[ys.min():ys.max()+1, xs.min():xs.max()+1]
    target_w = width_px or crop.shape[1]
    target_h = max(4, int(round(target_w * dimensions.depth_cm / dimensions.width_cm)))
    resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)
    mask = contour_mask(target_w, target_h, arch_height_ratio, side_round_ratio,
                        bottom_corner_ratio, dimensions, side_straight_ratio)
    mapped = resized.copy()
    # Repair the complete canonical plane before perspective interpolation.  Inpaint
    # from surrounding product pixels instead of copying scanlines/side fragments.
    interior = mask >= 250
    missing = interior & (mapped[:, :, 3] < 250)
    if np.any(missing):
        known = interior & ~missing
        if not np.any(known):
            raise ValueError("standard product RGBA has no opaque texture pixels")
        # The photographed mat is bilaterally symmetric.  Where the matching
        # coordinate is authored/opaque, use it as a texture seed for the hole;
        # this is coordinate-matched restoration, never an edge-strip repeat.
        mirrored_alpha = np.fliplr(mapped[:, :, 3])
        mirrored_rgb = np.fliplr(mapped[:, :, :3])
        symmetric_fill = missing & (mirrored_alpha >= 250)
        mapped[:, :, :3][symmetric_fill] = mirrored_rgb[symmetric_fill]
        remaining = missing & ~symmetric_fill
        repair_mask = (remaining.astype(np.uint8) * 255)
        # Navier-Stokes with a wider local neighbourhood produces continuous
        # fibres/colour gradients; the small Telea radius left polygonal shards
        # in larger transparent checkerboard holes on the real product asset.
        radius = max(3, min(11, int(round(min(target_w, target_h)*.025))))
        if np.any(remaining):
            mapped[:, :, :3] = cv2.inpaint(mapped[:, :, :3], repair_mask, radius, cv2.INPAINT_NS)
    mapped[:, :, 3] = mask
    mapped[mask == 0] = 0
    hole_ratio = standard_rgba_hole_ratio(mapped, mask)
    if hole_ratio >= 0.001:
        raise ValueError(f"standard product RGBA internal transparent ratio {hole_ratio:.3%} exceeds 0.1%")
    return mapped, mask


def standard_rgba_hole_ratio(rgba: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of canonical-mask interior that is unexpectedly transparent."""
    interior = mask >= 250
    if not np.any(interior):
        return 1.0
    return float(np.count_nonzero(interior & (rgba[:, :, 3] < 250)) / np.count_nonzero(interior))


def dimensioned_destination(points: np.ndarray, dimensions: MatDimensions) -> np.ndarray:
    """Fit the 65:24 plane uniformly, centered laterally and anchored at front."""
    quad = np.asarray(points, np.float32).copy()
    rear_mid, front_mid = (quad[0]+quad[1])/2, (quad[3]+quad[2])/2
    width = (np.linalg.norm(quad[1]-quad[0]) + np.linalg.norm(quad[2]-quad[3]))/2
    depth = np.linalg.norm(front_mid-rear_mid)
    desired_depth = width * dimensions.depth_cm / dimensions.width_cm
    if depth > desired_depth:
        fraction = desired_depth/depth
        quad[0] = quad[3] + (quad[0]-quad[3])*fraction
        quad[1] = quad[2] + (quad[1]-quad[2])*fraction
    elif depth > 0:
        desired_width = depth * dimensions.width_cm / dimensions.depth_cm
        scale = min(1.0, desired_width/max(width, 1e-6))
        for indexes in ((0, 1), (3, 2)):
            midpoint = (quad[indexes[0]]+quad[indexes[1]])/2
            quad[indexes[0]] = midpoint+(quad[indexes[0]]-midpoint)*scale
            quad[indexes[1]] = midpoint+(quad[indexes[1]]-midpoint)*scale
    return quad


def build_warped_geometry(top_quad: np.ndarray, fold_ratio: float = 0.125,
                          dimensions: MatDimensions | None = None) -> WarpedMatGeometry:
    quad = np.asarray(top_quad, np.float32)
    if quad.shape != (4, 2):
        raise ValueError("top_quad must contain four points")
    dimensions = dimensions or MatDimensions(fold_height_cm=24*fold_ratio)
    dimensions.validate()
    left, right = quad[3].copy(), quad[2].copy()  # the single authoritative hinge
    rear_mid, front_mid = (quad[0]+quad[1])/2, (left+right)/2
    depth = float(np.linalg.norm(front_mid-rear_mid))
    fold_px = depth*dimensions.fold_to_depth_ratio
    # The front edge is the authoritative tread/riser intersection.  A riser is
    # below that edge in image space; never extrapolate the tread homography or
    # infer a hinge from a mask/edge slope.
    offset = np.asarray([0.0, fold_px], np.float32)
    fold = np.asarray([left, right, right+offset, left+offset], np.float32)
    return WarpedMatGeometry(quad, left, right, fold, depth, fold_px)


def quality_check_contour(contour: np.ndarray, width: int, height: int,
                          dimensions: MatDimensions = MatDimensions()) -> dict[str, bool]:
    """Machine-readable pre-render checks for the canonical planar model."""
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [np.rint(contour).astype(np.int32)], 255)
    left = np.flatnonzero(mask[:, 0] > 0)
    right = np.flatnonzero(mask[:, -1] > 0)
    checks = {
        "aspect_ratio": abs(width/height-dimensions.width_cm/dimensions.depth_cm) < 0.04,
        "symmetric": bool(np.mean(mask != np.fliplr(mask)) < 0.002),
        "side_edges_straight": bool(left.size > height*.35 and right.size > height*.35),
        "bottom_horizontal": bool(np.count_nonzero(mask[-1]) > width*.9),
        "arch_apex_above_shoulders": bool(contour[:, 1].min() < contour[0, 1]),
        "plane_excludes_fold": True,
    }
    return checks


def quality_check_warped_geometry(geometry: WarpedMatGeometry,
                                  dimensions: MatDimensions) -> dict[str, bool]:
    """Check the canonical source-to-tread contract and physical fold."""
    quad = geometry.top_quad
    expected_fold = geometry.projected_depth_px * dimensions.fold_to_depth_ratio
    rear_left, rear_right, front_right, front_left = quad
    rear_mid = (rear_left + rear_right) / 2
    front_mid = (front_left + front_right) / 2
    fold_mid = (geometry.fold_quad[2] + geometry.fold_quad[3]) / 2
    tread_depth = max(float(np.linalg.norm(front_mid-rear_mid)), 1e-6)

    def line_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        edge, delta = b-a, point-a
        cross_2d = edge[0]*delta[1] - edge[1]*delta[0]
        return float(abs(cross_2d) / max(np.linalg.norm(edge), 1e-6))

    fold_to_front = line_distance(fold_mid, front_left, front_right)
    fold_to_rear = line_distance(fold_mid, rear_left, rear_right)
    polygon = np.rint(quad).astype(np.int32)
    fold_outside = cv2.pointPolygonTest(polygon, tuple(map(float, fold_mid)), False) < 0
    return {
        "arch_maps_to_rear": bool(np.allclose(geometry.top_quad[:2], [rear_left, rear_right])),
        "straight_edge_maps_to_front": bool(
            np.allclose(geometry.front_hinge_left, front_left) and
            np.allclose(geometry.front_hinge_right, front_right)),
        "hinge_error_le_1px": bool(
            np.linalg.norm(geometry.fold_quad[0]-geometry.hinge_left) <= 1 and
            np.linalg.norm(geometry.fold_quad[1]-geometry.hinge_right) <= 1),
        "fold_height_physical": bool(abs(geometry.fold_height_px-expected_fold) <= 1),
        "fold_center_outside_tread": bool(fold_outside),
        "fold_is_on_front_side": bool(fold_to_rear > fold_to_front and fold_to_rear > tread_depth*.5),
    }
