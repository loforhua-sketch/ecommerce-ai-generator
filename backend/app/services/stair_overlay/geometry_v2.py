"""MVP-04 explicit six-point tread and riser geometry.

This module is intentionally independent from the legacy four-point fitting
helpers.  V1 data can be loaded for annotation, but is never renderable with a
fold until the two physical riser points have been supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


POINT_NAMES = (
    "rear_left", "rear_right", "front_right", "front_left",
    "fold_bottom_right", "fold_bottom_left",
)


def _point(value: Any, name: str) -> np.ndarray:
    point = np.asarray(value, dtype=np.float32)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError(f"{name} must be one finite [x, y] point")
    return point


def _signed_area(quad: np.ndarray) -> float:
    x, y = quad[:, 0], quad[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _segments_cross(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    def side(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        pq, pr = q - p, r - p
        return float(pq[0] * pr[1] - pq[1] * pr[0])
    return side(a, b, c) * side(a, b, d) < 0 and side(c, d, a) * side(c, d, b) < 0


def _validate_quad(quad: np.ndarray, name: str) -> None:
    if abs(_signed_area(quad)) <= 1e-3:
        raise ValueError(f"{name} quadrilateral area must be greater than zero")
    if (_segments_cross(quad[0], quad[1], quad[2], quad[3]) or
            _segments_cross(quad[1], quad[2], quad[3], quad[0])):
        raise ValueError(f"{name} quadrilateral must not self-intersect")


@dataclass(frozen=True)
class TreadGeometry:
    rear_left: np.ndarray
    rear_right: np.ndarray
    front_right: np.ndarray
    front_left: np.ndarray
    fold_bottom_right: np.ndarray
    fold_bottom_left: np.ndarray

    @classmethod
    def from_json(cls, record: dict[str, Any]) -> "TreadGeometry":
        if not isinstance(record, dict):
            raise ValueError("tread record must be an object")
        if all(name in record for name in POINT_NAMES):
            values = [record[name] for name in POINT_NAMES]
        elif isinstance(record.get("points"), list) and len(record["points"]) == 6:
            values = record["points"]
        else:
            missing = [name for name in POINT_NAMES if name not in record]
            raise ValueError("six-point tread is incomplete: " + ", ".join(missing))
        geometry = cls(*(_point(value, name) for name, value in zip(POINT_NAMES, values)))
        geometry.validate()
        return geometry

    def to_json(self, index: int | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if index is not None:
            result["index"] = index
        for name in POINT_NAMES:
            result[name] = [float(v) for v in getattr(self, name)]
        return result

    @property
    def top_quad(self) -> np.ndarray:
        return np.asarray([self.rear_left, self.rear_right,
                           self.front_right, self.front_left], np.float32)

    @property
    def fold_quad(self) -> np.ndarray:
        return np.asarray([self.front_left, self.front_right,
                           self.fold_bottom_right, self.fold_bottom_left], np.float32)

    def validate(self, image_size: tuple[int, int] | None = None) -> None:
        points = np.asarray([getattr(self, name) for name in POINT_NAMES], np.float32)
        if points.shape != (6, 2) or not np.isfinite(points).all():
            raise ValueError("all six tread points are required and must be finite")
        if image_size is not None:
            width, height = image_size
            if np.any(points[:, 0] < 0) or np.any(points[:, 0] >= width) or \
                    np.any(points[:, 1] < 0) or np.any(points[:, 1] >= height):
                raise ValueError("tread point lies outside the source image")
        if float(np.mean(points[2:4, 1])) <= float(np.mean(points[0:2, 1])):
            raise ValueError("front edge must be in front of the rear edge")
        if float(np.mean(points[4:6, 1])) <= float(np.mean(points[2:4, 1])):
            raise ValueError("fold_bottom average Y must be greater than front average Y")
        _validate_quad(self.top_quad, "top")
        _validate_quad(self.fold_quad, "fold")
        if (_segments_cross(self.rear_left, self.front_left,
                            self.rear_right, self.front_right) or
                _segments_cross(self.front_left, self.fold_bottom_left,
                                self.front_right, self.fold_bottom_right)):
            raise ValueError("left/right point order crosses")
        top_polygon = self.top_quad.astype(np.float32)
        for point in (self.fold_bottom_left, self.fold_bottom_right):
            if cv2.pointPolygonTest(top_polygon, tuple(map(float, point)), False) >= 0:
                raise ValueError("fold_bottom cannot lie inside the tread polygon")


@dataclass(frozen=True)
class TreadGeometryDocument:
    image_width: int
    image_height: int
    treads: tuple[TreadGeometry, ...]
    tread_reference_width_cm: float | None = None
    depth_reference_cm: float | None = None
    needs_fold_annotation: bool = False
    legacy_treads: tuple[np.ndarray, ...] = ()

    @property
    def renderable_with_fold(self) -> bool:
        return bool(self.treads) and not self.needs_fold_annotation


def load_tread_geometry(path: Path, expected_size: tuple[int, int] | None = None) -> TreadGeometryDocument:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("treads"), list):
        raise ValueError("treads.json must contain a treads array")
    width, height = payload.get("image_width"), payload.get("image_height")
    if not isinstance(width, int) or not isinstance(height, int):
        raise ValueError("treads.json requires image_width and image_height")
    if expected_size is not None and (width, height) != expected_size:
        raise ValueError(f"coordinate image is {width}x{height}, expected {expected_size[0]}x{expected_size[1]}")
    records = payload["treads"]
    is_v2 = payload.get("version") == 2 and payload.get("point_schema") == "six-point-tread-fold"
    if not is_v2:
        legacy = []
        for record in records:
            points = np.asarray(record.get("points") if isinstance(record, dict) else None,
                                dtype=np.float32)
            if points.shape != (4, 2) or not np.isfinite(points).all():
                raise ValueError("legacy tread records must contain four finite points")
            legacy.append(points)
        return TreadGeometryDocument(width, height, (), None, None, True, tuple(legacy))
    geometries = tuple(TreadGeometry.from_json(record) for record in records)
    if not geometries:
        raise ValueError("at least one tread is required")
    for geometry in geometries:
        geometry.validate((width, height))
    scale = payload.get("scene_scale") or {}
    reference = scale.get("tread_reference_width_cm")
    depth = scale.get("depth_reference_cm")
    if reference is not None and (not isinstance(reference, (int, float)) or reference <= 65):
        raise ValueError("tread_reference_width_cm must be greater than 65")
    return TreadGeometryDocument(width, height, geometries,
                                 float(reference) if reference is not None else None,
                                 float(depth) if depth is not None else None)


def save_tread_geometry(path: Path, image_size: tuple[int, int],
                        treads: list[TreadGeometry],
                        tread_reference_width_cm: float | None = None,
                        depth_reference_cm: float | None = None) -> None:
    width, height = image_size
    for geometry in treads:
        geometry.validate(image_size)
    payload: dict[str, Any] = {
        "version": 2,
        "point_schema": "six-point-tread-fold",
        "image_width": width,
        "image_height": height,
        "treads": [geometry.to_json(i) for i, geometry in enumerate(treads, 1)],
    }
    if tread_reference_width_cm is not None or depth_reference_cm is not None:
        payload["scene_scale"] = {}
        if tread_reference_width_cm is not None:
            payload["scene_scale"]["tread_reference_width_cm"] = tread_reference_width_cm
        if depth_reference_cm is not None:
            payload["scene_scale"]["depth_reference_cm"] = depth_reference_cm
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def centered_segment(left: np.ndarray, right: np.ndarray, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < ratio <= 1:
        raise ValueError("product width ratio must be in (0, 1]")
    midpoint = (left + right) * 0.5
    half = (right - left) * (ratio * 0.5)
    return (midpoint - half).astype(np.float32), (midpoint + half).astype(np.float32)


def product_target_geometry(geometry: TreadGeometry, width_ratio: float,
                            depth_ratio: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Center the same physical width and depth model on every annotated tread."""
    rl, rr = centered_segment(geometry.rear_left, geometry.rear_right, width_ratio)
    fl, fr = centered_segment(geometry.front_left, geometry.front_right, width_ratio)
    fbl, fbr = centered_segment(geometry.fold_bottom_left,
                                geometry.fold_bottom_right, width_ratio)
    if not 0 < depth_ratio <= 1:
        raise ValueError("depth ratio must be in (0, 1]")
    rl = fl + (rl - fl) * depth_ratio
    rr = fr + (rr - fr) * depth_ratio
    return (np.asarray([rl, rr, fr, fl], np.float32),
            np.asarray([fl, fr, fbr, fbl], np.float32))
