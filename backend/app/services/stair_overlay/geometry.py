"""Geometry helpers and backwards-compatible tread coordinate persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TreadEdges:
    """Authoritative tread semantics; input order is TL, TR, BR, BL."""

    rear_left: np.ndarray
    rear_right: np.ndarray
    front_right: np.ndarray
    front_left: np.ndarray

    def as_quad(self) -> np.ndarray:
        return np.asarray(
            [self.rear_left, self.rear_right, self.front_right, self.front_left],
            dtype=np.float32,
        )


def get_tread_edges(points: Iterable[Iterable[float]]) -> TreadEdges:
    quad = np.asarray(points, dtype=np.float32)
    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        raise ValueError("tread points must be four finite points in TL, TR, BR, BL order")
    return TreadEdges(*(point.copy() for point in quad))


def validate_margin(margin: float) -> float:
    if not 0.0 <= margin <= 0.25:
        raise ValueError("边距 margin 必须在 0 到 0.25 之间")
    return margin


def inset_quad(
    points: Iterable[Iterable[float]],
    left_margin: float = 0.08,
    right_margin: float = 0.08,
    rear_margin: float = 0.08,
    front_margin: float = 0.0,
) -> np.ndarray:
    """Inset a perspective quad using bilinear coordinates, not image axes."""
    margins = (left_margin, right_margin, rear_margin, front_margin)
    for value in margins:
        validate_margin(value)
    quad = np.asarray(points, dtype=np.float32)
    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        raise ValueError("踏面坐标必须是 4 个有效的二维点")

    def point(u: float, v: float) -> np.ndarray:
        tl, tr, br, bl = quad
        return ((1-u)*(1-v)*tl + u*(1-v)*tr + u*v*br + (1-u)*v*bl)

    return np.asarray([
        point(left_margin, rear_margin),
        point(1.0-right_margin, rear_margin),
        point(1.0-right_margin, 1.0-front_margin),
        point(left_margin, 1.0-front_margin),
    ], dtype=np.float32)


def shrink_quad(points: Iterable[Iterable[float]], margin: float = 0.06) -> np.ndarray:
    """Legacy symmetric inset (kept for callers and old tests)."""
    return inset_quad(points, margin / 2, margin / 2, margin / 2, margin / 2)


def _validate_points(points: object, width: int, height: int, tread_index: int) -> list[list[int]]:
    if not isinstance(points, list) or len(points) != 4:
        raise ValueError(f"第 {tread_index} 个踏面必须包含 4 个点")
    result = []
    for point_index, point in enumerate(points, 1):
        if (not isinstance(point, (list, tuple)) or len(point) != 2 or
                isinstance(point[0], bool) or isinstance(point[1], bool) or
                not isinstance(point[0], (int, float)) or not isinstance(point[1], (int, float))):
            raise ValueError(f"第 {tread_index} 个踏面的第 {point_index} 个点格式错误")
        x, y = float(point[0]), float(point[1])
        if not np.isfinite([x, y]).all() or not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"第 {tread_index} 个踏面的第 {point_index} 个点超出图片范围")
        result.append([int(round(x)), int(round(y))])
    return result


def save_treads(path: Path, image_width: int, image_height: int,
                 treads: Iterable[Iterable[Iterable[float]]]) -> None:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("图片尺寸必须为正整数")
    raw = list(treads)
    if not raw:
        raise ValueError("至少需要选择一个踏面")
    records = [{"index": i, "points": _validate_points(list(p), image_width, image_height, i)}
               for i, p in enumerate(raw, 1)]
    payload = {"version": 2, "image_width": image_width, "image_height": image_height, "treads": records}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_treads(path: Path, expected_width: int, expected_height: int) -> list[list[list[int]]]:
    if not path.is_file():
        raise FileNotFoundError(f"踏面坐标文件不存在: {path}；请先不带 --reuse-points 运行")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"踏面坐标 JSON 已损坏: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("踏面坐标 JSON 顶层必须是对象")
    width, height = payload.get("image_width"), payload.get("image_height")
    if not isinstance(width, int) or not isinstance(height, int):
        raise ValueError("踏面坐标 JSON 缺少有效的 image_width/image_height")
    if (width, height) != (expected_width, expected_height):
        raise ValueError(f"坐标图片尺寸为 {width}x{height}，当前为 {expected_width}x{expected_height}")
    records = payload.get("treads")
    if not isinstance(records, list) or not records:
        raise ValueError("踏面坐标 JSON 至少需要一个踏面")
    result = []
    for i, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f"第 {i} 个踏面记录格式错误")
        # Version 1 used the same records, but index may be omitted by hand-authored files.
        if "index" in record and record["index"] != i:
            raise ValueError("踏面 index 必须从 1 开始连续编号")
        result.append(_validate_points(record.get("points"), width, height, i))
    return result
