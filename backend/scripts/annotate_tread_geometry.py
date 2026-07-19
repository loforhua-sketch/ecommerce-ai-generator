"""Add fold-bottom points to legacy four-point tread annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.stair_overlay.geometry_v2 import (  # noqa: E402
    POINT_NAMES, TreadGeometry, load_tread_geometry, save_tread_geometry,
)


COLORS = {
    "rear": (0, 210, 0),
    "front": (255, 90, 0),
    "fold": (0, 0, 255),
}
FIXED_POINT_COUNT = 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="为旧版四点踏面补标折边底边")
    parser.add_argument("--input-image", type=Path, required=True)
    parser.add_argument("--treads-json", type=Path, required=True)
    return parser


def _load_seed(path: Path) -> tuple[list[list[list[float]]], dict]:
    if not path.exists():
        raise ValueError("treads.json does not exist")
    raw = json.loads(path.read_text(encoding="utf-8"))
    document = load_tread_geometry(path)
    if document.needs_fold_annotation:
        # Legacy point order is fixed: rear_left, rear_right, front_right, front_left.
        return [[point.tolist() for point in quad] for quad in document.legacy_treads], raw
    return [
        [[float(value) for value in getattr(tread, name)] for name in POINT_NAMES]
        for tread in document.treads
    ], raw


def _draw(image: np.ndarray, treads: list[list[list[float]]], current: int) -> np.ndarray:
    canvas = image.copy()
    overlay = canvas.copy()
    for points in treads:
        if len(points) >= 6:
            fold_quad = np.asarray([points[3], points[2], points[4], points[5]], np.int32)
            cv2.fillPoly(overlay, [fold_quad], COLORS["fold"], cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

    for index, points in enumerate(treads):
        pts = np.asarray(points, np.int32)
        cv2.line(canvas, tuple(pts[0]), tuple(pts[1]), COLORS["rear"], 3, cv2.LINE_AA)
        cv2.line(canvas, tuple(pts[3]), tuple(pts[2]), COLORS["front"], 3, cv2.LINE_AA)
        cv2.polylines(canvas, [pts[:4]], True, COLORS["rear"], 2, cv2.LINE_AA)
        if len(pts) >= 6:
            cv2.line(canvas, tuple(pts[5]), tuple(pts[4]), COLORS["fold"], 3, cv2.LINE_AA)
            cv2.polylines(canvas, [pts[[3, 2, 4, 5]]], True, COLORS["fold"], 2,
                          cv2.LINE_AA)
        for point_index, point in enumerate(pts):
            color = COLORS["rear" if point_index < 2 else "front" if point_index < 4 else "fold"]
            cv2.circle(canvas, tuple(point), 7, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, str(index + 1), tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX,
                    .8, (255, 255, 255), 2, cv2.LINE_AA)

    point_offset = len(treads[current]) - FIXED_POINT_COUNT
    current_point = POINT_NAMES[FIXED_POINT_COUNT + min(point_offset, 1)]
    next_point = (POINT_NAMES[FIXED_POINT_COUNT + 1]
                  if point_offset == 0 else "下一层" if current + 1 < len(treads) else "保存")
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 116), (25, 25, 25), -1)
    lines = (
        f"当前级：{current + 1}/{len(treads)}",
        f"当前点击：{current_point}",
        f"下一点：{next_point}",
    )
    for line_index, line in enumerate(lines):
        cv2.putText(canvas, line, (20, 28 + line_index * 27), cv2.FONT_HERSHEY_SIMPLEX,
                    .65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Z/Backspace undo | R redo current | S save | Esc cancel",
                (20, 108), cv2.FONT_HERSHEY_SIMPLEX, .45, (210, 210, 210), 1, cv2.LINE_AA)
    return canvas


def annotate(image: np.ndarray, seed: list[list[list[float]]]) -> list[TreadGeometry] | None:
    if not seed or any(len(points) < FIXED_POINT_COUNT for points in seed):
        raise ValueError("every tread must contain the four fixed legacy points")
    window = "MVP-04 fold-bottom annotation"
    treads = [[list(point) for point in points] for points in seed]
    current = next((i for i, points in enumerate(treads) if len(points) < 6), len(treads) - 1)

    def click(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        nonlocal current
        if event != cv2.EVENT_LBUTTONDOWN or len(treads[current]) >= 6:
            return
        treads[current].append([float(x), float(y)])
        if len(treads[current]) == 6 and current + 1 < len(treads):
            current += 1

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, click)
    while True:
        cv2.imshow(window, _draw(image, treads, current))
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("z"), ord("Z"), 8, 127):
            if len(treads[current]) > FIXED_POINT_COUNT:
                treads[current].pop()
            elif current > 0 and len(treads[current - 1]) > FIXED_POINT_COUNT:
                current -= 1
                treads[current].pop()
        elif key in (ord("r"), ord("R")):
            del treads[current][FIXED_POINT_COUNT:]
        elif key in (ord("s"), ord("S")):
            if not all(len(points) == 6 for points in treads):
                print("无法保存：所有层都必须标注两个 fold_bottom 点", file=sys.stderr)
                continue
            try:
                result = [
                    TreadGeometry(*[np.asarray(point, np.float32) for point in points])
                    for points in treads
                ]
                for geometry in result:
                    geometry.validate((image.shape[1], image.shape[0]))
                cv2.destroyAllWindows()
                return result
            except ValueError as exc:
                print(f"无法保存：{exc}", file=sys.stderr)
        elif key == 27:
            cv2.destroyAllWindows()
            return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    treads_json = args.treads_json.resolve()
    image = cv2.imread(str(args.input_image), cv2.IMREAD_COLOR)
    if image is None:
        print(f"无法读取图片：{args.input_image}", file=sys.stderr)
        return 1
    try:
        seed, _raw = _load_seed(treads_json)
        result = annotate(image, seed)
        if result is None:
            print("已取消，未修改 JSON")
            return 1
        # Validation is complete before either the backup or original file is written.
        print(f"SAVE TARGET: {treads_json}")
        backup = treads_json.with_name("treads.v1.backup.json")
        shutil.copyfile(treads_json, backup)
        save_tread_geometry(treads_json, (image.shape[1], image.shape[0]), result)

        try:
            saved = json.loads(treads_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("saved file could not be read back as JSON") from exc
        if saved.get("version") != 2:
            raise RuntimeError("saved file validation failed: version != 2")
        if saved.get("point_schema") != "six-point-tread-fold":
            raise RuntimeError(
                'saved file validation failed: point_schema != "six-point-tread-fold"'
            )
        saved_treads = saved.get("treads")
        if not isinstance(saved_treads, list) or any(
            not isinstance(tread, dict)
            or "fold_bottom_right" not in tread
            or "fold_bottom_left" not in tread
            for tread in saved_treads
        ):
            raise RuntimeError(
                "saved file validation failed: every tread must contain "
                "fold_bottom_right and fold_bottom_left"
            )

        print(f"SAVED V2: {treads_json}")
        return 0
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
