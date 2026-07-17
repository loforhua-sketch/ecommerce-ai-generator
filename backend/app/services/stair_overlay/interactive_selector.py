"""OpenCV mouse UI for selecting stair tread quadrilaterals."""

from __future__ import annotations

import cv2
import numpy as np


def select_treads(image: np.ndarray, max_width: int = 1400, max_height: int = 850) -> list[list[list[int]]]:
    """Interactively select treads and return coordinates in original image space."""
    height, width = image.shape[:2]
    scale = min(1.0, max_width / width, max_height / height)
    display_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    base = cv2.resize(image, display_size, interpolation=cv2.INTER_AREA) if scale < 1 else image.copy()
    current: list[tuple[int, int]] = []
    confirmed: list[list[tuple[int, int]]] = []
    window = "Stair tread selector"

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(current) < 4:
            current.append((x, y))

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    try:
        while True:
            canvas = base.copy()
            for index, quad in enumerate(confirmed, start=1):
                polygon = np.asarray(quad, dtype=np.int32)
                cv2.polylines(canvas, [polygon], True, (0, 220, 0), 2, cv2.LINE_AA)
                center = tuple(np.mean(polygon, axis=0).astype(int))
                cv2.putText(canvas, str(index), center, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            for point in current:
                cv2.circle(canvas, point, 5, (0, 0, 255), -1, cv2.LINE_AA)
            if len(current) > 1:
                cv2.polylines(canvas, [np.asarray(current, dtype=np.int32)], False, (0, 165, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "Click: TL, TR, BR, BL | Enter: confirm | Backspace: undo point", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "U: undo tread | Q/Esc: finish", (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 255), 2, cv2.LINE_AA)
            cv2.imshow(window, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10) and len(current) == 4:
                confirmed.append(current.copy())
                current.clear()
            elif key in (8, 127) and current:
                current.pop()
            elif key in (ord("u"), ord("U")) and confirmed:
                confirmed.pop()
            elif key in (ord("q"), ord("Q"), 27):
                break
    finally:
        cv2.destroyWindow(window)
    if not confirmed:
        raise ValueError("未选择任何踏面：请至少选择并按 Enter 确认一个踏面")
    return [
        [[min(width - 1, max(0, round(x / scale))), min(height - 1, max(0, round(y / scale)))] for x, y in quad]
        for quad in confirmed
    ]
