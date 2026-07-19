from pathlib import Path

import cv2
import numpy as np


MASTER_DIR = Path(__file__).resolve().parents[1] / "backend" / "test_assets" / "mvp01" / "master"


def _read(name: str) -> np.ndarray:
    path = MASTER_DIR / name
    assert path.is_file(), f"missing master asset: {path}"
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert image is not None, f"unreadable master asset: {path}"
    return image


def test_master_top_dimensions_are_65_by_24_cm() -> None:
    top = _read("master_top.png")
    assert top.shape == (480, 1300, 4)  # fixed scale: 20 px/cm


def test_master_fold_dimensions_are_65_by_3_cm() -> None:
    fold = _read("master_fold.png")
    assert fold.shape == (60, 1300, 4)  # fixed scale: 20 px/cm


def test_master_alpha_has_no_internal_holes() -> None:
    alpha = _read("master_top.png")[:, :, 3]
    transparent = (alpha < 24).astype(np.uint8)
    exterior = np.zeros_like(transparent)
    flood_mask = np.zeros((alpha.shape[0] + 2, alpha.shape[1] + 2), np.uint8)
    for seed in ((0, 0), (alpha.shape[1] - 1, 0),
                 (0, alpha.shape[0] - 1), (alpha.shape[1] - 1, alpha.shape[0] - 1)):
        if transparent[seed[1], seed[0]]:
            filled = transparent.copy()
            cv2.floodFill(filled, flood_mask.copy(), seed, 2)
            exterior |= (filled == 2).astype(np.uint8)
    internal_holes = (transparent == 1) & (exterior == 0)
    assert np.count_nonzero(internal_holes) == 0


def test_master_has_transparency_not_black_background() -> None:
    top = _read("master_top.png")
    alpha = top[:, :, 3]
    assert all(alpha[y, x] == 0 for x, y in (
        (0, 0), (top.shape[1] - 1, 0),
        (0, top.shape[0] - 1), (top.shape[1] - 1, top.shape[0] - 1),
    ))
    assert np.count_nonzero(alpha == 0) > top.shape[0] * top.shape[1] * 0.05
    # A baked rectangular background would make every border substantially opaque.
    assert np.count_nonzero(alpha[0] >= 24) < top.shape[1] * 0.10


def test_large_arc_is_on_top_and_bottom_is_straight() -> None:
    alpha = _read("master_top.png")[:, :, 3]
    solid = alpha >= 24
    top_y = np.argmax(solid, axis=0)
    center = alpha.shape[1] // 2
    shoulder = alpha.shape[1] // 10
    assert top_y[center] < top_y[shoulder]
    assert top_y[center] < top_y[-shoulder - 1]
    # The main bottom span is a horizontal, opaque shared edge; only small
    # symmetric corner transitions may be transparent.
    assert np.mean(solid[-1, shoulder:-shoulder]) == 1.0


def test_fold_texture_is_independent_from_cropped_top() -> None:
    top = _read("master_top.png")
    fold = _read("master_fold.png")
    assert not np.array_equal(fold[0], top[-1])
    assert not np.array_equal(fold[-1], top[-fold.shape[0]])
    assert not np.array_equal(fold[0], top[0])
