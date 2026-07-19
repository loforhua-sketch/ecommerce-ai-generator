import cv2
import numpy as np
import pytest

from backend.app.services.product_master import (
    decode_product_image,
    generate_product_master,
    save_product_master,
)


def _product_image() -> np.ndarray:
    image = np.full((240, 500, 3), 248, np.uint8)
    points = np.array([[50, 80], [250, 30], [450, 80], [450, 210], [50, 210]])
    cv2.fillPoly(image, [points], (35, 95, 185))
    cv2.circle(image, (150, 125), 18, (210, 60, 30), -1)
    return image


def test_fixed_master_has_dimensioned_transparent_png() -> None:
    master = generate_product_master(_product_image(), width_px=650)
    assert master.rgba.shape == (240, 650, 4)
    assert (master.width_cm, master.depth_cm) == (65, 24)
    assert master.rgba[0, 0, 3] == 0
    assert master.rgba[-1, master.width_px // 2, 3] == 255
    decoded = cv2.imdecode(np.frombuffer(master.png_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(decoded, master.rgba)


def test_master_generation_is_deterministic() -> None:
    first = generate_product_master(_product_image(), width_px=650)
    second = generate_product_master(_product_image(), width_px=650)
    np.testing.assert_array_equal(first.rgba, second.rgba)


def test_three_master_assets_are_saved(tmp_path) -> None:
    master = generate_product_master(_product_image(), width_px=650)
    outputs = save_product_master(master, tmp_path)
    assert set(outputs) == {"top", "fold", "mask", "debug", "debug_fold"}
    assert {path.name for path in outputs.values()} == {
        "master_top.png", "master_fold.png", "master_mask.png",
        "debug_master_shape.png", "debug_master_fold.png"
    }
    assert all(path.is_file() for path in outputs.values())
    top = cv2.imread(str(outputs["top"]), cv2.IMREAD_UNCHANGED)
    fold = cv2.imread(str(outputs["fold"]), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(outputs["mask"]), cv2.IMREAD_UNCHANGED)
    debug = cv2.imread(str(outputs["debug"]), cv2.IMREAD_UNCHANGED)
    debug_fold = cv2.imread(str(outputs["debug_fold"]), cv2.IMREAD_UNCHANGED)
    assert top.shape == (240, 650, 4)
    assert fold.shape == (30, 650, 4)
    assert mask.shape == (240, 650)
    assert debug.shape == (269, 650, 3)
    assert debug_fold.shape == (78, 650, 3)
    np.testing.assert_array_equal(mask, top[:, :, 3])
    np.testing.assert_array_equal(fold[0], top[-1])


def test_user_top_arc_is_preserved_not_regenerated() -> None:
    first_source = _product_image()
    second_source = np.full_like(first_source, 248)
    second_points = np.array([[50, 60], [180, 45], [250, 15], [320, 45],
                              [450, 60], [450, 210], [50, 210]])
    cv2.fillPoly(second_source, [second_points], (35, 95, 185))
    first = generate_product_master(first_source, width_px=650)
    second = generate_product_master(second_source, width_px=650)
    first_top = np.argmax(first.mask >= 24, axis=0)
    second_top = np.argmax(second.mask >= 24, axis=0)
    assert not np.array_equal(first_top, second_top)
    assert np.max(np.abs(second_top.astype(int) - first_top.astype(int))) > 10


def test_decode_rejects_invalid_upload() -> None:
    with pytest.raises(ValueError, match="无法读取"):
        decode_product_image(b"not-an-image")


@pytest.mark.parametrize("width", [259, 5201])
def test_master_width_is_bounded(width: int) -> None:
    with pytest.raises(ValueError, match="width_px"):
        generate_product_master(_product_image(), width_px=width)
