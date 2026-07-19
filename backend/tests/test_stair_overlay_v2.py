import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import pytest

from backend.app.services.stair_overlay.geometry_v2 import (
    TreadGeometry, centered_segment, load_tread_geometry,
    product_target_geometry, save_tread_geometry,
)
from backend.app.services.stair_overlay.renderer_v2 import (
    compose_overlay_v2, render_product_instance,
)
from backend.scripts.run_stair_overlay_v2 import build_parser, main


def _geometry(y: float = 10, scale: float = 1) -> TreadGeometry:
    return TreadGeometry(
        np.float32([10, y]), np.float32([110, y]),
        np.float32([115, y + 40 * scale]), np.float32([5, y + 40 * scale]),
        np.float32([115, y + 50 * scale]), np.float32([5, y + 50 * scale]),
    )


def _sources() -> tuple[np.ndarray, np.ndarray]:
    top = np.full((24, 65, 4), (20, 80, 210, 255), np.uint8)
    top[:3, :, 0] = 120
    fold = np.full((3, 65, 4), (60, 100, 180, 255), np.uint8)
    fold[0] = top[-1]
    return top, fold


def _temp() -> Path:
    path = Path(__file__).parent / f".v2_{uuid4().hex}"
    path.mkdir()
    return path


def test_v1_four_point_json_is_migration_only() -> None:
    directory = _temp()
    path = directory / "treads.json"
    try:
        path.write_text(json.dumps({"image_width": 120, "image_height": 80,
            "treads": [{"index": 1, "points": [[10, 10], [110, 10], [115, 50], [5, 50]]}]}),
            encoding="utf-8")
        document = load_tread_geometry(path)
        assert document.needs_fold_annotation
        assert not document.renderable_with_fold
        assert document.legacy_treads[0].shape == (4, 2)
    finally:
        path.unlink(missing_ok=True)
        directory.rmdir()


def test_v2_six_point_json_round_trip() -> None:
    directory = _temp()
    path = directory / "treads.json"
    try:
        save_tread_geometry(path, (130, 100), [_geometry()], 90)
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["version"] == 2
        assert raw["point_schema"] == "six-point-tread-fold"
        document = load_tread_geometry(path, (130, 100))
        assert document.renderable_with_fold
        assert document.tread_reference_width_cm == 90
        np.testing.assert_array_equal(document.treads[0].fold_bottom_left, [5, 60])
    finally:
        path.unlink(missing_ok=True)
        directory.rmdir()


def test_six_point_order_and_fold_direction_validation() -> None:
    _geometry().validate()
    bad_fold = _geometry()
    bad_fold = TreadGeometry(bad_fold.rear_left, bad_fold.rear_right,
        bad_fold.front_right, bad_fold.front_left,
        np.float32([115, 45]), np.float32([5, 45]))
    with pytest.raises(ValueError, match="fold_bottom average Y"):
        bad_fold.validate()
    crossed = TreadGeometry(np.float32([10, 10]), np.float32([110, 10]),
        np.float32([5, 50]), np.float32([115, 50]),
        np.float32([5, 60]), np.float32([115, 60]))
    with pytest.raises(ValueError, match="self-intersect|crosses"):
        crossed.validate()


def test_top_and_fold_use_two_homographies_and_exact_hinge() -> None:
    top, fold = _sources()
    faces = render_product_instance(top, fold, _geometry(), 65 / 90, (130, 100))
    np.testing.assert_array_equal(faces.top_quad[[3, 2]], faces.fold_quad[:2])
    assert not np.allclose(faces.top_homography, faces.fold_homography)
    assert cv2.connectedComponents((faces.fold[:, :, 3] >= 18).astype(np.uint8))[0] - 1 == 1
    assert np.mean(faces.fold_quad[2:, 1]) > np.mean(faces.fold_quad[:2, 1])


def test_one_product_instance_and_no_duplicate_fold(monkeypatch) -> None:
    import backend.app.services.stair_overlay.renderer_v2 as renderer
    calls = 0
    original = renderer._composite

    def counted(background, overlay):
        nonlocal calls
        calls += 1
        return original(background, overlay)

    monkeypatch.setattr(renderer, "_composite", counted)
    top, fold = _sources()
    debug = {}
    stair = np.zeros((140, 140, 3), np.uint8)
    geometries = (_geometry(10), _geometry(75))
    result = compose_overlay_v2(stair, top, fold, geometries, 90, debug=debug)
    assert calls == len(geometries)
    assert int(debug["composite_count"][0]) == len(geometries)
    assert result.shape == stair.shape
    assert cv2.connectedComponents((debug["fold_warp"][:, :, 3] >= 18).astype(np.uint8))[0] - 1 == 2


def test_width_ratio_is_identical_across_treads_and_visual_size_is_perspective() -> None:
    small = _geometry(10, .5)
    large = _geometry(50, 1)
    ratio = 65 / 90
    small_top, _ = product_target_geometry(small, ratio)
    large_top, _ = product_target_geometry(large, ratio)
    assert np.linalg.norm(small_top[1] - small_top[0]) == pytest.approx(
        np.linalg.norm(small.rear_right - small.rear_left) * ratio)
    assert np.linalg.norm(large_top[2] - large_top[3]) == pytest.approx(
        np.linalg.norm(large.front_right - large.front_left) * ratio)
    left, right = centered_segment(large.front_left, large.front_right, ratio)
    assert np.linalg.norm(right - left) / np.linalg.norm(
        large.front_right - large.front_left) == pytest.approx(ratio)


def test_missing_reference_width_blocks_and_does_not_write_result() -> None:
    directory = _temp()
    master = directory / "master"
    master.mkdir()
    treads = directory / "treads.json"
    image = directory / "stair.png"
    top, fold = _sources()
    try:
        cv2.imwrite(str(image), np.zeros((100, 130, 3), np.uint8))
        cv2.imwrite(str(master / "master_top.png"), top)
        cv2.imwrite(str(master / "master_fold.png"), fold)
        save_tread_geometry(treads, (130, 100), [_geometry()])
        assert main(["--reuse-points", "--input-image", str(image), "--treads-json", str(treads),
                     "--master-dir", str(master), "--output-dir", str(directory)]) == 1
        assert not (directory / "result_v2.png").exists()
    finally:
        for child in master.glob("*"):
            child.unlink()
        master.rmdir()
        for child in directory.glob("*"):
            child.unlink()
        directory.rmdir()


def test_v2_cli_has_reference_width_and_does_not_import_legacy_renderer() -> None:
    args = build_parser().parse_args(["--reuse-points", "--tread-reference-width-cm", "90"])
    assert args.tread_reference_width_cm == 90
    import backend.app.services.stair_overlay.renderer_v2 as renderer
    assert "build_warped_geometry" not in renderer.__dict__
    assert "compose_stair_overlay" not in renderer.__dict__


def test_polished_alpha_has_no_hidden_rgb_black_corners_and_continuous_hinge() -> None:
    top, fold = _sources()
    faces = render_product_instance(top, fold, _geometry(), 65 / 90, (130, 100))
    for face in (faces.top, faces.fold, faces.product_instance):
        assert np.all(face[:, :, :3][face[:, :, 3] == 0] <= 1)
        for y, x in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
            assert not (face[y, x, 3] >= 254 and np.all(face[y, x, :3] <= 1))
    hinge = np.rint(faces.fold_quad[:2]).astype(int)
    samples = np.linspace(hinge[0], hinge[1], 20).round().astype(int)
    samples[:, 0] = np.clip(samples[:, 0], 0, 129)
    samples[:, 1] = np.clip(samples[:, 1], 0, 99)
    assert np.count_nonzero(faces.product_instance[samples[:, 1], samples[:, 0], 3] > 0) >= 18
    assert faces.product_instance.shape[:2] == (100, 130)
