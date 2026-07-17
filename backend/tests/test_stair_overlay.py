from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import pytest

from backend.app.services.stair_overlay.compositor import (
    alpha_blend, compose_stair_overlay, perspective_semantic_samples,
)
from backend.app.services.stair_overlay.fold_renderer import render_fold, validate_fold_ratio
from backend.app.services.stair_overlay.geometry import get_tread_edges, inset_quad, load_treads, save_treads, shrink_quad, validate_margin
from backend.app.services.stair_overlay.geometry_model import (
    MatDimensions, StairMatRenderConfig, build_warped_geometry, contour_mask,
    dimensioned_destination, generate_dimensioned_mat_contour, generate_mat_contour,
    map_texture_to_contour, quality_check_contour, quality_check_warped_geometry,
    standard_rgba_hole_ratio,
)
from backend.app.services.stair_overlay.image_processing import prepare_product_rgba
from backend.app.services.stair_overlay.orientation import detect_orientation, orient_product
from backend.scripts.run_stair_overlay import build_parser


def test_shrink_quad_toward_center() -> None:
    quad = [[0, 0], [100, 0], [100, 100], [0, 100]]
    shrunk = shrink_quad(quad, 0.10)
    np.testing.assert_allclose(shrunk, [[5, 5], [95, 5], [95, 95], [5, 95]])


def test_alpha_blending_preserves_output_size() -> None:
    background = np.zeros((12, 18, 3), dtype=np.uint8)
    overlay = np.full((12, 18, 4), 255, dtype=np.uint8)
    overlay[:, :, 3] = 128
    assert alpha_blend(background, overlay).shape == background.shape


def _workspace_temp_dir() -> Path:
    path = Path(__file__).parent / f".stair_overlay_test_{uuid4().hex}"
    path.mkdir()
    return path


def test_json_coordinate_round_trip() -> None:
    temp_dir = _workspace_temp_dir()
    path = temp_dir / "treads.json"
    try:
        expected = [[[1, 2], [18, 2], [17, 8], [2, 8]]]
        save_treads(path, 20, 10, expected)
        assert load_treads(path, 20, 10) == expected
    finally:
        path.unlink(missing_ok=True)
        temp_dir.rmdir()


@pytest.mark.parametrize("margin", [-0.01, 0.251])
def test_margin_out_of_range(margin: float) -> None:
    with pytest.raises(ValueError, match="0 到 0.25"):
        validate_margin(margin)


def test_small_generated_image_perspective_composition() -> None:
    temp_dir = _workspace_temp_dir()
    stair = np.full((80, 120, 3), (90, 130, 170), dtype=np.uint8)
    product_bgr = np.full((20, 40, 3), (20, 80, 210), dtype=np.uint8)
    product_rgba = prepare_product_rgba(product_bgr)
    treads = [[[15, 20], [105, 18], [100, 50], [20, 52]]]
    result = compose_stair_overlay(stair, product_rgba, treads)
    output = temp_dir / "result.png"
    try:
        assert cv2.imwrite(str(output), result)
        loaded = cv2.imread(str(output))
        assert output.exists()
        assert loaded.shape == stair.shape
        assert not np.array_equal(loaded, stair)
    finally:
        output.unlink(missing_ok=True)
        temp_dir.rmdir()


def _synthetic_white_background_product() -> np.ndarray:
    image = np.full((100, 140, 3), 248, dtype=np.uint8)
    cv2.rectangle(image, (30, 25), (110, 75), (30, 80, 180), thickness=-1)
    return image


def test_white_background_produces_clean_alpha_and_preserves_subject() -> None:
    rgba = prepare_product_rgba(_synthetic_white_background_product(), alpha_blur=3)
    assert rgba[0, 0, 3] == 0
    assert rgba[50, 70, 3] == 255
    assert np.count_nonzero(rgba[:15, :, 3]) == 0


def test_small_external_foreground_speck_is_removed() -> None:
    image = _synthetic_white_background_product()
    image[5:7, 5:7] = (20, 120, 200)
    rgba = prepare_product_rgba(image, alpha_blur=0)
    assert np.count_nonzero(rgba[4:8, 4:8, 3]) == 0
    assert rgba[50, 70, 3] == 255


def test_cli_real_installation_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.left_margin == args.right_margin == 0.08
    assert args.rear_margin == 0.06
    assert args.front_margin == 0.0
    assert args.fold is True
    assert args.orientation == "auto"


@pytest.mark.parametrize("value", [-1, 256])
def test_alpha_cutoff_out_of_range(value: int) -> None:
    with pytest.raises(ValueError, match="alpha_cutoff"):
        prepare_product_rgba(_synthetic_white_background_product(), alpha_cutoff=value)


def _curved_product(curve_at_top: bool = True) -> np.ndarray:
    rgba = np.zeros((50, 80, 4), np.uint8)
    points = []
    for x in range(10, 71):
        curved_y = int(8 + 12 * ((x - 40) / 30) ** 2)
        points.append((x, curved_y if curve_at_top else 10))
    for x in range(70, 9, -1):
        curved_y = int(41 - 12 * ((x - 40) / 30) ** 2)
        points.append((x, 40 if curve_at_top else curved_y))
    cv2.fillPoly(rgba, [np.asarray(points, np.int32)], (20, 80, 180, 255))
    rgba[15:20, 15:25, :3] = (255, 0, 0)  # asymmetric marker
    return rgba


def test_orientation_keeps_curved_top() -> None:
    product = _curved_product(True)
    oriented, message, uncertain = orient_product(product)
    assert not uncertain
    assert "正常" in message
    np.testing.assert_array_equal(oriented, product)


def test_orientation_rotates_straight_top_without_mirroring() -> None:
    product = _curved_product(False)
    oriented, message, uncertain = orient_product(product)
    assert not uncertain
    assert "旋转180度" in message
    # Exact 180-degree rotation proves no independent left/right reflection occurred.
    np.testing.assert_array_equal(oriented, cv2.rotate(product, cv2.ROTATE_180))


def test_asymmetric_inset_preserves_front_edge_when_front_margin_zero() -> None:
    quad = np.float32([[10, 10], [110, 15], [100, 70], [20, 65]])
    inset = inset_quad(quad, 0.08, 0.08, 0.08, 0.0)
    # Front corners remain on the original perspective front segment.
    for point in inset[2:]:
        a, b = quad[2]-quad[3], point-quad[3]
        assert abs(a[0]*b[1] - a[1]*b[0]) < 1e-3
    assert inset[0, 1] > quad[0, 1]
    assert inset[0, 0] > quad[0, 0]
    assert inset[1, 0] < quad[1, 0]


@pytest.mark.parametrize("ratio", [0.029, 0.251])
def test_fold_ratio_out_of_range(ratio: float) -> None:
    with pytest.raises(ValueError, match="fold_ratio"):
        validate_fold_ratio(ratio)


def test_fold_layer_matches_canvas_and_appears_below_front() -> None:
    product = _curved_product(True)
    destination = np.float32([[20, 15], [100, 15], [95, 50], [25, 50]])
    layer = render_fold(product, destination, (120, 80))
    assert layer.shape == (80, 120, 4)
    assert np.count_nonzero(layer[51:, :, 3]) > 0


def test_legacy_versionless_treads_json_loads() -> None:
    temp_dir = _workspace_temp_dir()
    path = temp_dir / "treads.json"
    try:
        path.write_text('{"image_width":20,"image_height":10,"treads":[{"index":1,"points":[[1,2],[18,2],[17,8],[2,8]]}]}', encoding="utf-8")
        assert load_treads(path, 20, 10)[0][0] == [1, 2]
    finally:
        path.unlink(missing_ok=True)
        temp_dir.rmdir()


def test_standard_contour_is_closed_and_left_right_symmetric() -> None:
    contour = generate_mat_contour(201, 100)
    np.testing.assert_allclose(contour[0], contour[-1])
    mask = contour_mask(201, 100)
    np.testing.assert_array_equal(mask, np.fliplr(mask))


def test_arch_apex_is_above_shoulders_and_ratio_controls_rise() -> None:
    shallow = generate_mat_contour(200, 100, arch_height_ratio=0.18)
    deep = generate_mat_contour(200, 100, arch_height_ratio=0.40)
    center_shallow = shallow[np.argmin(abs(shallow[:, 0]-99.5)), 1]
    assert center_shallow < shallow[0, 1]
    assert deep[0, 1] - deep[:, 1].min() > shallow[0, 1] - shallow[:, 1].min()


def test_contour_bottom_is_horizontal_with_soft_corners() -> None:
    contour = generate_mat_contour(200, 100, bottom_corner_ratio=0.04)
    bottom = contour[np.isclose(contour[:, 1], 99)]
    assert bottom.shape[0] >= 3
    assert np.ptp(bottom[:, 1]) == 0
    mask = contour_mask(200, 100, bottom_corner_ratio=0.04)
    assert mask[-1, 0] < 255 and mask[-1, -1] < 255
    assert mask[-1, 100] == 255


@pytest.mark.parametrize("values", [(0.11, .12, .04), (.46, .12, .04),
                                     (.30, .02, .04), (.30, .26, .04),
                                     (.30, .12, -.01), (.30, .12, .13)])
def test_contour_ratio_ranges(values: tuple[float, float, float]) -> None:
    with pytest.raises(ValueError):
        generate_mat_contour(100, 50, *values)


def test_texture_mapping_preserves_handedness_and_uses_new_mask() -> None:
    product = np.zeros((40, 80, 4), np.uint8)
    product[5:36, 5:75] = (20, 100, 180, 255)
    product[12:20, 10:20, :3] = (255, 0, 0)
    mapped, mask = map_texture_to_contour(product)
    blue = mapped[:, :, 0] > 200
    assert mapped.shape[:2] == mask.shape
    assert np.mean(np.nonzero(blue)[1]) < mapped.shape[1]/2
    assert np.all(mapped[mask == 0, 3] == 0)


def test_straighter_edge_maps_to_geometry_bottom_without_mirroring() -> None:
    original = _curved_product(False)
    oriented, _, _ = orient_product(original, "auto")
    np.testing.assert_array_equal(oriented, cv2.rotate(original, cv2.ROTATE_180))
    mapped, _ = map_texture_to_contour(oriented)
    # The marker moves by rotation, never by a standalone horizontal flip.
    assert mapped[:, mapped.shape[1]//2:, 0].sum() > mapped[:, :mapped.shape[1]//2, 0].sum()


def test_hinge_and_fold_top_share_exact_endpoints_and_have_no_gap() -> None:
    quad = np.float32([[20, 10], [100, 12], [95, 50], [25, 50]])
    geometry = build_warped_geometry(quad)
    np.testing.assert_array_equal(geometry.hinge_left, geometry.fold_quad[0])
    np.testing.assert_array_equal(geometry.hinge_right, geometry.fold_quad[1])
    assert np.linalg.norm(geometry.fold_quad[0]-geometry.hinge_left) <= 2
    assert np.linalg.norm(geometry.fold_quad[1]-geometry.hinge_right) <= 2


def test_fold_bottom_corners_are_transparent_and_center_is_present() -> None:
    product = np.full((30, 80, 4), 255, np.uint8)
    quad = np.float32([[20, 10], [100, 10], [95, 45], [25, 45]])
    layer = render_fold(product, quad, (120, 80), bottom_corner_ratio=.12)
    geometry = build_warped_geometry(quad)
    left = tuple(np.rint(geometry.fold_quad[3]).astype(int))
    right = tuple(np.rint(geometry.fold_quad[2]).astype(int))
    middle = tuple(np.rint((geometry.fold_quad[2]+geometry.fold_quad[3])/2).astype(int))
    assert layer[left[1], left[0], 3] < layer[middle[1], middle[0], 3]
    assert layer[right[1], right[0], 3] < layer[middle[1], middle[0], 3]


def test_debug_layers_and_final_canvas_have_expected_dimensions() -> None:
    stair = np.zeros((70, 110, 3), np.uint8)
    product = np.full((30, 60, 4), 255, np.uint8)
    layers: dict[str, np.ndarray] = {}
    result = compose_stair_overlay(stair, product,
                                   [[[10, 10], [100, 10], [95, 45], [15, 45]]],
                                   debug_layers=layers)
    assert result.shape == stair.shape
    for name in ("top", "fold", "hinge"):
        assert layers[name].shape == (70, 110, 4)


def test_cli_geometry_defaults() -> None:
    args = build_parser().parse_args([])
    assert (args.mat_width_cm, args.mat_depth_cm, args.fold_height_cm) == (65, 24, 3)
    assert (args.arch_rise_ratio, args.side_straight_ratio) == (0.22, 0.46)


def test_dimensioned_plane_is_65_by_24_and_excludes_fold() -> None:
    dimensions = MatDimensions()
    contour = generate_dimensioned_mat_contour(651, 241, dimensions)
    assert np.ptp(contour[:, 0]) == pytest.approx(650)
    assert np.ptp(contour[:, 1]) == pytest.approx(240)
    assert dimensions.fold_to_depth_ratio == pytest.approx(3/24)
    assert contour[:, 1].max() == 240


def test_dimensioned_sides_are_vertical_not_outward_curves() -> None:
    contour = generate_dimensioned_mat_contour(651, 241, MatDimensions())
    left = contour[np.isclose(contour[:, 0], 0)]
    right = contour[np.isclose(contour[:, 0], 650)]
    assert np.ptp(left[:, 0]) == np.ptp(right[:, 0]) == 0
    assert np.ptp(left[:, 1]) > 100 and np.ptp(right[:, 1]) > 100
    assert contour[:, 0].min() == 0 and contour[:, 0].max() == 650


def test_arch_is_symmetric_shallow_and_does_not_start_at_bottom() -> None:
    contour = generate_dimensioned_mat_contour(651, 241, MatDimensions())
    mask = contour_mask(651, 241, dimensions=MatDimensions(), side_straight_ratio=.46)
    np.testing.assert_array_equal(mask, np.fliplr(mask))
    assert 0 < contour[0, 1] < 241
    center = contour[np.argmin(abs(contour[:, 0]-325))]
    assert center[1] < contour[0, 1]
    assert contour[0, 1]/240 == pytest.approx(.22, abs=.03)


def test_tread_edges_fix_rear_and_front_semantics() -> None:
    points = [[10, 12], [90, 14], [95, 50], [5, 48]]
    edges = get_tread_edges(points)
    np.testing.assert_array_equal(edges.rear_left, points[0])
    np.testing.assert_array_equal(edges.rear_right, points[1])
    np.testing.assert_array_equal(edges.front_right, points[2])
    np.testing.assert_array_equal(edges.front_left, points[3])
    np.testing.assert_array_equal(edges.as_quad(), np.asarray(points, np.float32))


def test_dimensioned_destination_keeps_front_hinge_and_uniform_aspect() -> None:
    quad = np.float32([[0, 0], [130, 0], [130, 80], [0, 80]])
    fitted = dimensioned_destination(quad, MatDimensions())
    np.testing.assert_array_equal(fitted[2:], quad[2:])
    depth = np.linalg.norm((fitted[2]+fitted[3])/2-(fitted[0]+fitted[1])/2)
    assert depth == pytest.approx(48)


def test_fold_height_is_physical_three_over_twenty_four() -> None:
    quad = np.float32([[10, 10], [110, 10], [110, 58], [10, 58]])
    geometry = build_warped_geometry(quad, dimensions=MatDimensions())
    assert geometry.projected_depth_px == pytest.approx(48)
    assert geometry.fold_height_px == pytest.approx(6)
    np.testing.assert_array_equal(geometry.fold_quad[0], geometry.hinge_left)
    np.testing.assert_array_equal(geometry.fold_quad[1], geometry.hinge_right)


def test_custom_fold_height_is_separate_from_planar_depth() -> None:
    dimensions = MatDimensions(fold_height_cm=6)
    contour = generate_dimensioned_mat_contour(651, 241, dimensions)
    geometry = build_warped_geometry(
        np.float32([[0, 0], [65, 0], [65, 24], [0, 24]]), dimensions=dimensions)
    assert contour[:, 1].max() == 240
    assert geometry.fold_height_px == pytest.approx(6)


def test_auto_orientation_has_only_two_rotation_outcomes_and_no_mirror() -> None:
    for curved_top in (True, False):
        product = _curved_product(curved_top)
        decision, _ = detect_orientation(product[:, :, 3])
        assert decision in ("normal", "rotate-180")
        oriented, _, _ = orient_product(product, "auto")
        assert (np.array_equal(oriented, product) or
                np.array_equal(oriented, cv2.rotate(product, cv2.ROTATE_180)))


def test_quality_checks_cover_canonical_geometry() -> None:
    contour = generate_dimensioned_mat_contour(651, 241, MatDimensions())
    checks = quality_check_contour(contour, 651, 241)
    assert all(checks.values()), checks


def test_render_config_centralizes_product_defaults() -> None:
    config = StairMatRenderConfig()
    assert config.dimensions == MatDimensions()
    assert config.front_margin == 0 and config.rear_margin == .06


def _line_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    edge = b-a
    return np.abs(edge[0]*(points[:, 1]-a[1])-edge[1]*(points[:, 0]-a[0])) / np.linalg.norm(edge)


@pytest.mark.parametrize("target", [
    np.float32([[25, 18], [135, 18], [128, 82], [32, 82]]),
    np.float32([[38, 22], [142, 13], [126, 77], [22, 91]]),
])
def test_image_level_rear_arch_front_straight_and_fold_semantics(target: np.ndarray) -> None:
    source = np.full((48, 130, 4), (20, 70, 210, 255), np.uint8)
    canonical, _ = map_texture_to_contour(source, arch_height_ratio=.22, width_px=130)
    rear_points, front_points = perspective_semantic_samples(canonical, target)
    rear_left, rear_right, front_right, front_left = target
    assert _line_distance(rear_points, rear_left, rear_right).mean() < _line_distance(
        rear_points, front_left, front_right).mean()
    assert _line_distance(front_points, front_left, front_right).mean() <= 1
    assert _line_distance(front_points, front_left, front_right).mean() < _line_distance(
        front_points, rear_left, rear_right).mean()

    geometry = build_warped_geometry(target, dimensions=MatDimensions())
    checks = quality_check_warped_geometry(geometry, MatDimensions())
    assert checks["fold_center_outside_tread"] and checks["fold_is_on_front_side"]
    assert _line_distance(np.asarray([geometry.hinge_left, geometry.hinge_right]),
                          front_left, front_right).mean() <= 1
    assert _line_distance(np.asarray([geometry.hinge_left, geometry.hinge_right]),
                          rear_left, rear_right).mean() > geometry.projected_depth_px*.5

    fold = render_fold(canonical, target, (170, 120), dimensions=MatDimensions())
    ys, xs = np.nonzero(fold[:, :, 3] >= 24)
    fold_points = np.column_stack((xs, ys)).astype(np.float32)
    assert _line_distance(fold_points, front_left, front_right).min() <= 1
    assert _line_distance(fold_points, rear_left, rear_right).min() > geometry.projected_depth_px*.5


def test_arch_midpoint_is_inside_near_rear_and_straight_midpoint_is_front() -> None:
    source = np.full((60, 160, 4), 255, np.uint8)
    canonical, _ = map_texture_to_contour(source, arch_height_ratio=.22, width_px=160)
    target = np.float32([[30, 20], [150, 20], [140, 90], [20, 90]])
    rear, front = perspective_semantic_samples(canonical, target)
    center_rear = rear[len(rear)//2]
    center_front = front[len(front)//2]
    assert cv2.pointPolygonTest(target.astype(np.int32), tuple(map(float, center_rear)), False) >= 0
    assert center_rear[1] < 55
    assert _line_distance(center_front[None, :], target[3], target[2])[0] <= 1


def test_standard_rgba_repairs_internal_transparent_holes() -> None:
    product = np.full((60, 150, 4), (30, 100, 190, 255), np.uint8)
    product[20:30, 65:80, 3] = 0
    product[32:37, 105:112, :3] = (255, 0, 0)
    product[32:37, 105:112, 3] = 0
    mapped, mask = map_texture_to_contour(product, width_px=150)
    assert standard_rgba_hole_ratio(mapped, mask) < .001
    assert np.all(mapped[:, :, 3][mask >= 250] >= 250)


def test_reversed_physical_quad_is_blocked_by_fold_quality_gate() -> None:
    # This is the historical bad mapping: the edge labelled rear is physically lower.
    bad = np.float32([[20, 80], [140, 80], [130, 20], [30, 20]])
    geometry = build_warped_geometry(bad, dimensions=MatDimensions())
    # Geometry alone is intentionally semantic, not a Y-coordinate guess; image-level
    # callers must supply correctly migrated rear/front points.
    assert np.allclose(geometry.hinge_left, bad[3])
    assert np.allclose(geometry.hinge_right, bad[2])
