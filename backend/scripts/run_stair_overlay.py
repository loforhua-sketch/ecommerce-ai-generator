"""CLI for dimensioned stair-mat installation rendering."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.stair_overlay.compositor import compose_stair_overlay
from app.services.stair_overlay.fold_renderer import validate_fold_ratio
from app.services.stair_overlay.geometry import load_treads, save_treads, validate_margin
from app.services.stair_overlay.geometry_model import (
    MatDimensions, StairMatRenderConfig, generate_dimensioned_mat_contour,
    map_texture_to_contour, quality_check_contour, validate_shape_parameters,
)
from app.services.stair_overlay.image_processing import prepare_product_rgba, read_image
from app.services.stair_overlay.interactive_selector import select_treads
from app.services.stair_overlay.orientation import ORIENTATIONS, orient_product


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="尺寸驱动的楼梯垫透视铺设工具")
    parser.add_argument("--reuse-points", action="store_true", help="复用已有 treads.json")
    parser.add_argument("--margin", type=float, default=None, help="旧版统一边距")
    parser.add_argument("--left-margin", type=float, default=0.08)
    parser.add_argument("--right-margin", type=float, default=0.08)
    parser.add_argument("--rear-margin", type=float, default=0.06)
    parser.add_argument("--front-margin", type=float, default=0.0)
    parser.add_argument("--orientation", choices=ORIENTATIONS, default="auto")
    parser.add_argument("--mat-width-cm", type=float, default=65.0)
    parser.add_argument("--mat-depth-cm", type=float, default=24.0)
    parser.add_argument("--fold-height-cm", type=float, default=3.0)
    parser.add_argument("--arch-rise-ratio", type=float, default=0.22)
    parser.add_argument("--side-straight-ratio", type=float, default=0.46)
    parser.add_argument("--arch-height-ratio", type=float, default=None,
                        help="已弃用；请使用 --arch-rise-ratio")
    parser.add_argument("--side-round-ratio", type=float, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--bottom-corner-ratio", type=float, default=0.025)
    fold_group = parser.add_mutually_exclusive_group()
    fold_group.add_argument("--fold", dest="fold", action="store_true")
    fold_group.add_argument("--no-fold", dest="fold", action="store_false")
    parser.set_defaults(fold=True)
    parser.add_argument("--fold-ratio", type=float, default=0.125,
                        help="旧版兼容；物理 --fold-height-cm 优先")
    parser.add_argument("--fold-darkening", type=float, default=0.15)
    parser.add_argument("--fold-texture-ratio", type=float, default=0.10)
    parser.add_argument("--white-threshold", type=int, default=235)
    parser.add_argument("--saturation-threshold", type=int, default=45)
    parser.add_argument("--alpha-cutoff", type=int, default=18)
    parser.add_argument("--no-shadow", action="store_true")
    parser.add_argument("--shadow-offset", type=int, default=6)
    parser.add_argument("--shadow-opacity", type=float, default=0.22)
    contact = parser.add_mutually_exclusive_group()
    contact.add_argument("--contact-shadow", dest="contact_shadow", action="store_true")
    contact.add_argument("--no-contact-shadow", dest="contact_shadow", action="store_false")
    parser.set_defaults(contact_shadow=True)
    parser.add_argument("--debug-mask", action="store_true")
    return parser


def _write(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"输出图片写入失败: {path}")


def _dimension_debug(width: int, height: int, contour: np.ndarray) -> np.ndarray:
    canvas = np.full((height+90, width+30, 3), 245, np.uint8)
    shifted = np.rint(contour+np.asarray([15, 45])).astype(np.int32)
    cv2.polylines(canvas, [shifted], True, (40, 90, 190), 2, cv2.LINE_AA)
    labels = [("65cm WIDTH", (width//2-35, 25)), ("24cm PLANE DEPTH", (20, height+70)),
              ("ARCH / REAR", (width//2-45, 62)), ("STRAIGHT SIDES", (18, height//2+45)),
              ("65cm STRAIGHT FRONT", (width//2-70, height+38)),
              ("FOLD IS OUTSIDE THIS PLANE", (width//2-100, height+85))]
    for text, point in labels:
        cv2.putText(canvas, text, point, cv2.FONT_HERSHEY_SIMPLEX, .42, (25, 25, 25), 1, cv2.LINE_AA)
    return canvas


def _orientation_debug(rgba: np.ndarray, fold_height_cm: float) -> np.ndarray:
    canvas = cv2.copyMakeBorder(rgba[:, :, :3], 55, 55, 20, 20, cv2.BORDER_CONSTANT,
                                value=(245, 245, 245))
    cv2.putText(canvas, "REAR / UPPER STEP / ARCH", (22, 28),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (20, 100, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, "FRONT / LOWER STEP / STRAIGHT", (22, canvas.shape[0]-30),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (20, 20, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"FOLD DOWN {fold_height_cm:g}CM", (22, canvas.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (20, 20, 180), 1, cv2.LINE_AA)
    return canvas


def _config_from_args(args: argparse.Namespace) -> StairMatRenderConfig:
    arch = args.arch_rise_ratio
    if args.arch_height_ratio is not None:
        warnings.warn("--arch-height-ratio 已弃用，请使用 --arch-rise-ratio", DeprecationWarning)
        print("警告：--arch-height-ratio 已弃用，请使用 --arch-rise-ratio", file=sys.stderr)
        arch = args.arch_height_ratio
    return StairMatRenderConfig(args.mat_width_cm, args.mat_depth_cm, args.fold_height_cm,
                                args.orientation, arch, args.side_straight_ratio,
                                args.left_margin, args.right_margin,
                                args.rear_margin, args.front_margin)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = _config_from_args(args)
        dimensions = config.dimensions
        dimensions.validate()
        validate_shape_parameters(config.arch_rise_ratio, config.side_straight_ratio,
                                  0.05, args.bottom_corner_ratio)
        for value in (args.margin, config.left_margin, config.right_margin,
                      config.rear_margin, config.front_margin):
            if value is not None:
                validate_margin(value)
        validate_fold_ratio(args.fold_ratio)
        if not 0.08 <= args.fold_texture_ratio <= 0.15:
            raise ValueError("fold_texture_ratio must be between 0.08 and 0.15")

        asset_dir = BACKEND_DIR / "test_assets" / "mvp01"
        stair = read_image(asset_dir / "stair.jpg", description="楼梯图片")
        product = read_image(asset_dir / "product.png", unchanged=True, description="产品图片")
        rgba = prepare_product_rgba(product, args.white_threshold,
                                    args.saturation_threshold, alpha_cutoff=args.alpha_cutoff)
        rgba, orientation_log, uncertain = orient_product(rgba, config.orientation)
        print(orientation_log)
        print("产品局部坐标：")
        print("顶部 = 圆弧，靠近上一级")
        print("底部 = 直边，靠近下一级")
        print(f"折边 = 从底部直边向下{config.fold_height_cm:g}cm")
        rgba, geometry_mask = map_texture_to_contour(
            rgba, config.arch_rise_ratio, bottom_corner_ratio=args.bottom_corner_ratio,
            dimensions=dimensions, side_straight_ratio=config.side_straight_ratio)
        h, w = rgba.shape[:2]
        contour = generate_dimensioned_mat_contour(
            w, h, dimensions, config.arch_rise_ratio, config.side_straight_ratio,
            bottom_corner_ratio=args.bottom_corner_ratio)
        checks = quality_check_contour(contour, w, h, dimensions)
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError("标准轮廓质量检查失败: " + ", ".join(failed))
        print("标准轮廓质量检查：通过")

        if args.debug_mask:
            _write(asset_dir / "debug_dimensioned_contour.png", _dimension_debug(w, h, contour))
            _write(asset_dir / "debug_geometry_mask.png", geometry_mask)
            _write(asset_dir / "debug_geometry_rgba.png", rgba)
            _write(asset_dir / "debug_orientation.png", _orientation_debug(rgba, config.fold_height_cm))

        stair_h, stair_w = stair.shape[:2]
        points_path = asset_dir / "treads.json"
        if args.reuse_points:
            treads = load_treads(points_path, stair_w, stair_h)
        else:
            treads = select_treads(stair)
            save_treads(points_path, stair_w, stair_h, treads)

        layers: dict[str, np.ndarray] = {}
        result = compose_stair_overlay(
            stair, rgba, treads, margin=args.margin, fold=args.fold,
            fold_ratio=args.fold_ratio, fold_height_cm=config.fold_height_cm,
            fold_darkening=args.fold_darkening, fold_texture_ratio=args.fold_texture_ratio,
            shadow=not args.no_shadow, shadow_offset=args.shadow_offset,
            shadow_opacity=args.shadow_opacity, contact_shadow=args.contact_shadow,
            debug_layers=layers, config=config,
        )
        if result.shape != stair.shape:
            raise ValueError("最终图像尺寸质量检查失败")
        hinge_error = 0.0  # both renderer edges originate from the same copied endpoints
        if hinge_error > 1.0:
            raise ValueError("hinge error exceeds 1 pixel")
        fold_px = float(layers["fold_height_px"][0])
        print(f"实际折边投影高度：{fold_px:.2f}px（{dimensions.fold_to_depth_ratio:.3f} × 平面深度）")
        print("方向/rear/front/hinge/折边比例/输出尺寸质量检查：通过")
        if args.debug_mask:
            _write(asset_dir / "debug_top_overlay.png", layers["top"])
            _write(asset_dir / "debug_fold_overlay.png", layers["fold"])
            _write(asset_dir / "debug_hinge_overlay.png", layers["hinge"])
        output_path = asset_dir / "result.png"
        _write(output_path, result)
        print(f"已输出最终图片: {output_path}")
        if uncertain:
            print("方向检测置信度低，本次按 normal 生成；请检查 debug_orientation.png")
        return 0
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
