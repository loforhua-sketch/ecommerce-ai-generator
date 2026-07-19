"""Formal MVP-04 stair overlay entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.stair_overlay.geometry_v2 import load_tread_geometry  # noqa: E402
from app.services.stair_overlay.renderer_v2 import compose_overlay_v2  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    asset_dir = BACKEND_DIR / "test_assets" / "mvp01"
    parser = argparse.ArgumentParser(description="MVP-04 双平面楼梯垫投影")
    parser.add_argument("--reuse-points", action="store_true",
                        help="读取六点treads.json；v2正式渲染必须使用")
    parser.add_argument("--tread-reference-width-cm", type=float)
    parser.add_argument("--input-image", type=Path, default=asset_dir / "stair.jpg")
    parser.add_argument("--treads-json", type=Path, default=asset_dir / "treads.json")
    parser.add_argument("--master-dir", type=Path, default=asset_dir / "master")
    parser.add_argument("--output-dir", type=Path, default=asset_dir)
    parser.add_argument("--debug-mask", action="store_true")
    return parser


def _read(path: Path, flags: int) -> object:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return image


def _write(path: Path, image: object) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"无法写入图像: {path}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not args.reuse_points:
            raise ValueError("v2不在渲染过程中猜测几何；请使用--reuse-points或先运行六点标注工具")
        stair = _read(args.input_image, cv2.IMREAD_COLOR)
        top = _read(args.master_dir / "master_top.png", cv2.IMREAD_UNCHANGED)
        fold = _read(args.master_dir / "master_fold.png", cv2.IMREAD_UNCHANGED)
        if top.ndim != 3 or top.shape[2] != 4 or fold.ndim != 3 or fold.shape[2] != 4:
            raise ValueError("master_top.png和master_fold.png必须为BGRA")
        if top.shape[1] != fold.shape[1]:
            raise ValueError("top/fold源纹理宽度必须一致")
        document = load_tread_geometry(args.treads_json, (stair.shape[1], stair.shape[0]))
        if document.needs_fold_annotation:
            raise ValueError("当前treads.json仅有4点，需要补标每级fold_bottom_left和fold_bottom_right")
        reference = args.tread_reference_width_cm or document.tread_reference_width_cm
        if reference is None:
            raise ValueError("未提供tread_reference_width_cm；禁止按每级踏面铺满")
        debug: dict[str, object] = {}
        result = compose_overlay_v2(stair, top, fold, document.treads, reference,
                                    document.depth_reference_cm, debug)
        if result.shape != stair.shape:
            raise ValueError("result_v2输出尺寸与输入楼梯图不一致")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.debug_mask:
            _write(args.output_dir / "debug_geometry_v2.png", debug["geometry"])
            _write(args.output_dir / "debug_top_warp_v2.png", debug["top_warp"])
            _write(args.output_dir / "debug_fold_warp_v2.png", debug["fold_warp"])
            _write(args.output_dir / "debug_product_instance_v2.png", debug["product_instance"])
            _write(args.output_dir / "debug_final_mask_v2.png", debug["final_mask"])
            _write(args.output_dir / "debug_top_alpha_v2.png", debug["top_alpha"])
            _write(args.output_dir / "debug_fold_alpha_v2.png", debug["fold_alpha"])
            _write(args.output_dir / "debug_composite_alpha_v2.png", debug["composite_alpha"])
            _write(args.output_dir / "debug_edge_check_v2.png", debug["edge_check"])
        _write(args.output_dir / "result_v2.png", result)
        print(f"已生成: {args.output_dir / 'result_v2.png'}")
        return 0
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
