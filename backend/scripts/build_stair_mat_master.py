"""Build the three fixed 65 x 24 x 3 cm stair-mat master assets."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent

# Keep the documented command runnable when the system Python lacks the
# project's image dependencies but the checked project environment has them.
if importlib.util.find_spec("cv2") is None:
    project_python = BACKEND_DIR / ".venv" / "Scripts" / "python.exe"
    if project_python.is_file() and Path(sys.executable).resolve() != project_python.resolve():
        raise SystemExit(subprocess.call(
            [str(project_python), str(Path(__file__).resolve()), *sys.argv[1:]]
        ))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.product_master import (  # noqa: E402
    DEFAULT_MASTER_DIR,
    generate_product_master,
    save_product_master,
)
from backend.app.services.stair_overlay.image_processing import read_image  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成固定 65×24×3 cm 楼梯垫母版")
    parser.add_argument("--input", type=Path, required=True, help="方向与轮廓已正确的产品图片")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = read_image(args.input, unchanged=True, description="product image")
    master = generate_product_master(source)
    outputs = save_product_master(master, DEFAULT_MASTER_DIR)
    for name, output in outputs.items():
        print(f"{name}={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
