"""Generate the fixed canonical product master used by later render stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .stair_overlay.image_processing import prepare_product_rgba


MASTER_WIDTH_PX = 1300
MASTER_FORMAT = "png"
MASTER_WIDTH_CM = 65.0
MASTER_DEPTH_CM = 24.0
MASTER_FOLD_CM = 3.0
DEFAULT_MASTER_DIR = Path(__file__).resolve().parents[2] / "test_assets" / "mvp01" / "master"


@dataclass(frozen=True)
class ProductMaster:
    """Canonical BGRA master and the decisions used to produce it."""

    rgba: np.ndarray
    width_cm: float
    depth_cm: float
    orientation: str = "normal"
    orientation_uncertain: bool = False

    @property
    def width_px(self) -> int:
        return int(self.rgba.shape[1])

    @property
    def height_px(self) -> int:
        return int(self.rgba.shape[0])

    def png_bytes(self) -> bytes:
        return encode_png(self.rgba)

    @property
    def mask(self) -> np.ndarray:
        return self.rgba[:, :, 3].copy()

    @property
    def fold_rgba(self) -> np.ndarray:
        """Return the straight-front 65 x 3 cm texture strip."""
        fold_height = max(1, int(round(self.width_px * 3.0 / self.width_cm)))
        source_height = max(1, int(round(self.height_px * 3.0 / self.depth_cm)))
        # The hinge is the top row of the fold and the bottom row of the top
        # plane. Sampling inward from that shared edge preserves continuity.
        source = self.rgba[-source_height:, :, :][::-1]
        if source.shape[0] == fold_height:
            return source.copy()
        return cv2.resize(source, (self.width_px, fold_height), interpolation=cv2.INTER_AREA)


def encode_png(image: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError("无法编码产品母版 PNG")
    return encoded.tobytes()


def decode_product_image(content: bytes) -> np.ndarray:
    """Decode an uploaded image without relying on a temporary filesystem path."""
    if not content:
        raise ValueError("产品图片不能为空")
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("无法读取产品图片")
    return image


def save_product_master(
    master: ProductMaster,
    output_dir: Path = DEFAULT_MASTER_DIR,
) -> dict[str, Path]:
    """Persist the three fixed MVP-03A master assets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    assets = {
        "top": (output_dir / "master_top.png", master.rgba),
        "fold": (output_dir / "master_fold.png", master.fold_rgba),
        "mask": (output_dir / "master_mask.png", master.mask),
        "debug": (output_dir / "debug_master_shape.png", build_debug_master_shape(master)),
        "debug_fold": (output_dir / "debug_master_fold.png", build_debug_master_fold(master)),
    }
    paths: dict[str, Path] = {}
    for name, (output_path, image) in assets.items():
        temporary_path = output_path.with_suffix(".png.tmp")
        temporary_path.write_bytes(encode_png(image))
        temporary_path.replace(output_path)
        paths[name] = output_path
    return paths


def build_debug_master_shape(master: ProductMaster) -> np.ndarray:
    """Visualize the fixed top, downward fold and their pixel-shared hinge."""
    top, fold = master.rgba, master.fold_rgba
    height = top.shape[0] + fold.shape[0] - 1
    width = top.shape[1]
    debug = np.full((height, width, 3), 232, np.uint8)
    tile = max(8, width // 80)
    yy, xx = np.indices((height, width))
    debug[((xx // tile + yy // tile) % 2) == 0] = 248

    def blend(layer: np.ndarray, y: int) -> None:
        region = debug[y:y + layer.shape[0]]
        alpha = layer[:, :, 3:4].astype(np.float32) / 255.0
        region[:] = np.clip(layer[:, :, :3] * alpha + region * (1.0 - alpha), 0, 255)

    blend(top, 0)
    blend(fold, top.shape[0] - 1)
    edge = cv2.subtract(master.mask, cv2.erode(master.mask, np.ones((3, 3), np.uint8)))
    debug[:top.shape[0]][edge > 0] = (40, 210, 40)
    hinge_y = top.shape[0] - 1
    debug[hinge_y, :, :] = (0, 220, 255)
    cv2.putText(debug, "TOP 65x24cm", (20, 38), cv2.FONT_HERSHEY_SIMPLEX,
                .8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(debug, "SHARED EDGE / FOLD 65x3cm", (20, hinge_y - 12),
                cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 220, 255), 2, cv2.LINE_AA)
    return debug


def build_debug_master_fold(master: ProductMaster) -> np.ndarray:
    """Visualize the standalone downward fold and its shared top edge."""
    fold = master.fold_rgba
    pad = max(24, fold.shape[0] // 2)
    debug = np.full((fold.shape[0] + pad * 2, fold.shape[1], 3), 232, np.uint8)
    tile = max(8, fold.shape[1] // 80)
    yy, xx = np.indices(debug.shape[:2])
    debug[((xx // tile + yy // tile) % 2) == 0] = 248
    region = debug[pad:pad + fold.shape[0]]
    alpha = fold[:, :, 3:4].astype(np.float32) / 255.0
    region[:] = np.clip(fold[:, :, :3] * alpha + region * (1.0 - alpha), 0, 255)
    debug[pad, :, :] = (0, 220, 255)
    cv2.arrowedLine(debug, (fold.shape[1] // 2, pad + 8),
                    (fold.shape[1] // 2, pad + fold.shape[0] - 4),
                    (255, 255, 255), 3, cv2.LINE_AA, tipLength=.22)
    cv2.putText(debug, "SHARED EDGE", (20, pad - 8), cv2.FONT_HERSHEY_SIMPLEX,
                .65, (0, 180, 220), 2, cv2.LINE_AA)
    cv2.putText(debug, "FOLD DOWN 3cm", (20, pad + fold.shape[0] + 24),
                cv2.FONT_HERSHEY_SIMPLEX, .65, (40, 40, 40), 2, cv2.LINE_AA)
    return debug


def generate_product_master(
    product: np.ndarray,
    *,
    width_px: int = MASTER_WIDTH_PX,
) -> ProductMaster:
    """Create one deterministic, dimensioned master from a product photograph.

    The returned image is always BGRA. Its plane represents 65 x 24 cm with the
    curved edge at the rear/top and the straight fold edge at the front/bottom.
    """
    if not 260 <= width_px <= 5200:
        raise ValueError("width_px must be between 260 and 5200")

    prepared = prepare_product_rgba(product)
    height_px = int(round(width_px * MASTER_DEPTH_CM / MASTER_WIDTH_CM))
    master = _preserve_user_arc(prepared, width_px, height_px)

    return ProductMaster(
        rgba=master,
        width_cm=MASTER_WIDTH_CM,
        depth_cm=MASTER_DEPTH_CM,
        orientation="normal",
        orientation_uncertain=False,
    )


def _preserve_user_arc(product_rgba: np.ndarray, width: int, height: int) -> np.ndarray:
    """Normalize dimensions while retaining the photographed top boundary.

    No parametric arc is generated here. Every top-boundary sample comes from
    the user's alpha silhouette; only the sides, bottom and small bottom corners
    are made canonical.
    """
    alpha = product_rgba[:, :, 3]
    ys, xs = np.nonzero(alpha >= 24)
    if not xs.size:
        raise ValueError("产品图片中没有可用的产品轮廓")
    crop = product_rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)

    source_alpha = resized[:, :, 3] >= 24
    valid = source_alpha.any(axis=0)
    if valid.sum() < width * 0.5:
        raise ValueError("产品顶部轮廓不完整")
    top = np.zeros(width, np.float32)
    top[valid] = np.argmax(source_alpha[:, valid], axis=0)
    columns = np.arange(width)
    top[~valid] = np.interp(columns[~valid], columns[valid], top[valid])
    top = np.clip(np.rint(top), 0, height - 2).astype(np.int32)

    # Exact extracted top samples; straight vertical sides; horizontal bottom;
    # a small symmetric transition only at the two bottom corners.
    radius = max(1, int(round(min(width, height) * 0.025)))
    points = [(x, int(top[x])) for x in range(width)]
    points.append((width - 1, height - 1 - radius))
    angles = np.linspace(0, np.pi / 2, max(4, radius + 1))
    points.extend((int(round(width - 1 - radius + radius * np.cos(a))),
                   int(round(height - 1 - radius + radius * np.sin(a)))) for a in angles)
    points.append((radius, height - 1))
    points.extend((int(round(radius - radius * np.cos(a))),
                   int(round(height - 1 - radius + radius * np.sin(a))))
                  for a in angles[::-1])
    points.append((0, int(top[0])))
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [np.asarray(points, np.int32)], 255, lineType=cv2.LINE_AA)
    mask[-1, radius:width - radius] = 255

    mapped = resized.copy()
    missing = (mask >= 250) & (mapped[:, :, 3] < 250)
    if np.any(missing):
        mapped[:, :, :3] = cv2.inpaint(
            mapped[:, :, :3], missing.astype(np.uint8) * 255,
            max(3, int(round(min(width, height) * 0.025))), cv2.INPAINT_NS,
        )
    mapped[:, :, 3] = mask
    mapped[mask == 0, :3] = 0
    return mapped
