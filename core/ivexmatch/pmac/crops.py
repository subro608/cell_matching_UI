"""Soma / neighbourhood crops exactly as PMAC was trained and evaluated.

`_reflect_crop`, `_resize_u8` and `extract_single` are verbatim copies of crop_data.py (all 7 copies in experiments/ share md5
c3d3c030907f48965e11f0b86e6b429d). `_resize_u8` also accepts (h, w, 3) composites (PIL RGB resizes each band like mode L);
`to_model_layout` turns those into the (3, 224, 224) layout the model expects."""
from __future__ import annotations
from typing import Sequence
import numpy as np
from PIL import Image


def _reflect_crop(image: np.ndarray, center_xy: Sequence[float], half_wh: Sequence[float]) -> np.ndarray:
    height, width = image.shape[:2]
    half_x = max(1.0, float(half_wh[0]))
    half_y = max(1.0, float(half_wh[1]))
    x0 = int(np.floor(center_xy[0] - half_x))
    x1 = int(np.ceil(center_xy[0] + half_x))
    y0 = int(np.floor(center_xy[1] - half_y))
    y1 = int(np.ceil(center_xy[1] + half_y))

    pad_x0 = max(0, -x0)
    pad_y0 = max(0, -y0)
    pad_x1 = max(0, x1 - width)
    pad_y1 = max(0, y1 - height)
    if pad_x0 or pad_y0 or pad_x1 or pad_y1:
        mode = "reflect" if (max(pad_x0, pad_x1) < width and max(pad_y0, pad_y1) < height) else "edge"
        pad = ((pad_y0, pad_y1), (pad_x0, pad_x1)) + ((0, 0),) * (image.ndim - 2)
        image = np.pad(image, pad, mode=mode)
        x0 += pad_x0
        x1 += pad_x0
        y0 += pad_y0
        y1 += pad_y0
    return image[y0:y1, x0:x1]


def _resize_u8(patch: np.ndarray, out_px: int) -> np.ndarray:
    if patch.size == 0:
        return np.zeros((out_px, out_px) + patch.shape[2:], dtype=np.uint8)
    image = Image.fromarray(np.asarray(patch, dtype=np.uint8))
    return np.asarray(image.resize((out_px, out_px), Image.BILINEAR), dtype=np.uint8)


def extract_single(image: np.ndarray, xy: Sequence[float], half_global: np.ndarray, half_local: np.ndarray,
                   out_px: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        _resize_u8(_reflect_crop(image, xy, half_global), out_px),
        _resize_u8(_reflect_crop(image, xy, half_local), out_px),
    )


def to_model_layout(crops: np.ndarray) -> np.ndarray:
    """(n, 224, 224) stays; (n, 224, 224, 3) composites -> (n, 3, 224, 224)."""
    return crops if crops.ndim == 3 else np.ascontiguousarray(crops.transpose(0, 3, 1, 2))
