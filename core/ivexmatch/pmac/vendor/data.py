"""Inference-only subset of the PMAC training `data.py` (pmac_gfp_dapi_e200_2026-10-25/data.py), hand-written for ivex.

Exposes exactly the names the vendored model modules import. letterbox / content_mask / map_points / sample_map are verbatim
copies. Training-only names exist so the modules import, and raise if a training path is reached. The crop scale (IV/EX
magnification of the checkpoint's training subjects) is set per checkpoint with set_crop_scale() before the model is built."""
from pathlib import Path
import numpy as np
from torch.nn import functional as F

SOURCE = Path(__file__).resolve().parent; HERE = SOURCE
TRAIN = []                                   # inference only: no subject is ever a training subject here
FIELD_SIZE = 224
_CROP_SCALE = None


def set_crop_scale(scale):
    global _CROP_SCALE
    _CROP_SCALE = np.asarray(scale, float); assert _CROP_SCALE.shape == (2,) and (_CROP_SCALE > 0).all()


def crop_scale():
    assert _CROP_SCALE is not None, 'call set_crop_scale() with the checkpoint crop scale before building the model'
    return _CROP_SCALE.copy()


def letterbox(a, size=FIELD_SIZE):
    h, w = a.shape[:2]; side = ((max(h, w) + 3) // 4) * 4; left, top = (side - w) // 2, (side - h) // 2
    out = np.zeros((side, side) + a.shape[2:], np.uint8); out[top:top + h, left:left + w] = a
    return out, dict(native_wh=[w, h], resized_wh=[w * size / side, h * size / side], offset_xy=[left * size / side, top * size / side], size=size,
                     native_padded_size=side, native_offset_xy=[left, top])


def content_mask(geom, grid=14):
    edges = np.arange(grid + 1) * FIELD_SIZE / grid; left, top = geom['offset_xy']; w, h = geom['resized_wh']
    x = (edges[1:] > left) & (edges[:-1] < left + w); y = (edges[1:] > top) & (edges[:-1] < top + h)
    return y[:, None] & x[None, :]


def map_points(xy, geom):
    xy = np.asarray(xy, np.float32); wh = np.asarray(geom['native_wh'], np.float32)
    assert np.isfinite(xy).all() and (xy >= -.5).all() and (xy < wh).all()
    return (xy + .5) * (np.asarray(geom['resized_wh'], np.float32) / wh) + np.asarray(geom['offset_xy'], np.float32) - .5


def sample_map(fmap, xy, size=FIELD_SIZE):
    grid = (2 * xy + 1) / size - 1
    return F.grid_sample(fmap.float(), grid[None, None], mode='bilinear', padding_mode='border', align_corners=False)[0, :, 0].T


def is_training_record(a):
    return False


def batch_indices(a, qids, rng):
    raise RuntimeError('ivex vendor: training batches are not available in the inference package')
