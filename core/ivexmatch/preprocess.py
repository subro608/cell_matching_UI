"""Bring uploaded IV / EX data onto the grids PMAC was trained on (the JY369 recipe, jy369_pmac_gfpdapi_2026-10-25).

  IV  native planes -> (optional flip / rotation) -> INTER_AREA resample to IV_UM -> preprocessing "B": p1-p99.5 of the NONZERO
      voxels of every max(1, Z // 8)-th plane -> uint8 (run_pmac.build_volume / prepare_B_volumes.py).
  EX  GFP max projection over the chosen depth band -> INTER_AREA resample to EX_UM -> p1-p99.5 stretch (seg_ex.py); with DAPI,
      composite (DAPI, GFP, DAPI) where DAPI is stretched p0.5-p99.8.
Point maps convert pixel centres between native and working grids: (xy + .5) * (new / old) - .5."""
from __future__ import annotations
import numpy as np
import cv2

IV_UM = 1.87                                    # jy306 IV grid PMAC was trained on
EX_UM = 0.645511602259543 * 4200 / 1627         # jy306 EX MIP grid (1.666 um/px)


def grid_shape(hw, um, target_um):
    return round(hw[0] * um / target_um), round(hw[1] * um / target_um)


def to_grid(xy, native_hw, grid_hw):
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    return ((xy + .5) * np.asarray([grid_hw[1] / native_hw[1], grid_hw[0] / native_hw[0]]) - .5).astype(np.float32)


def from_grid(xy, native_hw, grid_hw):
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    return ((xy + .5) * np.asarray([native_hw[1] / grid_hw[1], native_hw[0] / grid_hw[0]]) - .5).astype(np.float32)


def stretch(a, p):
    lo, hi = np.percentile(a, p); return (np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def b_normalize(vol):
    """Preprocessing B on a float (Z, H, W) volume -> uint8, stats."""
    vol = np.asarray(vol, np.float32); sample = vol[::max(1, len(vol) // 8)]; nzv = sample[sample != 0]
    assert nzv.size, 'IV volume is all zeros'
    lo, hi = np.percentile(nzv, [1, 99.5]); hi = max(hi, lo + 1)
    return (np.clip((vol - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8), dict(lo=float(lo), hi=float(hi))


def iv_volume(native, um_xy, flip=False, rot180=False):
    """native (Z, H, W) -> uint8 (Z, h, w) on IV_UM, B-normalized. flip = mirror x; rot180 = the JY369 turn. Arbitrary angles are
    applied by ivexmatch.pose.apply_pose instead (it also maps the cell coordinates)."""
    native = np.asarray(native); Z, H, W = native.shape; h, w = grid_shape((H, W), um_xy, IV_UM); planes = []
    for z in range(Z):
        p = np.asarray(native[z], np.float32)
        if flip: p = p[:, ::-1]
        if rot180: p = p[::-1, ::-1]
        planes.append(cv2.resize(np.ascontiguousarray(p), (w, h), interpolation=cv2.INTER_AREA))
    v8, stats = b_normalize(np.stack(planes))
    return v8, dict(stats, native_hw=[H, W], grid_hw=[h, w], um=IV_UM, native_um=float(um_xy), flip=bool(flip), rot180=bool(rot180))


def ex_image(gfp_planes, um_xy, dapi_planes=None):
    """EX z-band (Z, H, W) or 2D MIP -> uint8 (h, w) grayscale, or (h, w, 3) composite (DAPI, GFP, DAPI) when DAPI is given."""
    gfp = np.asarray(gfp_planes, np.float32); gfp = gfp.max(0) if gfp.ndim == 3 else gfp
    H, W = gfp.shape; h, w = grid_shape((H, W), um_xy, EX_UM)
    g8 = stretch(cv2.resize(gfp, (w, h), interpolation=cv2.INTER_AREA), [1, 99.5])
    meta = dict(native_hw=[H, W], grid_hw=[h, w], um=EX_UM, native_um=float(um_xy), composite=dapi_planes is not None)
    if dapi_planes is None: return g8, meta
    dapi = np.asarray(dapi_planes, np.float32); dapi = dapi.max(0) if dapi.ndim == 3 else dapi; assert dapi.shape == gfp.shape
    d8 = stretch(cv2.resize(dapi, (w, h), interpolation=cv2.INTER_AREA), [0.5, 99.8])
    return np.stack([d8, g8, d8], -1), meta
