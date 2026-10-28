#!/usr/bin/env python3
"""Convert one of the lab's LUT pickles (Attila JY mice) into the OME-TIFFs the web app accepts. Run it LOCALLY on pickles you trust:
unpickling runs code, which is why the app refuses pickles from uploads.

Both schemas seen on scratch:
  JY195 / JY239 / JY180   {'in_vivo': {'Transformed': (Z, Y, X)}, 'exvivo_GCAMP': {'Transformed': ...}, 'exvivo_DAPI': {'Transformed': ...}, ...}
  jy280 / jy306 / jy316   {'in_vivo': (Z, Y, X), 'exvivo_GCAMP': (Z, Y, X)}            (bare arrays, no DAPI)
Writes OUT/<stem>_iv.ome.tif (Z, Y, X) and OUT/<stem>_ex.ome.tif (Z, C, Y, X: GCaMP [+ DAPI when it is on the same grid]); a DAPI on
another grid (JY195: (16, 2851, 4201) vs GCaMP (8, 1105, 1627)) goes to <stem>_ex_dapi.ome.tif instead. Float volumes are scaled to
uint16 over [p0.1, p99.99]. The pickles carry no pixel sizes: pass --iv-um / --iv-z-um / --ex-um to embed them (else enter them in the app).
usage: python tools/convert_lut_pickle.py LUT.pkl OUT_DIR [--iv-um 0.684 --iv-z-um 3 --ex-um 0.65 --ex-z-um 2]"""
import argparse, pickle
from pathlib import Path
import numpy as np
import tifffile

ap = argparse.ArgumentParser(); ap.add_argument('pkl'); ap.add_argument('out'); ap.add_argument('--iv-um', type=float); ap.add_argument('--iv-z-um', type=float)
ap.add_argument('--ex-um', type=float); ap.add_argument('--ex-z-um', type=float); a = ap.parse_args()
d = pickle.load(open(a.pkl, 'rb')); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); stem = Path(a.pkl).stem
vol = lambda v: np.asarray(v['Transformed'] if isinstance(v, dict) else v)


def u16(x):
    x = np.asarray(x)
    if x.dtype == np.uint16: return x
    lo, hi = np.percentile(x[::max(1, len(x) // 8)], [0.1, 99.99]); return (np.clip((x.astype(np.float32) - lo) / max(hi - lo, 1e-9), 0, 1) * 65535).astype(np.uint16)


def write(path, arr, axes, um, um_z, names=None):
    md = dict(axes=axes)
    if um: md.update(PhysicalSizeX=um, PhysicalSizeY=um, PhysicalSizeXUnit='µm', PhysicalSizeYUnit='µm')
    if um_z: md.update(PhysicalSizeZ=um_z, PhysicalSizeZUnit='µm')
    if names: md['Channel'] = dict(Name=names)
    tifffile.imwrite(path, arr, ome=True, compression='zlib', metadata=md); print(f'{path.name}: {arr.shape} {arr.dtype}, {path.stat().st_size >> 20} MB')


keys = list(d) if isinstance(d, dict) else []
assert 'in_vivo' in keys and 'exvivo_GCAMP' in keys, f'unexpected pickle layout: keys {keys[:12]}'
iv = u16(vol(d['in_vivo'])); write(out / f'{stem}_iv.ome.tif', iv, 'ZYX', a.iv_um, a.iv_z_um)
g = u16(vol(d['exvivo_GCAMP'])); dapi = u16(vol(d['exvivo_DAPI'])) if 'exvivo_DAPI' in keys else None
if dapi is not None and dapi.shape == g.shape:
    write(out / f'{stem}_ex.ome.tif', np.stack([g, dapi], 1), 'ZCYX', a.ex_um, a.ex_z_um, ['GFP', 'DAPI'])
else:
    write(out / f'{stem}_ex.ome.tif', g, 'ZYX', a.ex_um, a.ex_z_um)
    if dapi is not None: print(f'DAPI is on another grid {dapi.shape} than GCaMP {g.shape}: written separately'); write(out / f'{stem}_ex_dapi.ome.tif', dapi, 'ZYX', None, None)
