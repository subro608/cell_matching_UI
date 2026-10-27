#!/usr/bin/env python3
"""LOCAL end-to-end test input: a small IV / EX crop of JY369 row3 P02 (unpublished; stays on scratch, never in the repo or online).
IV = stitched Ch2 planes Z0:Z1, a SIDE x SIDE px window at the field centre; EX = GFP + DAPI, all 7 planes, the window that the
provisional pose maps that IV window onto (plus a margin). Written as plain TIFFs the way a user would upload them:
iv.tif (Z, H, W) uint16, ex.tif (Z, C, H, W) uint16. usage: python tools/make_test_crop_jy369.py [OUT] [SIDE] [Z0:Z1]"""
import json, sys
from pathlib import Path
import numpy as np
import tifffile

E = Path('/scratch/sd5963/cell_registration_2026/microscopy_ssl/experiments/jy369_pmac_gfpdapi_2026-10-25'); sys.path.insert(0, str(E))
import jy369_common as J
POSE = Path('/scratch/sd5963/cell_registration_2026/cellinvariance/data/new_data_pyramidal/registration/expanded_review_20260917/ex20_row3_ver3_P02_GFP/registration.json')
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else '/scratch/sd5963/ivex-match-data/test_jy369_crop'); OUT.mkdir(parents=True, exist_ok=True)
SIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 1100; Z0, Z1 = (int(v) for v in (sys.argv[3] if len(sys.argv) > 3 else '42:62').split(':'))
IV_UM, EX_UM = J.IV_SETS['stitched']['um'], J.EX_NATIVE_UM
iv = np.stack([J.load('iv_stitched', 'Ch2', z) for z in range(Z0, Z1)]); H, W = iv.shape[1:]
cy, cx = H // 2, W // 2; y0, x0 = cy - SIDE // 2, cx - SIDE // 2; iv = iv[:, y0:y0 + SIDE, x0:x0 + SIDE]
M = np.asarray(json.loads(POSE.read_text())['matrix_iv_um_to_ex_um'])
corners = np.array([[x0, y0], [x0 + SIDE, y0], [x0, y0 + SIDE], [x0 + SIDE, y0 + SIDE]], float) * IV_UM
ex_px = (corners @ M[:, :2].T + M[:, 2]) / EX_UM; lo = np.floor(ex_px.min(0) - 150).astype(int); hi = np.ceil(ex_px.max(0) + 150).astype(int)
gfp = np.stack([J.load(J.EX_DS, 'GFP', z) for z in range(7)]); dapi = np.stack([J.load(J.EX_DS, 'DAPI', z) for z in range(7)])
lo = np.maximum(lo, 0); hi = np.minimum(hi, gfp.shape[1:][::-1])
ex = np.stack([gfp[:, lo[1]:hi[1], lo[0]:hi[0]], dapi[:, lo[1]:hi[1], lo[0]:hi[0]]], 1)
tifffile.imwrite(OUT / 'iv.tif', iv.astype(np.uint16)); tifffile.imwrite(OUT / 'ex.tif', ex.astype(np.uint16))
info = dict(iv_um=IV_UM, iv_z_um=3.0, ex_um=EX_UM, iv_window=[x0, y0, SIDE, SIDE, Z0, Z1], ex_window=[int(lo[0]), int(lo[1]), int(hi[0] - lo[0]), int(hi[1] - lo[1])],
            iv_shape=list(iv.shape), ex_shape=list(ex.shape), note='JY369 row3 P02 crop for ivex end-to-end tests; unpublished')
(OUT / 'info.json').write_text(json.dumps(info, indent=1)); print(json.dumps(info))
