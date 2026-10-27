#!/usr/bin/env python3
"""Synthetic IV / EX pair for testing the cloud path without real data: random somata (Gaussian blobs) in a 3D IV stack and the same
cells in an EX GFP + DAPI image related by a known mirror / rotation / scale. Not biology; PMAC scores on it mean nothing.
usage: python make_synthetic_pair.py OUT [--theta 40] [--flip 1] [--scale 0.95]"""
import argparse, json
from pathlib import Path
import numpy as np
import tifffile

ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--theta', type=float, default=40); ap.add_argument('--flip', type=int, default=1)
ap.add_argument('--scale', type=float, default=0.95); ap.add_argument('--seed', type=int, default=0); a = ap.parse_args()
rng = np.random.default_rng(a.seed); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
IV_UM, IV_Z, NZ, IV_PX, EX_UM, EX_PX = 1.0, 3.0, 8, 480, 0.6, 1100
n = 260; cells = np.c_[rng.uniform(30, IV_PX * IV_UM - 30, (n, 2)), rng.uniform(0, NZ * IV_Z, n)]; radius = rng.uniform(5, 8, n)
yy, xx = np.mgrid[0:IV_PX, 0:IV_PX] * IV_UM
iv = np.zeros((NZ, IV_PX, IV_PX), np.float32)
for (x, y, zc), r in zip(cells, radius):
    s = slice(max(0, int((y - 3 * r) / IV_UM)), int((y + 3 * r) / IV_UM) + 1), slice(max(0, int((x - 3 * r) / IV_UM)), int((x + 3 * r) / IV_UM) + 1)
    for z in range(NZ):
        w = np.exp(-((z * IV_Z - zc) ** 2) / (2 * 4.0 ** 2))
        if w > 0.05: iv[z][s] += w * np.exp(-((xx[s] - x) ** 2 + (yy[s] - y) ** 2) / (2 * (r / 2) ** 2))
iv = (iv * 3000 + rng.normal(200, 40, iv.shape)).clip(0, 65535).astype(np.uint16)
th = np.radians(a.theta); R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]); c_iv = np.array([IV_PX * IV_UM / 2] * 2)
F = np.diag([-1., 1.]) if a.flip else np.eye(2)
ex_xy = (cells[:, :2] - c_iv) @ (a.scale * R @ F).T + EX_PX * EX_UM / 2
yy, xx = np.mgrid[0:EX_PX, 0:EX_PX] * EX_UM; gfp = np.zeros((EX_PX, EX_PX), np.float32); dapi = np.zeros_like(gfp)
for (x, y), r in zip(ex_xy, radius * a.scale):
    s = slice(max(0, int((y - 3 * r) / EX_UM)), int((y + 3 * r) / EX_UM) + 1), slice(max(0, int((x - 3 * r) / EX_UM)), int((x + 3 * r) / EX_UM) + 1)
    d2 = (xx[s] - x) ** 2 + (yy[s] - y) ** 2; gfp[s] += np.exp(-d2 / (2 * (r / 2) ** 2)); dapi[s] += np.exp(-d2 / (2 * (r / 4) ** 2))
ex = np.stack([np.stack([gfp, dapi])] * 3) * 2500 + rng.normal(150, 30, (3, 2, EX_PX, EX_PX))
tifffile.imwrite(out / 'iv.tif', iv); tifffile.imwrite(out / 'ex.tif', ex.clip(0, 65535).astype(np.uint16))
(out / 'truth.json').write_text(json.dumps(dict(iv_um=IV_UM, iv_z_um=IV_Z, ex_um=EX_UM, theta=a.theta, flip=bool(a.flip), scale=a.scale, n_cells=n), indent=1))
print(json.dumps(dict(iv=list(iv.shape), ex=list(ex.shape), theta=a.theta, flip=a.flip, scale=a.scale)))
