#!/usr/bin/env python3
"""LOCAL demo web bundle from the JY369 row3 P02 PMAC run (jy369_pmac_gfpdapi_2026-10-25) for viewer development.
UNPUBLISHED DATA: output stays on HPC scratch (default /scratch/sd5963/ivex-match-data/demo_jy369), never in the repo or online.

IV  = stitched Ch2 (74 planes, 0.6836 um/px, 3 um z); masks: Cellpose3 denoise+cyto3 2D (the PMAC queries), fine-tuned cyto2 / cyto3 3D.
EX  = row3 P02 GFP + DAPI max projections over the 7 planes (0.324 um/px); masks: stock cyto2 GFP and DAPI nuclei (the PMAC
      candidates), fine-tuned cyto3 3D (label max projection).
Matches = out/stitched_Ch2.npz (GFP+DAPI checkpoint, IV turned 180 deg before PMAC). Pose = the provisional assistant-picked
similarity (expanded_review_20260917, NOT ground truth), as a native-pixel affine, labelled as such.
usage: python tools/demo_bundle_jy369.py [OUT]"""
import json, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO / 'core'))
from ivexmatch import webexport as W
from ivexmatch.preprocess import from_grid

E = Path('/scratch/sd5963/cell_registration_2026/microscopy_ssl/experiments/jy369_pmac_gfpdapi_2026-10-25'); sys.path.insert(0, str(E))
import jy369_common as J
REG = Path('/scratch/sd5963/cell_registration_2026/cellinvariance/data/new_data_pyramidal/registration')
SEG = REG / 'segmentation'; POSE = REG / 'expanded_review_20260917/ex20_row3_ver3_P02_GFP/registration.json'
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else '/scratch/sd5963/ivex-match-data/demo_jy369'); OUT.mkdir(parents=True, exist_ok=True)
IV_UM, IV_Z_UM, EX_UM = J.IV_SETS['stitched']['um'], 3.0, J.EX_NATIVE_UM
t0 = time.time(); log = lambda s: print(f'[{time.time() - t0:5.0f}s] {s}', flush=True)

# images
nz = J.n_planes('iv_stitched'); iv = np.stack([J.load('iv_stitched', 'Ch2', z) for z in range(nz)])
iv_img = W.write_pyramid(OUT / 'iv.ome.tif', iv, 'ZYX', IV_UM, IV_Z_UM); log(f'IV pyramid {iv_img}')
gfp = np.stack([J.load(J.EX_DS, 'GFP', z) for z in range(7)]).max(0); dapi = np.stack([J.load(J.EX_DS, 'DAPI', z) for z in range(7)]).max(0)
ex_img = W.write_pyramid(OUT / 'ex.ome.tif', np.stack([gfp, dapi]), 'CYX', EX_UM, channel_names=['GFP', 'DAPI']); log(f'EX pyramid {ex_img}')

# masks
iv_sets = []; pm = np.load(E / 'masks/iv_stitched_Ch2.npz')['labels']
iv_sets.append(W.write_mask_set(OUT, 'iv', 'cp3_dn_cyto3', {z: pm[z] for z in range(nz)}, color=[255, 200, 0], label='Cellpose3 denoise+cyto3 (PMAC queries)'))
for name, f, col in (('ft_cyto2_3d', 'cyto2', [0, 220, 255]), ('ft_cyto3_3d', 'cyto3', [255, 80, 200])):
    m = np.load(SEG / f'iv/full_cyto23_3d_ft/iv_stitched_Ch2/{f}_masks_zyx.npz')['masks']
    iv_sets.append(W.write_mask_set(OUT, 'iv', name, {z: m[z] for z in range(nz)}, color=col, label=f'fine-tuned {f} 3D')); del m
log('IV masks ' + ', '.join(f"{s['name']} {sum(s['counts'].values())}" for s in iv_sets))
exm = np.load(E / 'ex/masks.npz'); dapi_scale = 6075 / exm['dapi_footprint'].shape[0]
ex_sets = [W.write_mask_set(OUT, 'ex', 'cyto2_gfp', {0: exm['gfp2d']}, color=[255, 200, 0], label='stock cyto2 GFP (PMAC candidates)'),
           W.write_mask_set(OUT, 'ex', 'nuclei_dapi', {0: exm['dapi_footprint']}, scale=dapi_scale, color=[80, 140, 255], label='DAPI nuclei (PMAC candidates)')]
m = np.load(SEG / 'ex/full_cyto23_3d_ft/ex20_row3_ver3_P02_GFP/cyto3_masks_zyx.npz')['masks'].max(0)
ex_sets.append(W.write_mask_set(OUT, 'ex', 'ft_cyto3_3d', {0: m}, color=[255, 80, 200], label='fine-tuned cyto3 3D (max projection)')); del m
log('EX masks ' + ', '.join(f"{s['name']} {sum(s['counts'].values())}" for s in ex_sets))

# matches: candidate i -> native EX px and the mask it came from (GFP label id or DAPI label id)
r = np.load(E / 'out/stitched_Ch2.npz'); cand = json.loads((E / 'ex/candidates.json').read_text()); n_gfp = cand['n_gfp']
ex_native = from_grid(r['ex_xy'], (6075, 6075), (J.EX_SIDE, J.EX_SIDE))
ex_list = [[round(float(x), 1), round(float(y), 1), 0 if i < n_gfp else 1, int(exm['gfp_ids'][i] if i < n_gfp else exm['dapi_ids'][i - n_gfp])]
           for i, (x, y) in enumerate(ex_native)]
rd = lambda a: np.round(a, 3).tolist()
cells = [[int(c[1]), int(c[0]), round(float(c[2]), 1), round(float(c[3]), 1), round(float(r['p_top1'][i]), 4), bool(r['mutual'][i]),
          r['combined_idx'][i].tolist(), rd(r['combined_score'][i]), r['soma+neighbourhood_idx'][i].tolist(), rd(r['soma+neighbourhood_score'][i]),
          r['region_idx'][i].tolist(), rd(r['region_score'][i])] for i, c in enumerate(r['cells'])]
(OUT / 'matches.json').write_text(json.dumps(dict(
    columns=['id', 'z', 'x', 'y', 'p_top1', 'mutual', 'combined_idx', 'combined', 'appearance_idx', 'appearance', 'location_idx', 'location'],
    ex_columns=['x', 'y', 'mask_set', 'mask_id'], ex_mask_sets=['cyto2_gfp', 'nuclei_dapi'], iv_mask_set='cp3_dn_cyto3', ex=ex_list, cells=cells,
    checkpoint='gfpdapi_synth best val loss (epoch 33)', note='IV turned 180 deg before PMAC; scores = log P(cell | query) for combined'), separators=(',', ':')))
log(f'matches: {len(cells)} IV cell-planes x {len(ex_list)} EX candidates')

# provisional pose (assistant-picked, NOT ground truth)
reg = json.loads(POSE.read_text()); M = np.asarray(reg['matrix_iv_um_to_ex_um'])
aff = W.native_affine(lambda P: ((P * IV_UM) @ M[:, :2].T + M[:, 2]) / EX_UM)
(OUT / 'pose.json').write_text(json.dumps(dict(label='provisional pose (assistant-picked 09-16/17, NOT ground truth)', provisional=True,
    iv_to_ex_native=aff, scale=reg['scale'], rotation_deg=reg['rotation_degrees'], ex_z_range=reg['ex_z_range'], source=str(POSE)), indent=1))

manifest = dict(version='ivex-web-1', title='JY369 row3 P02, stitched Ch2 (LOCAL DEMO, unpublished)', private=True,
                iv=dict(image=iv_img, um_xy=IV_UM, um_z=IV_Z_UM, nz=nz, default_z=52, mask_sets=iv_sets, channel='Ch2'),
                ex=dict(image=ex_img, um_xy=EX_UM, mask_sets=ex_sets, channels=['GFP', 'DAPI'], projection='max over 7 planes'),
                matches='matches.json', pose='pose.json')
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=1)); log(f'demo bundle -> {OUT}')
