#!/usr/bin/env python3
"""JY195 example for the site, from the validated benchmark (no new segmentation): one region of the exact-landmark benchmark B1.
  IV  = the JY195 B volume (39 planes on the 1.87 um PMAC grid; the whole in vivo field)
  EX  = the region's EX image (the 1.666 um grid PNG the benchmark uses)
  cells = the annotated landmark cells, drawn as 5 um discs (outlines) on their IV plane and on EX
  matches = the saved PMAC scores of the B100 run (B_train100_loso/B3/source_scores/JY195/exact_landmarks/<region>.npz), the
            ones ivex reproduces bit for bit (tests/parity/pmac_b1_jy195.py); top-k per IV cell + the annotated answer (truth_idx)
  orientation = similarity fitted to the region's annotated pairs (labelled as such; this example has no blind pose)
Output: a viewer bundle in OUT (stays on scratch; tools/publish_example.py --bundle-only uploads it to Blob examples/jy195/).
Run inside the container with the B100 env (PMAC_FOLD=JY195). usage: python tools/example_bundle_jy195.py [REGION] [OUT]"""
import json, os, sys
from pathlib import Path
import numpy as np
import cv2

REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO / 'core'))
B100 = Path('/scratch/sd5963/cell_registration_2026/microscopy_ssl/experiments/pmac_synth2_B_train100_loso_2026-10-22')
SAVED = Path('/scratch/sd5963/cell_registration_2026/microscopy_ssl/figures_pmac_synth2_2026-10-20/pmac/B_train100_loso/B3/source_scores/JY195/exact_landmarks')
REGION = sys.argv[1] if len(sys.argv) > 1 else '847_Region5'
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else f'/scratch/sd5963/ivex-match-data/examples/jy195_bundle'); OUT.mkdir(parents=True, exist_ok=True)
os.environ['PMAC_FOLD'] = 'JY195'; sys.path.insert(0, str(B100)); os.chdir(B100)
import data as ref_data, exact_landmarks
from ivexmatch import webexport as W
from ivexmatch.preprocess import IV_UM, EX_UM
from ivexmatch.register import fit_similarity

rows = {name: (b, ef) for name, b, ef in exact_landmarks.make_rows(ref_data.load_subject('JY195', allow_test=True), ref_data.Fields(allow_test=True))}
b, ef = rows[REGION]; n = int(b['n_gt']); vol = np.load(ref_data.B_GEOMETRY['JY195']['volume']); ex = np.asarray(ef['raw_ex'])
iv_xy = np.asarray(ef['iv_full_xy'][:n], float); iv_z = np.asarray(b['q_z'][:n], int); ex_xy = np.asarray(b['ex_xy'][:n], float)
s = np.load(SAVED / f'{REGION}.npz'); assert np.array_equal(s['truth'], np.arange(n))
print(REGION, 'n', n, 'IV volume', vol.shape, 'EX', ex.shape, 'PMAC top-1', int((s['combined'].argmax(1) == np.arange(n)).sum()), flush=True)

iv_img = W.write_pyramid(OUT / 'iv.ome.tif', vol, 'ZYX', IV_UM)
ex_img = W.write_pyramid(OUT / 'ex.ome.tif', ex[None] if ex.ndim == 2 else np.moveaxis(ex, -1, 0)[:1], 'CYX', EX_UM, channel_names=['GFP'])


def discs(shape, pts, r_px):
    lab = np.zeros(shape, np.int32)               # cv2 draws on int32, not uint32
    for i, (x, y) in enumerate(pts, start=1): cv2.circle(lab, (int(round(x)), int(round(y))), int(round(r_px)), int(i), -1)
    return lab


iv_planes = {}
for z in np.unique(iv_z):
    keep = np.flatnonzero(iv_z == z); lab = np.zeros(vol.shape[1:], np.int32)
    for i in keep: cv2.circle(lab, (int(round(iv_xy[i, 0])), int(round(iv_xy[i, 1]))), int(round(5 / IV_UM)), int(i + 1), -1)
    iv_planes[int(z)] = lab
iv_set = W.write_mask_set_packed(OUT, 'iv', 'landmarks', iv_planes, color=[255, 200, 0], label='annotated landmark cells')
ex_set = W.write_mask_set_packed(OUT, 'ex', 'landmarks', {0: discs(ex.shape[:2], ex_xy, 5 / EX_UM)}, color=[255, 200, 0], label='annotated landmark cells')

k = min(10, n); top = lambda M: np.argsort(-M, 1)[:, :k]
comb, app, loc = s['combined'], s['soma+neighbourhood'], s['region']
p_top1 = np.exp(np.sort(comb, 1)[:, -1]); best_q = comb.argmax(0)
cells = [[int(i + 1), int(iv_z[i]), round(float(iv_xy[i, 0]), 1), round(float(iv_xy[i, 1]), 1), round(float(p_top1[i]), 4), bool(best_q[comb[i].argmax()] == i),
          top(comb)[i].tolist(), np.round(comb[i, top(comb)[i]], 3).tolist(), top(app)[i].tolist(), np.round(app[i, top(app)[i]], 3).tolist(),
          top(loc)[i].tolist(), np.round(loc[i, top(loc)[i]], 3).tolist(), int(i)] for i in range(n)]
matches = dict(columns=['id', 'z', 'x', 'y', 'p_top1', 'mutual', 'combined_idx', 'combined', 'appearance_idx', 'appearance', 'location_idx', 'location', 'truth_idx'],
               ex_columns=['x', 'y', 'mask_set', 'mask_id'], ex_mask_sets=['landmarks'], iv_mask_set='landmarks',
               ex=[[round(float(x), 1), round(float(y), 1), 0, int(i + 1)] for i, (x, y) in enumerate(ex_xy)], cells=cells,
               note='JY195 exact-landmark benchmark B1: queries and candidates are the annotated landmark cells; truth_idx = the annotated match')
W.write_json_gz(OUT / 'matches_run0.json.gz', matches)

A, t = fit_similarity(iv_xy * IV_UM, ex_xy * EX_UM)                       # um -> um, then native px -> px
res = np.linalg.norm(iv_xy * IV_UM @ A.T + t - ex_xy * EX_UM, axis=1)
aff = W.native_affine(lambda P: ((P * IV_UM) @ A.T + t) / EX_UM)
correct = int((comb.argmax(1) == np.arange(n)).sum())
pose = dict(label=f'similarity fitted to the {n} annotated pairs (ground truth, not a blind pose)', provisional=True, iv_to_ex_native=aff,
            rotation_deg=float(np.degrees(np.arctan2(A[1, 0], A[0, 0]))), scale=float(np.sqrt(abs(np.linalg.det(A)))), median_residual_um=float(np.median(res)))
(OUT / 'pose.json').write_text(json.dumps(pose, indent=1))
files = ['iv.ome.tif', 'ex.ome.tif', iv_set['file'], ex_set['file'], 'matches_run0.json.gz', 'pose.json', 'manifest.json']
manifest = dict(version='ivex-web-2', title=f'JY195 · region {REGION.replace("_Region", " / ")} · {n} annotated cells', private=False, example=True,
                example_note=f'Mouse JY195, exact-landmark benchmark: PMAC ranks the right cell first for {correct} of {n} annotated cells. The true match is marked "true".',
                iv=dict(image=iv_img, um_xy=IV_UM, um_z=None, nz=int(vol.shape[0]), mask_sets=[iv_set], channel='GCaMP (B-normalized)',
                        default_z=int(np.bincount(iv_z).argmax())),
                ex=dict(image=ex_img, um_xy=EX_UM, mask_sets=[ex_set], channels=['GFP'], projection='EX max projection (benchmark grid)'),
                runs=[dict(name='run0', label=f'PMAC (validated benchmark run) · top-1 {correct}/{n}', kind='given', matches='matches_run0.json.gz', iv_to_ex_native=aff, match_model='landmarks')],
                pose='pose.json', match_model='landmarks', models=None, files=files, truth=True,
                source=dict(region=REGION, scores=str(SAVED / f'{REGION}.npz'), volume=ref_data.B_GEOMETRY['JY195']['volume']))
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=1))
thumbs = OUT / 'thumbs'; thumbs.mkdir(exist_ok=True)
mip = vol.max(0); cv2.imwrite(str(thumbs / 'iv.jpg'), cv2.resize(np.stack([mip] * 3, -1), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), [cv2.IMWRITE_JPEG_QUALITY, 90])
g = ex if ex.ndim == 2 else ex[..., 0]; cv2.imwrite(str(thumbs / 'ex.jpg'), np.stack([g // 4, g, g // 3], -1), [cv2.IMWRITE_JPEG_QUALITY, 90])
print(json.dumps(dict(out=str(OUT), n=n, correct=correct, pose_residual_um=round(float(np.median(res)), 2), files=files)))
