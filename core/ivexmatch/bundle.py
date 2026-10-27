"""Session bundle writer (layout from the plan; the viewer and the worker read the same files).

  meta.json                      inputs, voxel sizes, grids, PMAC checkpoint, timings
  img/{iv,ex}_grid.npy           working-grid images PMAC saw (IV B-normalized (Z, H, W), EX (H, W[, 3]))
  masks/{iv,ex}/<model>.npz      written by segment.py (labels + cells: z, id, x, y, area in native px)
  pose/pose.json                 blind pose: best / runner-up / p-value / hypotheses / inlier pairs
  pmac/<run>/topk.json           per IV cell-plane: id, z, x, y (native IV px), top-k EX ids + combined / appearance / location
                                 scores, P(top1), mutual-best; one run per pose hypothesis (runner-up when close)
Image tiles for the web viewer (multiscale) are added in M2."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np


def _ids(cells): return cells[:, 1].astype(int).tolist()


def _json(o): return o.tolist() if hasattr(o, 'tolist') else float(o)


def write_run(out, name, pose, match_model, iv_cells, ex_cells, res):
    """pmac/<name>/topk.json; iv_cells / ex_cells are the match model's (z, id, x, y, area) rows in native px."""
    ex_ids = np.asarray(_ids(ex_cells)); d = Path(out) / 'pmac' / name; d.mkdir(parents=True, exist_ok=True)
    cells = [dict(z=int(c[0]), id=int(c[1]), x=round(float(c[2]), 2), y=round(float(c[3]), 2),
                  top_ex_ids=ex_ids[res['combined_idx'][i]].tolist(), combined=np.round(res['combined_score'][i], 4).tolist(),
                  appearance_top_ex_ids=ex_ids[res['soma+neighbourhood_idx'][i]].tolist(), appearance=np.round(res['soma+neighbourhood_score'][i], 4).tolist(),
                  location_top_ex_ids=ex_ids[res['region_idx'][i]].tolist(), location=np.round(res['region_score'][i], 4).tolist(),
                  p_top1=round(float(res['p_top1'][i]), 4), mutual=bool(res['mutual'][i])) for i, c in enumerate(iv_cells)]
    (d / 'topk.json').write_text(json.dumps(dict(pose=pose, match_model=match_model, ex_cell_ids=ex_ids.tolist(), cells=cells), default=_json))


def write(out, args, iv, ex, pose, runs, pmac, seconds, gpu):
    out = Path(out); (out / 'img').mkdir(parents=True, exist_ok=True); (out / 'pose').mkdir(exist_ok=True)
    np.save(out / 'img/iv_grid.npy', iv['grid']); np.save(out / 'img/ex_grid.npy', ex['grid'])
    (out / 'pose/pose.json').write_text(json.dumps(pose, indent=1, default=_json))
    for r in runs: write_run(out, r['name'], r['pose'], r.get('match_model', args['match_model']), iv['cells'], ex['cells'], r['result'])
    meta = dict(version='ivex-bundle-1', created=time.strftime('%Y-%m-%dT%H:%M:%S%z'), args=args, pmac=pmac, gpu=gpu, seconds=seconds, runs=[r['name'] for r in runs],
                iv=dict(iv['meta'], n_cell_planes=int(len(iv['cells']))), ex=dict(ex['meta'], n_cells=int(len(ex['cells']))))
    (out / 'meta.json').write_text(json.dumps(meta, indent=1))
