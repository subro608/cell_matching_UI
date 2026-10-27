"""Segmentation parity on the JY369 test crop: the wrapper's IV masks (ivex analyze, 3D) vs the reference full-field masks
(registration/segmentation/iv/full_cyto23_3d_ft/iv_stitched_Ch2/<model>_masks_zyx.npz) cut to the same window. Reports per-model cell
counts and the share of reference cells (centroid inside the window, >= 8 px from its edge) with a wrapper centroid within 4 px on
the same plane, and the mean IoU of those pairs. Not bit parity: the reference ran on the whole field (different tiles / context).
usage: python parity_jy369_crop.py SESSION_DIR"""
import json, sys
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi

S = Path(sys.argv[1]); info = json.loads(Path('/scratch/sd5963/ivex-match-data/test_jy369_crop/info.json').read_text())
x0, y0, w, h, z0, z1 = info['iv_window']
REF = Path('/scratch/sd5963/cell_registration_2026/cellinvariance/data/new_data_pyramidal/registration/segmentation/iv/full_cyto23_3d_ft/iv_stitched_Ch2')
out = {}
for model, ref_name in (('cyto3_ft', 'cyto3'), ('cyto2_ft', 'cyto2')):
    mine = np.load(S / 'analysis/masks/iv' / f'{model}.npz')['labels']; ref = np.load(REF / f'{ref_name}_masks_zyx.npz')['masks'][z0:z1, y0:y0 + h, x0:x0 + w]
    hit = tot = 0; ious = []
    for z in range(len(ref)):
        r, m = ref[z], mine[z]; rid = np.unique(r); rid = rid[rid > 0]
        if not len(rid): continue
        rc = np.asarray(ndi.center_of_mass(r > 0, r, rid)).reshape(-1, 2); mid = np.unique(m); mid = mid[mid > 0]
        mc = np.asarray(ndi.center_of_mass(m > 0, m, mid)).reshape(-1, 2) if len(mid) else np.zeros((0, 2))
        keep = (rc[:, 0] >= 8) & (rc[:, 0] < h - 8) & (rc[:, 1] >= 8) & (rc[:, 1] < w - 8); tot += int(keep.sum())
        for i in np.flatnonzero(keep):
            if not len(mc): continue
            d = np.linalg.norm(mc - rc[i], axis=1); j = int(d.argmin())
            if d[j] <= 4:
                hit += 1; a, b = r == rid[i], m == mid[j]; ious.append((a & b).sum() / (a | b).sum())
    out[model] = dict(wrapper_cell_planes=int(sum(len(np.unique(p)) - 1 for p in mine)), reference_cell_planes=int(sum(len(np.unique(p)) - 1 for p in ref)),
                      reference_interior=tot, matched_within_4px=hit, recall=round(hit / max(tot, 1), 3), mean_iou=round(float(np.mean(ious)), 3) if ious else None)
print(json.dumps(out, indent=1))
