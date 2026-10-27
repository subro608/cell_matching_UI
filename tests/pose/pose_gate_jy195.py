"""PMAC blind-pose gate, JY195 part (plan > Verification > PMAC blind-pose gate; pre-declared, no tuning on these answers).

JY195 is the held-out fold of original_best.pt. 8 trials = the 8 regions with the most landmarks, each given one synthetic
transform of its IV (theta0 in 0/90/180/270 + 4 off-grid angles, half mirrored). Queries = the region's IV detector cells, gallery =
its EX detector candidates (B100 harness, as in testing). The blind pose never sees landmarks; they only grade it:
  flip right; |theta - theta_true| <= 5 deg (theta_true = Umeyama on the transformed landmarks); family-wise p < 0.05;
  median landmark error of the recovered pose <= 15 um.
Gate (with JY369 and CellGPR, 24 trials): >= 90% of trials pass. Writes logs/pose_gate_jy195.json."""
import json, os, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
FIG = Path('/scratch/sd5963/cell_registration_2026/microscopy_ssl/figures_pmac_synth2_2026-10-20/pmac')
B100 = Path('/scratch/sd5963/cell_registration_2026/microscopy_ssl/experiments/pmac_synth2_B_train100_loso_2026-10-22')
CKPT = FIG / 'spherical_similarity_analysis/checkpoints/original_best.pt'
TRANSFORMS = [(False, 0.), (True, 90.), (False, 180.), (True, 270.), (True, 37.), (False, 143.), (True, 211.), (False, 318.)]

os.environ['PMAC_FOLD'] = 'JY195'; sys.path.insert(0, str(B100)); os.chdir(B100)
import data as ref_data, exact_landmarks
sys.path.insert(0, str(REPO / 'core'))
import torch
from ivexmatch.pmac import PMAC
from ivexmatch.pose import blind_pose, rotate_plane, flip_xy, pose_to_iv
from ivexmatch.preprocess import IV_UM, EX_UM
from ivexmatch.register import fit_similarity


def transform_volume(vol, zs, flip, theta):
    planes = {}; M = None
    for z in zs: planes[z], M = rotate_plane(np.asarray(vol[z]), theta, flip)
    H2, W2 = next(iter(planes.values())).shape; out = np.zeros((len(vol), H2, W2), np.uint8)
    for z, p in planes.items(): out[z] = p
    return out, M


def transform_xy(xy, M, W, flip): src = flip_xy(xy, W) if flip else np.asarray(xy, float); return src @ M[:, :2].T + M[:, 2]


def wrap(d): return (d + 180) % 360 - 180


t0 = time.time()
scale = json.loads((B100 / 'folds/JY195/scale.json').read_text())['crop_scale']
m = PMAC(CKPT, crop_scale=scale)
fields = ref_data.Fields(allow_test=True); rows = dict(ref_data.load_subject('JY195', allow_test=True))
exact = {name: (b, ef) for name, b, ef in exact_landmarks.make_rows(list(rows.items()), fields)}
volume = np.load(ref_data.B_GEOMETRY['JY195']['volume'], mmap_mode='r'); W = volume.shape[2]
regions = sorted(exact, key=lambda r: -int(exact[r][0]['n_gt']))[:len(TRANSFORMS)]
trials = []
for region, (flip0, th0) in zip(regions, TRANSFORMS):
    a = rows[region]; f = fields.get(a); b, ef = exact[region]; n = int(b['n_gt'])
    cells = np.c_[f['iv_full_xy'], a['q_z']].astype(float); zs = sorted(set(a['q_z'].astype(int).tolist()))
    vol_t, M = transform_volume(volume, zs, flip0, th0)
    cells_t = cells.copy(); cells_t[:, :2] = transform_xy(cells[:, :2], M, W, flip0)
    out = blind_pose(m, vol_t, cells_t, f['raw_ex'], a['ex_xy'], log=lambda s: print(f'  [{region}] {s}', flush=True))
    L_t = transform_xy(ef['iv_full_xy'][:n], M, W, flip0); L_ex = np.asarray(b['ex_xy'], float)
    A_true, _ = fit_similarity((flip_xy(L_t, vol_t.shape[2]) if flip0 else L_t) * IV_UM, L_ex * EX_UM)
    theta_true = float(np.degrees(np.arctan2(A_true[1, 0], A_true[0, 0])) % 360)
    p = out['pose']; trial = dict(region=region, flip0=flip0, theta0=th0, n_iv_cells=len(cells), n_ex=len(a['ex_xy']), n_landmarks=n,
                                  theta_true=theta_true, p=out['p_familywise'], best=out['best'], runner_up=out['runner_up'], seconds=out['seconds'])
    if p is not None:
        err = np.linalg.norm(pose_to_iv(p, L_t) - L_ex, axis=1) * EX_UM
        trial.update(flip=p['flip'], theta=p['theta'], scale=p['scale'], n_inliers=p['n_inliers'], median_err_um=float(np.median(err)),
                     theta_err=float(abs(wrap(p['theta'] - theta_true))) if p['flip'] == flip0 else None)
        trial['passed'] = bool(p['flip'] == flip0 and trial['theta_err'] <= 5 and out['p_familywise'] < 0.05 and trial['median_err_um'] <= 15)
    else:
        trial['passed'] = False
    trials.append(trial)
    print(region, {k: trial.get(k) for k in ('flip0', 'theta0', 'flip', 'theta', 'theta_true', 'theta_err', 'p', 'median_err_um', 'n_inliers', 'passed', 'seconds')}, flush=True)
report = dict(dataset='JY195', checkpoint=str(CKPT), gpu=torch.cuda.get_device_name(0), trials=trials, n_pass=sum(t['passed'] for t in trials),
              n=len(trials), minutes=round((time.time() - t0) / 60, 1))
(REPO / 'logs').mkdir(exist_ok=True); (REPO / 'logs/pose_gate_jy195.json').write_text(json.dumps(report, indent=1, default=float))
print('POSE GATE JY195', report['n_pass'], '/', report['n'], 'passed', report['minutes'], 'min', flush=True)
