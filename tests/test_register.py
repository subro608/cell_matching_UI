"""Registration parity (plan > Verification): ivexmatch.register vs the original fits on the JY195 ground-truth pairs.

For every JY195 region in MARCO/data/cellreg_stack/index.json (src_kps -> trg_kps, grid px):
  * affine == prior_eval_affine.affine_fit (max abs diff < 1e-6 on the mapped points);
  * similarity == prior_eval_affine.similarity_fit (same);
  * leave-one-out affine errors == a direct re-implementation with affine_fit.
Run: python tests/test_register.py  (CPU, seconds)."""
import importlib.util, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO / 'core'))
from ivexmatch.register import Transform, loo_errors, slicewise

EXP = Path('/scratch/sd5963/cell_registration_2026/microscopy_ssl/experiments')
INDEX = Path('/scratch/sd5963/cell_registration_2026/MARCO/data/cellreg_stack/index.json')


def reference_fits():
    """affine_fit / similarity_fit from prior_eval_affine.py, extracted by source so its script body does not run."""
    src = (EXP / 'blind_pose_loso_JY195_2026-10-15/prior_eval_affine.py').read_text(); ns = {'np': np}
    for name in ('affine_fit', 'similarity_fit'):
        start = src.index(f'def {name}('); end = src.find('\ndef ', start + 1); exec(src[start:end if end > 0 else None], ns)
    return ns['affine_fit'], ns['similarity_fit']


def main():
    affine_fit, similarity_fit = reference_fits()
    rows = [r for r in json.loads(INDEX.read_text()) if r['subject'] == 'JY195']; worst = dict(affine=0., similarity=0., loo=0.); n = 0
    for r in rows:
        X, Y = np.asarray(r['src_kps'], float), np.asarray(r['trg_kps'], float)
        if len(X) < 4: continue
        n += 1; P = X + np.random.default_rng(0).normal(0, 5, X.shape)
        worst['affine'] = max(worst['affine'], np.abs(Transform('affine', X, Y)(P) - np.c_[P, np.ones(len(P))] @ affine_fit(X, Y)).max())
        worst['similarity'] = max(worst['similarity'], np.abs(Transform('similarity', X, Y)(P) - similarity_fit(X, Y)(P)).max())
        ref_loo = [np.linalg.norm(np.r_[X[i], 1] @ affine_fit(np.delete(X, i, 0), np.delete(Y, i, 0)) - Y[i]) for i in range(len(X))]
        worst['loo'] = max(worst['loo'], np.abs(loo_errors('affine', X, Y) - ref_loo).max())
    z = np.asarray(rows[0]['src_z'], float); X = np.asarray(rows[0]['src_kps'], float); Y = np.asarray(rows[0]['trg_kps'], float)
    sw = slicewise(np.c_[X, z], Y, sorted(set(z.tolist())), kind='similarity', dz=2)
    print('regions', n, 'worst |diff|', {k: float(v) for k, v in worst.items()}, '| slicewise planes', len(sw['planes']),
          'global rms', round(sw['global_']['rms_um'], 2))
    assert n > 0 and all(v < 1e-6 for v in worst.values()), worst
    print('REGISTER PARITY PASS')


if __name__ == '__main__':
    main()
