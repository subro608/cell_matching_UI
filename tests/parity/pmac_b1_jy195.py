"""Parity gate (plan > Verification > PMAC parity): ivex PMAC reproduces JY195 B1 = 293/439 with the B100 original_best.pt.

The B100 harness (data.load_subject + exact_landmarks.make_rows, PMAC_FOLD=JY195) supplies the reference rows. Per region:
  1. preprocessing: ivexmatch.pmac.build_record on the raw arrays (B volume, EX png, landmark xy, detector box) must give the
     reference crops (uint8, exactly equal) and field geometry (equal);
  2. model: ivex full_terms on OUR record vs the saved reference score matrices of the original run
     (B_train100_loso/B3/source_scores/JY195/exact_landmarks/*.npz): max |diff| per term, top-1 agreement;
  3. metric: combined top-1 correct, summed (reference 293/439 on a GCP A100-40GB; GPU-type drift <= 3 allowed).
Writes logs/parity_b1_jy195.json; exit 1 on failure."""
import json, os, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
FIG = Path('/scratch/sd5963/cell_registration_2026/microscopy_ssl/figures_pmac_synth2_2026-10-20/pmac')
B100 = Path('/scratch/sd5963/cell_registration_2026/microscopy_ssl/experiments/pmac_synth2_B_train100_loso_2026-10-22')
CKPT = FIG / 'spherical_similarity_analysis/checkpoints/original_best.pt'
SAVED = FIG / 'B_train100_loso/B3/source_scores/JY195/exact_landmarks'
EXPECTED, TOL = 293, 3

os.environ['PMAC_FOLD'] = 'JY195'; sys.path.insert(0, str(B100)); os.chdir(B100)
import data as ref_data, exact_landmarks                                    # reference (top-level modules of the B100 run)
sys.path.insert(0, str(REPO / 'core'))
import torch
from ivexmatch.pmac import PMAC, build_record

t0 = time.time()
scale = json.loads((B100 / 'folds/JY195/scale.json').read_text())['crop_scale']
m = PMAC(CKPT, crop_scale=scale)
fields = ref_data.Fields(allow_test=True)
rows = exact_landmarks.make_rows(ref_data.load_subject('JY195', allow_test=True), fields)
meta = ref_data.B_GEOMETRY['JY195']; volume = np.load(meta['volume'], mmap_mode='r')
report = dict(gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu', regions={}, expected=EXPECTED)
correct = total = 0; prep_ok = True; worst = {}
for name, b, ef in rows:
    n = int(b['n_gt']); ids = np.arange(n)
    a, f = build_record(volume, ef['iv_full_xy'][:n], b['q_z'], ef['raw_ex'], b['ex_xy'], scale, iv_box=meta['detector_box'])
    checks = dict(q_local=np.array_equal(a['q_local'][ids], b['q_local']), q_global=np.array_equal(a['q_global'][ids], b['q_global']),
                  ex_local=np.array_equal(a['ex_local'][ids], b['ex_local']), ex_global=np.array_equal(a['ex_global'][ids], b['ex_global']),
                  q_xy=np.array_equal(f['q_xy'], ef['q_xy']), ex_xy=np.array_equal(f['ex_xy'], ef['ex_xy']),
                  iv_geom=f['iv_geom'] == ef['iv_geom'], ex_geom=f['ex_geom'] == ef['ex_geom'],
                  planes=all(np.array_equal(f['iv'][z], ef['iv'][z]) for z in f['iv']) and np.array_equal(f['ex'], ef['ex']))
    prep_ok &= all(checks.values())
    terms = {k: v.float().cpu().numpy() for k, v in m.full_terms(a, f).items()}
    saved = np.load(SAVED / f'{name}.npz'); assert np.array_equal(saved['truth'], ids)
    diffs = {k: float(np.abs(terms[k] - saved[k]).max()) for k in terms}
    agree = {k: float((terms[k].argmax(1) == saved[k].argmax(1)).mean()) for k in terms}
    c = int((terms['combined'].argmax(1) == ids).sum()); correct += c; total += n
    for k, v in diffs.items(): worst[k] = max(worst.get(k, 0.), v)
    report['regions'][name] = dict(n=n, correct=c, reference=int((saved['combined'].argmax(1) == ids).sum()), prep=checks, max_abs_diff=diffs, top1_agree=agree)
    print(name, n, 'correct', c, 'ref', report['regions'][name]['reference'], 'prep', all(checks.values()), 'max|d| comb', round(diffs['combined'], 5), flush=True)
report.update(correct=correct, total=total, preprocessing_identical=bool(prep_ok), worst_max_abs_diff=worst, seconds=round(time.time() - t0),
              passed=bool(prep_ok and abs(correct - EXPECTED) <= TOL and total == 439))
(REPO / 'logs').mkdir(exist_ok=True); (REPO / 'logs/parity_b1_jy195.json').write_text(json.dumps(report, indent=1))
print('B1 JY195', correct, '/', total, '(expected', EXPECTED, ') preprocessing identical:', prep_ok, 'PASS' if report['passed'] else 'FAIL', flush=True)
sys.exit(0 if report['passed'] else 1)
