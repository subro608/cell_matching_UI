"""Parity gate (plan > Verification > PMAC parity): ivex PMAC reproduces the GFP+DAPI checkpoint's jy306 rows 4-5 validation
replay, 206/324 right cells (GCP A100; run_pmac.py allowed |drift| <= 3 on other GPUs).

Reference rows come from the e200 harness (PMAC_GD_ARM=gfpdapi_synth, data.load_subject('jy306va')). Two passes:
  A. reference inputs (incl. EX crops cut on the GPU by data.GpuCrops) through the ivex model -> model parity;
  B. ivex build_record on the raw arrays (B volume, composite EX, CPU PIL crops) -> end-to-end parity, plus crop diffs vs A.
Grading = run_gd.evaluate: correct when the combined top-1 is one of the landmark's same-cell candidates.
Writes logs/parity_jy306va_gfpdapi.json; exit 1 on failure."""
import json, os, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
E200 = Path('/scratch/sd5963/cell_registration_2026/microscopy_ssl/experiments/pmac_gfp_dapi_e200_2026-10-25')
CKPT = Path('/scratch/sd5963/cell_registration_2026/microscopy_ssl/experiments/jy369_pmac_gfpdapi_2026-10-25/ckpt/gfpdapi_synth_best_val_loss.pt')
EXPECTED, TOL = 206, 3

os.environ['PMAC_GD_ARM'] = 'gfpdapi_synth'; sys.path.insert(0, str(E200)); os.chdir(E200)
import data as ref_data                                                      # reference (top-level module of the e200 run)
sys.path.insert(0, str(REPO / 'core'))
import torch
from ivexmatch.pmac import PMAC, build_record

t0 = time.time()
scale = json.loads((E200 / 'scale.json').read_text())['crop_scale']
m = PMAC(CKPT, crop_scale=scale)
fields = ref_data.Fields(); rows = ref_data.load_subject('jy306va')
report = dict(gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu', expected=EXPECTED, regions={})
count = dict(A=0, B=0); total = 0; crop_diff = dict(q=0., ex=0.); ex_equal = []


def graded(terms, same):
    pred = terms['combined'].argmax(1).cpu().numpy(); return int(sum(int(p) in c for p, c in zip(pred, same)))


for region, a_ref in rows:
    f_ref = fields.get(a_ref); same = a_ref['_grade']['same_cell']; n = len(a_ref['q_xy']); total += n
    cA = graded(m.full_terms(a_ref, f_ref), same); count['A'] += cA
    meta = ref_data.B_GEOMETRY[ref_data.mouse_of(a_ref['_subject'])]; volume = np.load(meta['volume'], mmap_mode='r')
    a, f = build_record(volume, f_ref['iv_full_xy'], a_ref['q_z'], f_ref['raw_ex'], a_ref['ex_xy'], scale, iv_box=meta['detector_box'])
    qi = np.arange(n); ci = np.arange(len(a_ref['ex_xy']))
    crop_diff['q'] = max(crop_diff['q'], float(np.abs(np.asarray(a['q_local'][qi], np.int16) - np.asarray(a_ref['q_local'][qi], np.int16)).max()))
    ref_view, my_view = a_ref['ex_local'][ci], a['ex_local'][ci]                   # GPU crops (reference) vs CPU PIL crops, 256 at a time
    for c0 in range(0, len(ci), 256):
        ref_ex = ref_view[c0:c0 + 256].cpu().numpy().astype(np.int16); mine = my_view[c0:c0 + 256].astype(np.int16)
        crop_diff['ex'] = max(crop_diff['ex'], float(np.abs(mine - ref_ex).max())); ex_equal.append(float((mine == ref_ex).mean()))
    cB = graded(m.full_terms(a, f), same); count['B'] += cB
    report['regions'][region] = dict(n=n, gallery=len(ci), correct_reference_inputs=cA, correct_ivex_inputs=cB)
    print(region, n, 'A', cA, 'B', cB, flush=True)
report.update(total=total, correct_reference_inputs=count['A'], correct_ivex_inputs=count['B'], max_crop_abs_diff=crop_diff,
              ex_crop_pixels_equal=float(np.mean(ex_equal)), seconds=round(time.time() - t0),
              passed=bool(total == 324 and abs(count['A'] - EXPECTED) <= TOL and abs(count['B'] - EXPECTED) <= TOL))
(REPO / 'logs').mkdir(exist_ok=True); (REPO / 'logs/parity_jy306va_gfpdapi.json').write_text(json.dumps(report, indent=1))
print('jy306va', count, '/', total, '(expected', EXPECTED, ') crop diffs', crop_diff, 'PASS' if report['passed'] else 'FAIL', flush=True)
sys.exit(0 if report['passed'] else 1)
