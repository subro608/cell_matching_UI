"""ivex analyze: one dataset -> session bundle (the job the HPC worker / any GPU runner executes).

  1. read IV (Z, H, W[, C]) and EX (Z, H, W[, C]) TIFFs, pick channels, EX max projection over the depth band
  2. masks: every model x side, each in its own Cellpose environment (subprocess: python -m ivexmatch.segment)
  3. working grids: IV B-normalized on IV_UM, EX stretched on EX_UM (preprocess.py); match-model centroids mapped onto them
  4. PMAC blind pose (pose.py) -> rotate / mirror IV upright -> PMAC scoring of every IV cell-plane against every EX cell
  5. bundle: meta.json, img/*.npy, masks/<side>/<model>.npz, pose/pose.json, pmac/<run>/topk.json (see bundle.py)
  6. --web DIR: the viewer's web bundle (webexport.export_session)
ivex rematch: PMAC again at an orientation the user set (or with another mask set), reusing work/ and masks/; adds pmac/run<k>.
Runs under marco_env inside w2v2_cuda_cu128.sif; Cellpose interpreters come from IVEX_PY_CP3 / IVEX_PY_CP4.

usage: python -m ivexmatch.cli analyze --iv iv.tif --iv-um 0.684 --iv-z-um 3 --iv-channel 0 --ex ex.tif --ex-um 0.324 --ex-channel 1 \
           [--ex-dapi-channel 2] [--ex-z 0:7] [--models cyto2_ft,cyto3_ft,cpsam] [--match-model cyto3_ft] --out BUNDLE [--web WEB]
       python -m ivexmatch.cli rematch --out BUNDLE [--affine '[[a,b,tx],[c,d,ty]]' | --flip 0 --theta 177] [--match-model cpsam] [--web WEB]
       python -m ivexmatch.cli export-web --out BUNDLE --web WEB"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

from . import preprocess as pp, bundle, webexport
from .segment import MODELS

CORE = Path(__file__).resolve().parents[1]
PY = dict(cp3=os.environ.get('IVEX_PY_CP3', '/scratch/sd5963/venvs/cellpose_v3_env/bin/python'),
          cp4=os.environ.get('IVEX_PY_CP4', '/scratch/sd5963/venvs/cellpose_v42_env/bin/python'))
CHECKPOINTS = dict(  # v1 deployment checkpoints (state_dict + arm; crop scale of their training run)
    gray=dict(path='/scratch/sd5963/cell_registration_2026/microscopy_ssl/figures_pmac_synth2_2026-10-20/pmac/spherical_similarity_analysis/checkpoints/original_best.pt',
              scale='/scratch/sd5963/cell_registration_2026/microscopy_ssl/experiments/pmac_synth2_B_train100_loso_2026-10-22/folds/JY195/scale.json'),
    composite=dict(path='/scratch/sd5963/cell_registration_2026/microscopy_ssl/experiments/jy369_pmac_gfpdapi_2026-10-25/ckpt/gfpdapi_synth_best_val_loss.pt',
                   scale='/scratch/sd5963/cell_registration_2026/microscopy_ssl/experiments/pmac_gfp_dapi_e200_2026-10-25/scale.json'))


def read_stack(path, channel):
    """TIFF (Z, H, W[, C]) / (Z, C, H, W), or the worker's normalized (Z, C, Y, X) .npy -> (Z, H, W) of one channel."""
    if str(path).endswith('.npy'): return np.load(path, mmap_mode='r')[:, channel]
    import tifffile
    a = np.asarray(tifffile.imread(path)); a = a[None] if a.ndim == 2 else a
    if a.ndim == 4: a = a[..., channel] if a.shape[-1] <= 8 else a[:, channel]   # (Z, H, W, C) or (Z, C, H, W)
    assert a.ndim == 3, f'{path}: expected (Z, H, W[, C]), got {a.shape}'; return a


def segment_all(out, iv, ex_mip, a, log):
    work = out / 'work'; work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(CORE))
    for name in a.models.split(','):
        for side, src, um, d in (('iv', 'iv_native.npy', a.iv_um, a.iv_diameter_um), ('ex', 'ex_mip.npy', a.ex_um, a.ex_diameter_um)):
            dst = out / 'masks' / side / f'{name}.npz'; dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists(): log(f'masks {side}/{name}: cached'); continue
            py = PY['cp3'] if MODELS[name]['version'] == 3 else PY['cp4']
            cmd = [py, '-u', '-m', 'ivexmatch.segment', '--model', name, '--side', side, '--input', str(work / src), '--um', str(um), '--diameter_um', str(d), '--out', str(dst)]
            if side == 'iv' and a.iv_z_um: cmd += ['--um_z', str(a.iv_z_um)]
            log(' '.join(cmd)); subprocess.run(cmd, check=True, env=env)


def analyze(a, log=print):
    import torch
    from .pose import blind_pose
    t0 = time.time(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'work').mkdir(exist_ok=True); (out / 'work/args.json').write_text(json.dumps(vars(a), indent=1))
    iv = read_stack(a.iv, a.iv_channel); ex = read_stack(a.ex, a.ex_channel)
    z0, z1 = (int(v) for v in a.ex_z.split(':')) if a.ex_z else (0, len(ex)); ex = ex[z0:z1]; ex_mip = ex.max(0)
    dapi = read_stack(a.ex, a.ex_dapi_channel)[z0:z1] if a.ex_dapi_channel is not None else None
    log(f'IV {iv.shape} @ {a.iv_um} um/px, EX band {z0}:{z1} {ex.shape} @ {a.ex_um} um/px, DAPI {dapi is not None}')
    work = out / 'work'; work.mkdir(parents=True, exist_ok=True); np.save(work / 'iv_native.npy', iv); np.save(work / 'ex_mip.npy', ex_mip)
    np.save(work / 'ex_band.npy', ex)
    if dapi is not None: np.save(work / 'ex_dapi_band.npy', dapi); np.save(work / 'ex_dapi_mip.npy', dapi.max(0))
    progress(out, 'masks', 'Cell masks (3 Cellpose models)')
    if not a.skip_seg: segment_all(out, iv, ex_mip, a, log)
    progress(out, 'pose', 'PMAC blind pose')
    ctx = context(out, a.match_model, a.checkpoint, a.scale, log)
    pose = blind_pose(ctx['pmac'], ctx['iv_g'], ctx['iv_xyz'], ctx['ex_g'], ctx['ex_xy'], log=log)
    hyps = [dict(pose['pose'], kind='best')] if pose['pose'] else []
    ru = pose['runner_up']
    if pose['pose'] and ru and ru['score'] >= 0.7 * pose['best']['score']:
        hyps.append(dict(flip=ru['flip'], theta=ru['theta'], scale=ru['scale'], t=ru['t'], kind='runner_up', runner_up=True))
    if not hyps: hyps = [dict(flip=False, theta=0.0, unposed=True, kind='none')]
    progress(out, 'match', 'PMAC matching')
    runs = [dict(name=f'run{i}', pose=h, match_model=a.match_model, result=match(ctx, h, log, f'run{i}')) for i, h in enumerate(hyps)]
    pmac = ctx['pmac']
    bundle.write(out, args=vars(a), iv=dict(grid=ctx['iv_g'], meta=ctx['iv_meta'], cells=ctx['iv_cells'], xyz=ctx['iv_xyz']),
                 ex=dict(grid=ctx['ex_g'], meta=ctx['ex_meta'], cells=ctx['ex_cells'], xy=ctx['ex_xy']), pose=pose, runs=runs,
                 pmac=dict(arm=pmac.arm, crop_scale=pmac.crop_scale.tolist(), checkpoint=ctx['checkpoint']),
                 seconds=round(time.time() - t0, 1), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
    log(f'bundle written to {out} ({time.time() - t0:.0f}s)')
    if a.web: progress(out, 'export', 'Preparing the viewer files'); webexport.export_session(out, a.web, title=a.title, log=log)
    progress(out, 'done', 'Done')


def progress(out, step, text):
    """work/progress.json: the step the worker relays to the session page."""
    (Path(out) / 'work').mkdir(parents=True, exist_ok=True); (Path(out) / 'work/progress.json').write_text(json.dumps(dict(step=step, text=text, t=time.time())))


def context(out, match_model, checkpoint=None, scale=None, log=print):
    """Working grids, match-model cells and PMAC for a bundle directory (shared by analyze and rematch)."""
    from .pmac import PMAC
    out = Path(out); w = out / 'work'; iv = np.load(w / 'iv_native.npy'); ex = np.load(w / 'ex_band.npy')
    dapi = np.load(w / 'ex_dapi_band.npy') if (w / 'ex_dapi_band.npy').exists() else None
    iv_g, iv_meta = pp.iv_volume(iv, json.loads((out / 'work/args.json').read_text())['iv_um']); ex_g, ex_meta = pp.ex_image(ex, json.loads((out / 'work/args.json').read_text())['ex_um'], dapi)
    iv_cells = np.load(out / 'masks/iv' / f'{match_model}.npz')['cells']; ex_cells = np.load(out / 'masks/ex' / f'{match_model}.npz')['cells']   # z, id, x, y, area (native px)
    assert len(iv_cells) and len(ex_cells), f'{match_model}: no cells found (IV {len(iv_cells)}, EX {len(ex_cells)})'
    iv_xyz = np.c_[pp.to_grid(iv_cells[:, 2:4], iv_meta['native_hw'], iv_meta['grid_hw']), iv_cells[:, 0]]
    ex_xy = pp.to_grid(ex_cells[:, 2:4], ex_meta['native_hw'], ex_meta['grid_hw'])
    ck = CHECKPOINTS['composite' if dapi is not None else 'gray']; path = checkpoint or ck['path']
    pmac = PMAC(path, crop_scale=json.loads(Path(scale or ck['scale']).read_text())['crop_scale'])
    log(f'{len(iv_xyz)} IV cell-planes, {len(ex_xy)} EX cells ({match_model}); PMAC {pmac.arm} on {pmac.device}')
    return dict(pmac=pmac, iv_g=iv_g, iv_meta=iv_meta, ex_g=ex_g, ex_meta=ex_meta, iv_cells=iv_cells, ex_cells=ex_cells, iv_xyz=iv_xyz, ex_xy=ex_xy, checkpoint=str(path))


def match(ctx, h, log, name):
    from .pmac import build_record
    from .pose import apply_pose
    planes, cells = apply_pose(ctx['iv_g'], ctx['iv_xyz'], h)
    rec, f = build_record(planes, cells[:, :2], cells[:, 2].astype(int), ctx['ex_g'], ctx['ex_xy'], ctx['pmac'].crop_scale)
    res = ctx['pmac'].score(rec, f)
    log(f'PMAC {name} (flip={h["flip"]}, theta={h["theta"]:.1f}): mutual {int(res["mutual"].sum())}, P(top1) >= 0.5: {int((res["p_top1"] >= .5).sum())}')
    return res


def rematch(a, log=print):
    """One more PMAC run at the user's orientation (native affine from the viewer, or flip + theta) and/or with another mask set."""
    from .pose import pose_from_native_affine
    out = Path(a.out); meta = json.loads((out / 'meta.json').read_text()); mm = a.match_model or meta['args']['match_model']
    progress(out, 'match', 'PMAC matching at your orientation'); ctx = context(out, mm, meta['pmac']['checkpoint'], a.scale, log)
    if a.affine:
        A = json.loads(a.affine); h = dict(pose_from_native_affine(A, ctx['iv_meta']['native_hw'], ctx['iv_meta']['grid_hw'], ctx['ex_meta']['native_hw'], ctx['ex_meta']['grid_hw']), iv_to_ex_native=A)
    elif a.theta is not None: h = dict(flip=bool(a.flip), theta=float(a.theta), kind='manual')
    else: h = json.loads((out / 'pmac' / meta['runs'][0] / 'topk.json').read_text())['pose']          # same pose, other mask set
    name = f'run{len(meta["runs"])}'; res = match(ctx, h, log, name)
    bundle.write_run(out, name, h, mm, ctx['iv_cells'], ctx['ex_cells'], res); meta['runs'] = [name] + meta['runs']          # newest first = default
    (out / 'meta.json').write_text(json.dumps(meta, indent=1))
    if a.web: progress(out, 'export', 'Preparing the viewer files'); webexport.export_session(out, a.web, runs_only=True, log=log)
    progress(out, 'done', 'Done'); log(f'{name}: {h}')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='ivex'); sub = ap.add_subparsers(dest='cmd', required=True); s = sub.add_parser('analyze')
    s.add_argument('--iv', required=True); s.add_argument('--iv-um', type=float, required=True); s.add_argument('--iv-z-um', type=float); s.add_argument('--iv-channel', type=int, default=0)
    s.add_argument('--ex', required=True); s.add_argument('--ex-um', type=float, required=True); s.add_argument('--ex-channel', type=int, default=0)
    s.add_argument('--ex-dapi-channel', type=int); s.add_argument('--ex-z', default='')
    s.add_argument('--iv-diameter-um', type=float, default=11.); s.add_argument('--ex-diameter-um', type=float, default=11.)
    s.add_argument('--models', default='cyto2_ft,cyto3_ft,cpsam'); s.add_argument('--match-model', default='cyto3_ft')
    s.add_argument('--checkpoint'); s.add_argument('--scale'); s.add_argument('--skip-seg', action='store_true'); s.add_argument('--out', required=True)
    s.add_argument('--web'); s.add_argument('--title')
    r = sub.add_parser('rematch'); r.add_argument('--out', required=True); r.add_argument('--affine'); r.add_argument('--flip', type=int, default=0)
    r.add_argument('--theta', type=float); r.add_argument('--match-model'); r.add_argument('--scale'); r.add_argument('--web')
    e = sub.add_parser('export-web'); e.add_argument('--out', required=True); e.add_argument('--web', required=True); e.add_argument('--title')
    a = ap.parse_args(argv)
    if a.cmd == 'analyze': analyze(a)
    elif a.cmd == 'rematch': rematch(a)
    else: webexport.export_session(a.out, a.web, title=a.title)


if __name__ == '__main__':
    main()
