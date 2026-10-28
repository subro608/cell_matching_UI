#!/usr/bin/env python3
"""Run one session job inside the container: the job spec (request.json written by the web API) -> ivex analyze / rematch.

  SESSION/raw/{iv,ex}.tif   uploads (downloaded by the poller)      SESSION/analysis/   analysis bundle (cli.analyze)
  SESSION/web/              web bundle the poller uploads           SESSION/job.json    spec of the job being run
Uploads are untrusted: only tifffile parses them, after shape / size checks; nothing uploaded is executed.
usage: python run_job.py SESSION_DIR   (spec = SESSION_DIR/job.json)"""
import json, os, subprocess, sys
from pathlib import Path

LIMITS = dict(iv_voxels=400e6, ex_pixels=120e6, ex_planes=64, iv_planes=200)   # the viewer's < 500 MB per image file


IO_PY = os.environ.get('IVEX_PY_IO', '/scratch/sd5963/venvs/ivex_io_env/bin/python')      # ivexmatch.readers (ND2, zip of planes, TIFF, ...)
CORE = Path(__file__).resolve().parents[1] / 'core'


def raw_file(raw, side):
    """The uploaded file for one side, whatever its format (raw/iv.nd2, raw/ex.zip, raw/iv.tif, ...)."""
    found = [f for f in raw.glob(f'{side}.*') if not f.name.endswith('_zcyx.npy')]
    assert found, f'no {side.upper()} upload found'; return found[0]


def normalize(src, dst, position=0):
    """Any supported upload -> (Z, C, Y, X) .npy through ivexmatch.readers (in its own environment). Returns the shape."""
    r = subprocess.run([IO_PY, '-m', 'ivexmatch.readers', 'convert', str(src), '--out', str(dst), '--position', str(int(position or 0))],
                       env=dict(os.environ, PYTHONPATH=str(CORE)), capture_output=True, text=True)
    if r.returncode: raise SystemExit(f'could not read {src.name}: {(r.stderr.strip().splitlines() or ["?"])[-1]}')
    return json.loads(r.stdout.strip().splitlines()[-1])['shape']


def check_inputs(raw, spec):
    shapes = {side: normalize(raw_file(raw, side), raw / f'{side}_zcyx.npy', spec[side].get('position', 0)) for side in ('iv', 'ex')}
    iv, ex = shapes['iv'], shapes['ex']
    assert iv[0] * iv[2] * iv[3] <= LIMITS['iv_voxels'], f'IV stack too large ({iv[0] * iv[2] * iv[3] / 1e6:.0f} M voxels; limit {LIMITS["iv_voxels"] / 1e6:.0f} M)'
    assert iv[0] <= LIMITS['iv_planes'] and ex[0] <= LIMITS['ex_planes'], f'too many planes: IV {iv[0]}, EX {ex[0]}'
    assert ex[2] * ex[3] <= LIMITS['ex_pixels'], f'EX image too large ({ex[2]} x {ex[3]} px; limit {LIMITS["ex_pixels"] / 1e6:.0f} M px)'
    for side, sh in shapes.items():
        for k in ('channel', 'dapi_channel'):
            c = spec[side].get(k)
            assert c is None or c < sh[1], f'{side.upper()} has {sh[1]} channel(s); channel {c} does not exist'
    return shapes


def main(session):
    session = Path(session); spec = json.loads((session / 'job.json').read_text()); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core'))
    from ivexmatch import cli
    a, w = str(session / 'analysis'), str(session / 'web')
    if spec['kind'] == 'analyze':
        print(json.dumps(check_inputs(session / 'raw', spec)), flush=True)
        iv, ex = spec['iv'], spec['ex']
        argv = ['analyze', '--iv', str(session / 'raw/iv_zcyx.npy'), '--iv-um', str(iv['um_xy']), '--iv-channel', str(iv.get('channel', 0)),
                '--ex', str(session / 'raw/ex_zcyx.npy'), '--ex-um', str(ex['um_xy']), '--ex-channel', str(ex.get('channel', 0)),
                '--models', ','.join(spec.get('models') or ['cyto2_ft', 'cyto3_ft', 'cpsam']), '--match-model', spec.get('match_model', 'cyto3_ft'),
                '--out', a, '--web', w, '--title', spec.get('title') or 'ivex session']
        if iv.get('um_z'): argv += ['--iv-z-um', str(iv['um_z'])]
        if ex.get('dapi_channel') is not None: argv += ['--ex-dapi-channel', str(ex['dapi_channel'])]
        if ex.get('z_range'): argv += ['--ex-z', ex['z_range']]
        if iv.get('diameter_um'): argv += ['--iv-diameter-um', str(iv['diameter_um'])]
        if ex.get('diameter_um'): argv += ['--ex-diameter-um', str(ex['diameter_um'])]
    elif spec['kind'] == 'rematch':
        argv = ['rematch', '--out', a, '--web', w]
        if spec.get('affine'): argv += ['--affine', json.dumps(spec['affine'])]
        elif spec.get('theta') is not None: argv += ['--flip', str(int(bool(spec.get('flip')))), '--theta', str(spec['theta'])]
        if spec.get('match_model'): argv += ['--match-model', spec['match_model']]
    else:
        raise SystemExit(f'unknown job kind {spec["kind"]}')
    print('ivex ' + ' '.join(argv), flush=True); cli.main(argv)


if __name__ == '__main__':
    main(sys.argv[1])
