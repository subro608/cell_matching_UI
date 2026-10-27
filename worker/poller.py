#!/usr/bin/env python3
"""HPC pull worker: the only link between the web app (Vercel + Vercel Blob) and the GPUs. Outbound HTTPS only.

Every POLL_S seconds:
  1. read queue.json from Blob (CDN-cached: a cache hit is not a billed operation; the API overwrites it on every submit);
  2. if no GPU job is active, claim the oldest unhandled job: download its inputs to scratch, delete the uploads from Blob, write
     SESSION/job.json and `sbatch analyze.sbatch SESSION`;
  3. follow the active job (sacct + SESSION/analysis/work/progress.json); relay step changes to sessions/<sid>/state.json;
     on success upload the web bundle (only the changed files after a rematch), on failure the last error line;
  4. every HEARTBEAT_S write worker/heartbeat.json (the UI shows "worker offline" when it is older than OFFLINE_S);
  5. once a day delete expired sessions' files from Blob (scratch keeps the copy; a "restore" job uploads them again).
The sbatch wrapper resubmits this poller before the cpu_short 6 h limit (scrontab is disabled on this cluster).

Scratch layout: ROOT/<sid>/{raw,analysis,web,job.json}, ROOT/_worker/state.json (claimed / finished jobs).
usage: python poller.py [--once] [--max-seconds N]"""
from __future__ import annotations
import argparse, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('IVEX_SESSIONS', '/scratch/sd5963/ivex-sessions')); WSTATE = ROOT / '_worker/state.json'
EXAMPLES = Path(os.environ.get('IVEX_EXAMPLES', '/scratch/sd5963/ivex-match-data/examples'))       # <name>/{iv,ex}.ome.tif
NODE = os.environ.get('IVEX_NODE', '/scratch/sd5963/conda_envs/neuro_env_311/bin/node')
TOKEN_FILE = Path(os.environ.get('IVEX_BLOB_TOKEN_FILE', '/scratch/sd5963/.secrets/vercel_blob_ivex_rw'))
POLL_S, HEARTBEAT_S, RETAIN_DAYS = 60, 3 * 3600, 7
SID = re.compile(r'^[A-Za-z0-9_-]{8,64}$')
ENV = dict(os.environ, BLOB_READ_WRITE_TOKEN=TOKEN_FILE.read_text().strip())


def log(msg): print(time.strftime('%Y-%m-%d %H:%M:%S'), msg, flush=True)


def blob(*args, out=None, check=True):
    r = subprocess.run([NODE, str(HERE / 'blob.mjs'), *args], env=ENV, capture_output=True, text=True, timeout=3600)
    if r.returncode == 3: return None
    if check and r.returncode: raise RuntimeError(f'blob {args[0]} {args[1] if len(args) > 1 else ""}: {r.stderr.strip()[-300:]}')
    return r.stdout


def blob_json(path, fresh=False):
    s = blob('get', path, *(['--fresh'] if fresh else []))
    return None if s is None else json.loads(s)


def put_json(path, obj, max_age=60):
    tmp = ROOT / '_worker' / 'put.json'; tmp.write_text(json.dumps(obj)); blob('put', path, str(tmp), '--type', 'application/json', '--max-age', str(max_age))


def prune_queue(job_id):
    """Drop a claimed job from queue.json (ETag-guarded against a concurrent submit), so queue positions stay true."""
    tmp = ROOT / '_worker' / 'queue.json'
    for _ in range(5):
        r = subprocess.run([NODE, str(HERE / 'blob.mjs'), 'get', 'queue.json', str(tmp), '--fresh'], env=ENV, capture_output=True, text=True)
        if r.returncode: return
        etag = json.loads(r.stderr.strip().splitlines()[-1])['etag']; q = json.loads(tmp.read_text())
        q['jobs'] = [j for j in q['jobs'] if j['id'] != job_id]; tmp.write_text(json.dumps(q))
        w = subprocess.run([NODE, str(HERE / 'blob.mjs'), 'put', 'queue.json', str(tmp), '--type', 'application/json', '--max-age', '3600', '--if-match', etag],
                           env=ENV, capture_output=True, text=True)
        if w.returncode == 0: return
        time.sleep(2)
    log(f'could not prune {job_id} from the queue')


def load_state():
    return json.loads(WSTATE.read_text()) if WSTATE.exists() else dict(handled=[], active=None, last_heartbeat=0, last_expiry=0, done={})


def save_state(s): tmp = WSTATE.with_suffix('.tmp'); tmp.write_text(json.dumps(s, indent=1)); tmp.replace(WSTATE)


def set_status(sid, **kw):
    put_json(f'sessions/{sid}/state.json', dict(kw, sid=sid, updated=time.time())); log(f'{sid}: {kw.get("status")} {kw.get("text", "")}')


def slurm_state(job):
    r = subprocess.run(['sacct', '-n', '-X', '-P', '-j', str(job), '-o', 'State'], capture_output=True, text=True)
    s = r.stdout.strip().split('\n')[0].split()[0] if r.stdout.strip() else ''
    if not s:
        q = subprocess.run(['squeue', '-h', '-j', str(job), '-o', '%T'], capture_output=True, text=True).stdout.strip()
        s = q or 'UNKNOWN'
    return s


def claim(job, st):
    sid = job['sid']; assert SID.match(sid), f'bad session id {sid!r}'; d = ROOT / sid; (d / 'raw').mkdir(parents=True, exist_ok=True)
    if job['kind'] == 'analyze':
        req = blob_json(f'sessions/{sid}/request.json', fresh=True); assert req, 'request.json missing'
        spec = dict(req['spec'], sid=sid, kind='analyze')
        if req.get('example'):                                                                     # inputs from the worker's own copy
            src = EXAMPLES / req['example']; assert re.match(r'^[a-z0-9_-]+$', req['example']) and src.is_dir(), f'unknown example {req["example"]}'
            for side in ('iv', 'ex'): shutil.copyfile(src / f'{side}.ome.tif', d / 'raw' / f'{side}.tif')
        else:
            for side in ('iv', 'ex'):
                blob('get', f'sessions/{sid}/raw/{side}.tif', str(d / 'raw' / f'{side}.tif'), '--fresh')
                assert (d / 'raw' / f'{side}.tif').stat().st_size > 0, f'{side}.tif is empty'
            blob('del', f'sessions/{sid}/raw/iv.tif', f'sessions/{sid}/raw/ex.tif', check=False)     # free; scratch holds the copy now
    elif job['kind'] in ('rematch', 'restore'):
        assert (d / 'analysis/meta.json').exists(), 'no analysis on the worker for this session'
        spec = dict(job.get('spec') or {}, sid=sid, kind=job['kind'])
    else:
        raise ValueError(f'unknown job kind {job["kind"]}')
    if job['kind'] == 'restore':
        upload_web(sid, d / 'web', full=True); set_status(sid, status='done', text='Restored', kind='restore'); return None
    (d / 'job.json').write_text(json.dumps(spec, indent=1)); (d / 'analysis/work').mkdir(parents=True, exist_ok=True)
    (d / 'analysis/work/progress.json').unlink(missing_ok=True)
    r = subprocess.run(['sbatch', '--parsable', '--job-name', f'ivex_{job["kind"]}', str(HERE / 'analyze.sbatch'), str(d)], capture_output=True, text=True, check=True)
    slurm = r.stdout.strip().split(';')[0]
    set_status(sid, status='running', step='queued_hpc', text='Waiting for a GPU on NYU HPC', slurm=slurm, kind=job['kind'])
    return dict(id=job['id'], sid=sid, kind=job['kind'], slurm=slurm, step='queued_hpc', started=time.time())


def upload_web(sid, web, full):
    man = json.loads((web / 'manifest.json').read_text()); files = man['files']
    if not full: files = [f for f in files if f.startswith('matches_')] + ['manifest.json']       # rematch: images / masks unchanged
    for f in [f for f in files if f != 'manifest.json'] + ['manifest.json']:                       # manifest last: viewer sees a complete bundle
        blob('put', f'sessions/{sid}/web/{f}', str(web / f), '--max-age', str(30 * 86400 if f.endswith('.tif') else 300))
    log(f'{sid}: uploaded {len(files)} files')


def follow(a, st):
    d = ROOT / a['sid']; s = slurm_state(a['slurm']); prog = d / 'analysis/work/progress.json'
    p = json.loads(prog.read_text()) if prog.exists() else None
    if s in ('RUNNING', 'COMPLETING') and p and p['step'] != a['step'] and p['step'] not in ('done',):
        a['step'] = p['step']; set_status(a['sid'], status='running', step=p['step'], text=p['text'], slurm=a['slurm'], kind=a['kind'])
    elif s == 'RUNNING' and a['step'] == 'queued_hpc':
        a['step'] = 'started'; set_status(a['sid'], status='running', step='started', text='Running on NYU HPC', slurm=a['slurm'], kind=a['kind'])
    if s.startswith('COMPLETED'):
        upload_web(a['sid'], d / 'web', full=a['kind'] == 'analyze'); set_status(a['sid'], status='done', text='Done', kind=a['kind'], slurm=a['slurm'])
        st['done'][a['sid']] = time.time(); return None
    if s.split()[0] in ('FAILED', 'TIMEOUT', 'CANCELLED', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE'):
        set_status(a['sid'], status='failed', text=f'The job ended with {s}', error=last_error(a['slurm']), kind=a['kind'], slurm=a['slurm']); return None
    return a


def last_error(slurm):
    f = next((HERE.parent / 'logs/jobs').glob(f'*_{slurm}.out'), None)
    if not f: return None
    lines = [l for l in f.read_text(errors='replace').splitlines() if l.strip()]
    err = next((l for l in reversed(lines) if re.search(r'(Error|Exception|assert)', l)), lines[-1] if lines else '')
    return re.sub(r'/scratch/\S+/', '…/', err)[-400:]                                        # no scratch paths in a public page


def expire(st):
    now = time.time()
    for sid, t in list(st['done'].items()):
        if now - t < RETAIN_DAYS * 86400: continue
        web = ROOT / sid / 'web'
        if (web / 'manifest.json').exists():
            files = json.loads((web / 'manifest.json').read_text())['files']; blob('del', *[f'sessions/{sid}/web/{f}' for f in files], check=False)
        set_status(sid, status='expired', text=f'Files removed from the cloud after {RETAIN_DAYS} days; ask to restore them'); del st['done'][sid]


def tick(st, t0):
    q = blob_json('queue.json') or dict(jobs=[])
    if st['active']:
        try: st['active'] = follow(st['active'], st)
        except Exception as e: log(f'follow failed: {e}')
    if not st['active']:
        for job in q['jobs']:
            if job['id'] in st['handled']: continue
            st['handled'].append(job['id']); save_state(st)
            try:
                st['active'] = claim(job, st)
            except Exception as e:
                log(f'claim {job["id"]} failed: {e}'); set_status(job['sid'], status='failed', text='Could not start the job', error=str(e)[-400:], kind=job.get('kind'))
            prune_queue(job['id'])
            break
    if time.time() - st['last_heartbeat'] > HEARTBEAT_S:
        pending = [j for j in q['jobs'] if j['id'] not in st['handled']]
        put_json('worker/heartbeat.json', dict(t=time.time(), poller=os.environ.get('SLURM_JOB_ID'), pending=len(pending),
                                               active=bool(st['active']), interval_s=HEARTBEAT_S), max_age=300)
        st['last_heartbeat'] = time.time()
    if time.time() - st['last_expiry'] > 86400: expire(st); st['last_expiry'] = time.time()
    st['handled'] = st['handled'][-500:]; save_state(st)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--once', action='store_true'); ap.add_argument('--max-seconds', type=float, default=5.6 * 3600)
    a = ap.parse_args(); (ROOT / '_worker').mkdir(parents=True, exist_ok=True); t0 = time.time(); st = load_state()
    log(f'poller start (slurm {os.environ.get("SLURM_JOB_ID")}), active={st["active"]}')
    while True:
        try: tick(st, t0)
        except Exception as e: log(f'tick failed: {e}')
        if a.once or time.time() - t0 > a.max_seconds: break
        time.sleep(POLL_S)
    log('poller exit')


if __name__ == '__main__':
    sys.exit(main())
