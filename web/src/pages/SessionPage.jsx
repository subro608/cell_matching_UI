// A session: status while the job waits and runs, then the viewer. Polls /api/session (cheap: CDN-cached reads).
import { useEffect, useRef, useState } from 'react';
import Viewer from '../viewer/Viewer';
import { getSession, rematch, restore, saveVerify } from '../lib/api';
import { loadBundle, signedSource } from '../lib/data';
import { WorkerBanner } from './Home';
import Shell from '../components/Shell';
import Icon from '../components/Icon';

const STEPS = [['uploaded', 'Uploaded'], ['queued', 'Waiting for the HPC worker'], ['queued_hpc', 'Waiting for a GPU on NYU HPC'], ['masks', 'Cell masks (3 Cellpose models)'],
  ['pose', 'PMAC blind pose'], ['match', 'PMAC matching'], ['export', 'Preparing the viewer files'], ['done', 'Done']];

function stepIndex(st) {
  if (st.status === 'created' || st.status === 'uploading') return 0;
  if (st.status === 'queued') return 1;
  if (st.status === 'running') { const i = STEPS.findIndex(([k]) => k === st.step); return i >= 0 ? i : st.step === 'started' ? 3 : 2; }
  if (st.status === 'done') return STEPS.length - 1;
  return -1;
}

function Status({ s, onRestore }) {
  const st = s.state, at = stepIndex(st);
  return (
    <Shell worker={s.worker}>
    <div className="page narrow">
      <div className="kicker">Session</div>
      <h1>{s.spec.title}</h1>
      <p className="hint"><code>{s.sid}</code> · created {new Date(s.created * 1000).toLocaleString()} · bookmark this page; anyone with the link can open it.</p>
      <WorkerBanner worker={s.worker} />
      {st.status === 'failed' && <div className="card"><h2><Icon name="x" /> The analysis failed</h2><p>{st.text}</p>{st.error && <pre className="err">{st.error}</pre>}
        <p className="hint">Check the pixel sizes and channels, then start a new analysis. If it keeps failing, send this session id to the person who gave you the invite.</p></div>}
      {st.status === 'expired' && <div className="card"><h2>Files expired</h2><p>{st.text}</p><button className="primary" onClick={onRestore}>Restore the files</button></div>}
      {at >= 0 && (
        <div className="card">
          <h2>Progress</h2>
          <ol className="steps">{STEPS.map(([k, label], i) => <li key={k} className={i < at ? 'done' : i === at ? 'now' : ''}>{label}{i === at && st.text && st.text !== label ? <span className="hint"> · {st.text}</span> : ''}{i === at && st.slurm ? <span className="hint"> · Slurm {st.slurm}</span> : ''}</li>)}</ol>
          {st.status === 'created' && <p className="hint">The uploads did not finish. Start again from <a href="?page=new">New analysis</a>.</p>}
          <p className="hint" style={{ marginBottom: 0 }}>This page refreshes by itself. The analysis takes about 5–30 minutes once it has a GPU.</p>
        </div>
      )}
    </div>
    </Shell>
  );
}

export default function SessionPage({ sid }) {
  const [s, setS] = useState(null), [err, setErr] = useState(null), [b, setB] = useState(null);
  const runsKey = useRef(null);
  useEffect(() => {
    let live = true, timer;
    const poll = async () => {
      try {
        const r = await getSession(sid); if (!live) return; setS(r); setErr(null);
        if (r.bundle) {
          const k = JSON.stringify(r.bundle.manifest.runs?.map((x) => x.name)) + r.bundle.manifest.created;
          if (k !== runsKey.current) { runsKey.current = k; const nb = await loadBundle(signedSource(r.bundle.urls, r.bundle.prefix), r.bundle.manifest); if (live) setB({ ...nb, verify: r.bundle.verify }); }
        }
        const busy = !['done', 'failed', 'expired'].includes(r.state.status);
        timer = setTimeout(poll, busy ? 15000 : 120000);
      } catch (e) { if (live) { setErr(String(e.message || e)); timer = setTimeout(poll, 30000); } }
    };
    poll(); return () => { live = false; clearTimeout(timer); };
  }, [sid]);
  if (err && !s) return <div className="splash error">Could not open session <code>{sid}</code>: {err}</div>;
  if (!s) return <div className="splash">Opening session…</div>;
  if (!b) return <Status s={s} onRestore={() => restore(sid).then(() => window.location.reload()).catch((e) => setErr(String(e.message || e)))} />;
  const st = s.state, jobRunning = st.status === 'queued' || st.status === 'running';
  const session = { sid, verify: b.verify, canRematch: st.status === 'done', save: (doc) => saveVerify(sid, doc), rematch: (body) => rematch(sid, body).then((r) => { setS((x) => ({ ...x, state: { ...x.state, status: 'queued', kind: 'rematch' } })); return r; }) };
  return (
    <>
      {jobRunning && <div className="topbar">Re-matching on NYU HPC: {st.text || st.status}{st.slurm ? ` (Slurm ${st.slurm})` : ''}. The new matches appear here when ready.</div>}
      <Viewer key={runsKey.current} b={b} session={session} />
    </>
  );
}
