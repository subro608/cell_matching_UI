import { useEffect, useState } from 'react';
import { status } from '../lib/api';
import { store } from '../lib/data';

export function WorkerBanner({ worker }) {
  if (!worker) return null;
  if (worker.online) return <div className="banner ok">The analysis worker on NYU HPC is online{worker.pending ? ` · ${worker.pending} job(s) waiting` : ''}. Jobs wait for a free GPU, from a few minutes to a few hours.</div>;
  return <div className="banner warn">The analysis worker is offline{worker.last_seen_s ? ` (last seen ${(worker.last_seen_s / 3600).toFixed(1)} h ago)` : ''}. You can still upload; your job starts when it is back.</div>;
}

export default function Home() {
  const [st, setSt] = useState(null);
  useEffect(() => { status().then(setSt).catch(() => setSt(null)); }, []);
  const mine = store.get('ivex:sessions', []);
  return (
    <div className="page">
      <h1>ivex · in vivo ↔ ex vivo cell matching</h1>
      <p className="lead">Upload a 2-photon stack (IV) and a confocal stack of the same tissue (EX). You get cell masks from three Cellpose
        models, PMAC's automatic orientation and cell matches, a page to verify the matches, and the slice-wise registration the verified pairs imply.</p>
      <WorkerBanner worker={st?.worker} />
      <a className="cta" href="?page=new">Start a new analysis</a><a className="cta secondary" href="?page=about">How it works · privacy</a>
      <div className="card">
        <h2 style={{ marginTop: 0 }}>Try it without your own data</h2>
        <p>A synthetic example: 260 random somata in an 8-plane IV stack, and the same cells in an EX image (GFP + DAPI) that is the IV field mirrored,
          rotated 40° and scaled 0.95. Not biology, but every step works on it, and you can check that PMAC recovers the orientation.</p>
        <a className="cta secondary" href="?data=examples/demo/">Open the example result</a>
        <a className="cta secondary" href="?page=new">Analyze the example yourself</a>
        <div className="hint" style={{ marginTop: 8 }}>Files: <a href="examples/example_iv.ome.tif" download>example_iv.ome.tif</a> (IV, 8 × 480 × 480, 1 µm/px, 3 µm z) ·{' '}
          <a href="examples/example_ex.ome.tif" download>example_ex.ome.tif</a> (EX, 3 planes × GFP/DAPI × 1100 × 1100, 0.6 µm/px) · <a href="examples/example.json">truth</a></div>
      </div>
      {mine.length > 0 && (
        <div className="card">
          <h2 style={{ marginTop: 0 }}>Your sessions (this browser)</h2>
          <ul className="sessions">{mine.map((s) => <li key={s.sid}><a href={`?s=${s.sid}`}>{s.title || s.sid}</a><span className="hint">{new Date(s.created * 1000).toLocaleString()}</span></li>)}</ul>
          <div className="hint">Anyone with a session link can open it. Keep links to yourself and your collaborators.</div>
        </div>
      )}
      <div className="card">
        <h2 style={{ marginTop: 0 }}>Pilot</h2>
        <p>This is an invite-only pilot. The analysis runs on NYU's HPC cluster, so a job may wait in its queue. Results stay online for 7 days after
          the analysis finishes; you can ask to restore them later. Do not upload human or identifiable data.</p>
      </div>
    </div>
  );
}
