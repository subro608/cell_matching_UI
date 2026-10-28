import { useEffect, useState } from 'react';
import Shell, { useWorker } from '../components/Shell';
import Icon from '../components/Icon';
import { getExample } from '../lib/api';
import { store } from '../lib/data';

export function WorkerBanner({ worker }) {
  if (!worker) return null;
  if (worker.online) return <div className="banner ok"><Icon name="server" />The analysis worker on NYU HPC is online{worker.pending ? ` · ${worker.pending} job(s) waiting` : ''}. Jobs wait for a free GPU: minutes, sometimes hours.</div>;
  return <div className="banner warn"><Icon name="server" />The analysis worker is offline{worker.last_seen_s ? ` (last seen ${(worker.last_seen_s / 3600).toFixed(1)} h ago)` : ''}. You can still upload; your job starts when it is back.</div>;
}

const STEPS = [
  ['Masks', 'Three Cellpose models segment every IV plane and the EX projection. Show each, or only the cells they agree on.'],
  ['Orientation', 'PMAC finds the mirror, rotation and scale between the two fields on its own, with a shuffle-null p-value.'],
  ['Matches', 'For every IV cell, PMAC ranks the EX cells. Crops, scores and an overlay for each candidate.'],
  ['Register', 'Accept pairs with one key. Similarity, affine or thin-plate spline per plane, with leave-one-out error.'],
];

export default function Home() {
  const worker = useWorker(), [ex, setEx] = useState(null);
  useEffect(() => { getExample().then(setEx).catch(() => setEx(null)); }, []);
  const mine = store.get('ivex:sessions', []);
  const thumb = (k) => ex?.info?.thumbs?.[k] && ex.urls[ex.base + ex.info.thumbs[k]];
  return (
    <Shell active="home" worker={worker}>
      <div className="page">
        <div className="hero">
          <div>
            <div className="kicker">In vivo ↔ ex vivo</div>
            <h1>Find the same neurons <span className="g">in both microscopes.</span></h1>
            <p className="lead">Upload a 2-photon stack and a confocal image of the same tissue. ivex segments both, finds how they are oriented, proposes
              cell-to-cell matches with PMAC, and turns the pairs you verify into a slice-wise registration.</p>
            <div className="actions">
              <a className="btn primary lg" href="?page=new"><Icon name="upload" />Start an analysis</a>
              <a className="btn ghost lg" href="?page=example"><Icon name="eye" />Explore the JY195 example</a>
            </div>
          </div>
          <div className="hero-art">
            {thumb('iv') ? <><img className="a" src={thumb('iv')} alt="In vivo max projection (JY195)" /><img className="b" src={thumb('ex')} alt="Ex vivo region (JY195)" />
              <span className="tag" style={{ left: '8%', top: '4%' }}>in vivo · 2-photon</span><span className="tag" style={{ right: '8%', bottom: '4%' }}>ex vivo · confocal</span></>
              : <div className="splash"><div><div className="spinner" />loading example…</div></div>}
          </div>
        </div>
        <WorkerBanner worker={worker} />
        <h2>How it works</h2>
        <div className="steps-grid">{STEPS.map(([t, d], i) => <div className="step-card" key={t}><div className="n">{i + 1}</div><h3>{t}</h3><p>{d}</p></div>)}</div>
        {mine.length > 0 && (
          <div className="card">
            <h2>Your sessions <span className="hint">(this browser)</span></h2>
            <ul className="sessions">{mine.map((s) => <li key={s.sid}><a href={`?s=${s.sid}`}>{s.title || s.sid}</a><span className="hint">{new Date(s.created * 1000).toLocaleString()}</span></li>)}</ul>
            <div className="hint" style={{ marginTop: 8 }}>Anyone with a session link can open it. Share links only with collaborators.</div>
          </div>
        )}
        <div className="card">
          <h2>Pilot</h2>
          <p style={{ margin: 0 }}>Invite-only. Analysis runs on NYU's HPC cluster, so a job may wait in its queue. Results stay online for 7 days after the analysis and can be
            restored later. Do not upload human or identifiable data. <a href="?page=about">Privacy and retention</a></p>
        </div>
      </div>
    </Shell>
  );
}
