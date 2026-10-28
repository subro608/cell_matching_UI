import { useEffect, useState } from 'react';
import Shell, { useWorker } from '../components/Shell';
import { getExample } from '../lib/api';
import { store } from '../lib/data';

export function WorkerBanner({ worker }) {
  if (!worker) return null;
  if (worker.online) return <div className="banner ok">Analysis worker (NYU HPC) online{worker.pending ? ` · ${worker.pending} job(s) waiting` : ''}. Jobs wait for a free GPU, usually minutes.</div>;
  return <div className="banner warn">Analysis worker offline{worker.last_seen_s ? ` (last seen ${(worker.last_seen_s / 3600).toFixed(1)} h ago)` : ''}. Uploads are kept and processed when it returns.</div>;
}

const METHODS = [
  ['Segmentation', 'Cellpose: fine-tuned cyto2 and cyto3 (Cellpose 3), CP-SAM fine-tune (Cellpose 4). In vivo planes in 3D with z anisotropy; ex vivo on the projection of the chosen planes.', 'Masks per model; agreement view (cells ≥ 2 or 3 models find)'],
  ['Orientation', 'PMAC appearance descriptors (soma and neighbourhood crops) under every rotation (10° grid, refined to 1°), with and without mirroring; RANSAC similarity; 50 position shuffles.', 'Mirror, rotation, scale; inliers vs null; family-wise p'],
  ['Matching', 'PMAC (Position-aware Matching of Any Cell) at the found orientation: every in vivo cell-plane against every ex vivo cell.', 'Top-10 candidates per cell, P(top-1), mutual best'],
  ['Verification', 'Accept, reject or mark unsure; any ex vivo cell can be picked by hand. Nothing is accepted automatically.', 'Verified pairs (CSV)'],
  ['Registration', 'Similarity, affine or thin-plate spline from the verified pairs, per in vivo plane (pairs within ± Δz) or global; leave-one-out error.', 'Per-plane transforms (JSON)'],
];

export default function Home() {
  const worker = useWorker(), [ex, setEx] = useState(null);
  useEffect(() => { getExample().then(setEx).catch(() => setEx(false)); }, []);
  const mine = store.get('ivex:sessions', []);
  const img = (k) => ex && ex.info?.thumbs?.[k] && ex.urls[ex.base + ex.info.thumbs[k]];
  return (
    <Shell active="home" worker={worker}>
      <div className="page">
        <div className="kicker">Cell registration across imaging modalities</div>
        <h1>Matching in vivo two-photon cells to ex vivo confocal cells</h1>
        <p className="lead">ivex takes a two-photon stack and a confocal image of the same tissue, segments both, estimates their relative orientation,
          and proposes cell-to-cell correspondences with PMAC. You verify the correspondences; the verified pairs define a slice-wise registration
          that can be exported.</p>
        <div className="actions">
          <a className="btn primary lg" href="?page=new">New analysis</a>
          <a className="btn lg" href="?page=example">Open the JY195 example</a>
          <a className="btn lg" href="?page=about">Documentation</a>
        </div>
        <WorkerBanner worker={worker} />

        <figure className="figure">
          <div className="panels">
            <div className="panel-img">{img('iv') ? <img src={img('iv')} alt="JY195 in vivo, maximum projection" /> : <div className="splash"><div className="spinner" /></div>}<span className="lbl">a</span></div>
            <div className="panel-img">{img('ex') ? <img src={img('ex')} alt="JY195 ex vivo region" /> : <div className="splash"><div className="spinner" /></div>}<span className="lbl">b</span></div>
          </div>
          <figcaption><b>Example data, mouse JY195.</b> (a) In vivo two-photon field (green), maximum projection over 39 planes (1.87 µm/px). (b) One ex vivo confocal region (magenta)
            (1.67 µm/px) with 55 annotated cell pairs. On this region PMAC ranks the annotated partner first for 37 of 55 cells (67%; 293 of 439 over all
            13 regions). <a href="?page=example">Open it in the viewer</a>.</figcaption>
        </figure>

        <h2>Pipeline</h2>
        <table className="methods">
          <thead><tr><th>Step</th><th>Method</th><th>Output</th></tr></thead>
          <tbody>{METHODS.map(([s, m, o]) => <tr key={s}><td>{s}</td><td>{m}</td><td>{o}</td></tr>)}</tbody>
        </table>

        {mine.length > 0 && (<>
          <h2>Your sessions</h2>
          <ul className="sessions">{mine.map((s) => <li key={s.sid}><a href={`?s=${s.sid}`}>{s.title || s.sid}</a><span className="hint">{new Date(s.created * 1000).toLocaleString()}</span></li>)}</ul>
          <p className="hint">Stored in this browser. Anyone with a session link can open it.</p>
        </>)}

        <h2>Data handling</h2>
        <p>Invite-only pilot. Files go from your browser to a private store through short-lived signed links; analysis runs on NYU's HPC cluster. Results
          remain online for 7 days and can be restored afterwards. Do not upload human or identifiable data. Details in the <a href="?page=about">documentation</a>.</p>
      </div>
    </Shell>
  );
}
