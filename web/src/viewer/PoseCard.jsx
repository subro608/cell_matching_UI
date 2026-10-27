// Orientation card: PMAC's blind pose (flip / rotation / scale, inliers vs the shuffle null, family-wise p), the runner-up, and
// "Looks right" / "Use runner-up" / "Set manually". Setting it manually previews live on the overlay; "Match at this orientation"
// queues a re-match on the HPC worker (sessions only).
const fmt = (v, d = 1) => (Number.isFinite(v) ? v.toFixed(d) : '–');

function HypothesisChart({ hypotheses, best }) {
  if (!hypotheses?.length) return null;
  const W = 280, H = 70, max = Math.max(1, ...hypotheses.map((h) => h.score));
  const pts = (flip) => hypotheses.filter((h) => h.flip === flip).sort((a, b) => a.theta - b.theta);
  const x = (t) => (t / 360) * (W - 8) + 4, y = (s) => H - 12 - (s / max) * (H - 20);
  return (
    <svg className="hchart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Inliers for each rotation hypothesis">
      <line x1="4" x2={W - 4} y1={H - 12} y2={H - 12} className="axis" />
      {[0, 90, 180, 270, 360].map((t) => <text key={t} x={x(t)} y={H - 1} className="tick">{t}°</text>)}
      {[false, true].map((flip) => (
        <g key={String(flip)} className={flip ? 'flip' : 'noflip'}>
          {pts(flip).map((h) => <circle key={`${h.theta}`} cx={x(h.theta)} cy={y(h.score)} r={h.theta === best?.theta_grid && h.flip === best?.flip ? 3.5 : 1.6} />)}
        </g>
      ))}
    </svg>
  );
}

export default function PoseCard({ pose, runs, runIdx, setRunIdx, poseOk, setPoseOk, manual, setManual, onMatchManual, onRunnerUp, canRematch, busy }) {
  const run = runs[runIdx];
  if (!pose) {
    return (
      <section>
        <h2>Orientation</h2>
        <div className="hint">No blind pose in this bundle. {run?.iv_to_ex_native ? 'A given transform is used.' : 'Set it manually to link and overlay the views.'}</div>
        <ManualControls manual={manual} setManual={setManual} onMatchManual={onMatchManual} canRematch={canRematch} busy={busy} />
      </section>
    );
  }
  const null95 = pose.null_max?.length ? [...pose.null_max].sort((a, b) => a - b)[Math.floor(0.95 * (pose.null_max.length - 1))] : null;
  const ruRun = runs.findIndex((r) => r.kind === 'runner_up');
  return (
    <section className="posecard">
      <h2>Orientation {pose.provisional ? '(provisional)' : '· PMAC blind pose'}</h2>
      {pose.provisional ? <div className="hint warn">{pose.label}</div> : (
        <>
          <div className={`pose-head ${pose.uncertain ? 'uncertain' : 'ok'}`}>
            {pose.uncertain ? 'Orientation uncertain, please check' : 'Orientation found'}
            <span className="hint"> · p = {fmt(pose.p_familywise, 3)}{pose.n_null ? ` (${pose.n_null} shuffles)` : ''}</span>
          </div>
          <div className="kv">
            <span>flip</span><b>{pose.flip ? 'yes' : 'no'}</b><span>rotation</span><b>{fmt(pose.rotation_deg)}°</b>
            <span>scale</span><b>{fmt(pose.scale, 3)}</b><span>inliers</span><b>{pose.best?.score ?? '–'} <span className="hint">vs null 95% {null95 ?? '–'}</span></b>
            <span>residual</span><b>{fmt(pose.median_residual_um)} µm</b><span>queries</span><b>{pose.n_query} IV · {pose.n_ex} EX</b>
          </div>
          <HypothesisChart hypotheses={pose.hypotheses} best={pose.best} />
          <div className="hint">dots: inliers per rotation · <span className="noflip-key">no flip</span> · <span className="flip-key">flipped</span></div>
        </>
      )}
      {runs.length > 1 && (
        <div className="field" style={{ margin: '10px 0 4px' }}><label>Matches at</label>
          <select value={runIdx} onChange={(e) => setRunIdx(Number(e.target.value))}>{runs.map((r, i) => <option key={r.name} value={i}>{r.label}</option>)}</select>
        </div>
      )}
      <div className="seg">
        <button className={poseOk === true ? 'on' : ''} onClick={() => setPoseOk(true)}>Looks right</button>
        {pose.runner_up && (
          <button disabled={busy || (ruRun < 0 && !canRematch)} title={ruRun < 0 && !canRematch ? 'Needs the server (session mode)' : ''}
            onClick={() => (ruRun >= 0 ? setRunIdx(ruRun) : onRunnerUp(pose.runner_up))}>
            Use runner-up ({pose.runner_up.flip ? 'flip, ' : ''}{fmt(pose.runner_up.theta, 0)}°, {pose.runner_up.score})
          </button>
        )}
        <button className={manual.on ? 'on' : ''} onClick={() => setManual((m) => ({ ...m, on: !m.on }))}>Set manually</button>
      </div>
      {manual.on && <ManualControls manual={manual} setManual={setManual} onMatchManual={onMatchManual} canRematch={canRematch} busy={busy} />}
    </section>
  );
}

function ManualControls({ manual, setManual, onMatchManual, canRematch, busy }) {
  const set = (k) => (e) => setManual((m) => ({ ...m, on: true, [k]: e.target.type === 'checkbox' ? e.target.checked : Number(e.target.value) }));
  return (
    <div className="manual">
      <div className="hint">Preview on the overlay (O). Drag the sliders until the cells line up.</div>
      <label className="row">rotation <input type="range" min={-180} max={180} step={0.5} value={manual.theta} onChange={set('theta')} /><b>{manual.theta.toFixed(1)}°</b></label>
      <label className="row">scale <input type="range" min={0.6} max={1.6} step={0.005} value={manual.scale} onChange={set('scale')} /><b>{manual.scale.toFixed(3)}</b></label>
      <label className="row">shift x <input type="range" min={-2000} max={2000} step={1} value={manual.dx} onChange={set('dx')} /><b>{manual.dx} µm</b></label>
      <label className="row">shift y <input type="range" min={-2000} max={2000} step={1} value={manual.dy} onChange={set('dy')} /><b>{manual.dy} µm</b></label>
      <label className="row"><input type="checkbox" checked={manual.flip} onChange={set('flip')} />mirror IV (flip x)</label>
      <div className="seg">
        <button disabled={!canRematch || busy} title={canRematch ? '' : 'Needs the server (session mode)'} onClick={onMatchManual}>Match at this orientation</button>
        <button onClick={() => setManual((m) => ({ ...m, ...m.init, on: true }))}>Reset</button>
      </div>
    </div>
  );
}
