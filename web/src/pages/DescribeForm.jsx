// After the worker has read the uploads (state 'probed'): what it found, as a pre-filled form. The user confirms channels, position,
// pixel sizes and the EX projection planes; then the analysis is queued.
import { useState } from 'react';
import { describe, submit } from '../lib/api';

function Side({ side, m, urls, prefix, v, set }) {
  const label = side === 'iv' ? 'In vivo' : 'Ex vivo';
  const chans = m.channel_names.map((n, c) => <option key={c} value={c}>{n}</option>);
  return (
    <div className="card">
      <h2>{label} · {m.format}</h2>
      <div className="hint">{m.Z} plane{m.Z > 1 ? 's' : ''} × {m.C} channel{m.C > 1 ? 's' : ''} × {m.Y} × {m.X} px · {m.dtype}
        {m.um_source ? ` · pixel size from ${m.um_source}` : ''}</div>
      {(m.warnings || []).map((w) => <div key={w} className="msg">{w}</div>)}
      {m.positions?.length > 1 && (
        <div className="field"><label>Position (one per analysis; previews show position {m.position})</label>
          <select value={v.position} onChange={set('position')}>{m.positions.map((p) => <option key={p.index} value={p.index}>{p.index}: {p.name}</option>)}</select></div>
      )}
      <div className="inline">
        <div className="field"><label>Pixel size (µm)</label><input type="number" step="any" value={v.um_xy} onChange={set('um_xy')} /></div>
        {side === 'iv' && <div className="field"><label>Z step (µm)</label><input type="number" step="any" value={v.um_z} onChange={set('um_z')} /></div>}
        <div className="field"><label>Cell diameter (µm)</label><input type="number" step="any" value={v.diameter_um} onChange={set('diameter_um')} /></div>
      </div>
      <div className="inline">
        <div className="field"><label>{side === 'iv' ? 'Channel to match on' : 'GFP / matching channel'}</label><select value={v.channel} onChange={set('channel')}>{chans}</select></div>
        {side === 'ex' && <div className="field"><label>DAPI channel (optional)</label><select value={v.dapi_channel} onChange={set('dapi_channel')}><option value="">none</option>{chans}</select></div>}
      </div>
      {side === 'ex' && m.Z > 1 && (
        <div className="inline">
          <div className="field"><label>Max-project planes from</label><input type="number" min={0} max={m.Z - 1} value={v.z0} onChange={set('z0')} /></div>
          <div className="field"><label>To (exclusive)</label><input type="number" min={1} max={m.Z} value={v.z1} onChange={set('z1')} /></div>
        </div>
      )}
      <div className="preview">{Object.entries(m.previews || {}).map(([c, p]) => (
        <figure key={c}><img src={urls[prefix + p]} alt={`${label} ${m.channel_names[c]}`} style={{ width: 170, border: '1px solid var(--line)', background: '#000', display: 'block' }} />
          <figcaption>{m.channel_names[c]}{Number(c) === Number(v.channel) ? ' · matching' : ''}{side === 'ex' && Number(c) === Number(v.dapi_channel) ? ' · DAPI' : ''}</figcaption></figure>))}</div>
    </div>
  );
}

const guessDapi = (names) => { const i = names.findIndex((n) => /dapi|hoechst/i.test(n)); return i >= 0 ? i : ''; };
const guessGfp = (names) => Math.max(0, names.findIndex((n) => /gfp|gcamp|ch2|green|488/i.test(n)));

export default function DescribeForm({ s }) {
  const p = s.probe, sp = s.spec;
  const init = (side) => {
    const m = p[side], d = sp[side];
    return { position: m.position || 0, um_xy: d.um_xy ?? m.um_xy ?? '', um_z: d.um_z ?? m.um_z ?? '', diameter_um: 11, channel: guessGfp(m.channel_names),
      dapi_channel: side === 'ex' ? guessDapi(m.channel_names) : '', z0: 0, z1: m.Z };
  };
  const [v, setV] = useState({ iv: init('iv'), ex: init('ex') }), [busy, setBusy] = useState(false), [err, setErr] = useState(null);
  const set = (side) => (k) => (e) => setV((x) => ({ ...x, [side]: { ...x[side], [k]: e.target.value } }));
  const missing = ['iv', 'ex'].filter((side) => !(Number(v[side].um_xy) > 0)).map((x) => `${x.toUpperCase()} pixel size`);
  if (!(Number(v.iv.um_z) > 0) && p.iv.Z > 1) missing.push('IV z step');
  const go = async () => {
    setBusy(true); setErr(null);
    const n = (x) => (x === '' || x == null ? null : Number(x));
    try {
      await describe(s.sid, { title: sp.title, models: sp.models, match_model: sp.match_model,
        iv: { um_xy: n(v.iv.um_xy), um_z: n(v.iv.um_z), channel: n(v.iv.channel), diameter_um: n(v.iv.diameter_um), position: n(v.iv.position) },
        ex: { um_xy: n(v.ex.um_xy), channel: n(v.ex.channel), dapi_channel: n(v.ex.dapi_channel), diameter_um: n(v.ex.diameter_um), position: n(v.ex.position),
          z_range: p.ex.Z > 1 && (Number(v.ex.z0) > 0 || Number(v.ex.z1) < p.ex.Z) ? `${v.ex.z0}:${v.ex.z1}` : '' } });
      await submit(s.sid); window.location.reload();
    } catch (e) { setErr(String(e.message || e)); setBusy(false); }
  };
  return (
    <>
      <p className="lead">Your files have been read. Check what was found, fill in anything the files do not state, and start the analysis.</p>
      <div className="grid2"><Side side="iv" m={p.iv} urls={p.urls} prefix={p.prefix} v={v.iv} set={set('iv')} /><Side side="ex" m={p.ex} urls={p.urls} prefix={p.prefix} v={v.ex} set={set('ex')} /></div>
      {Number(v.iv.position) !== (p.iv.position || 0) || Number(v.ex.position) !== (p.ex.position || 0) ? <div className="msg">Previews show the first position; the analysis uses the one you chose.</div> : null}
      {missing.length > 0 && <div className="msg bad">Still needed: {missing.join(', ')}.</div>}
      <button className="primary btn lg" disabled={busy || missing.length > 0} onClick={go}>{busy ? 'Starting…' : 'Start the analysis'}</button>
      {err && <div className="err">{err}</div>}
    </>
  );
}
