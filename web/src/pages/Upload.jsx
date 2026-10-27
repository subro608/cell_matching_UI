// New analysis: pick the two files (read locally for size, channels, pixel sizes and previews), describe them, upload, submit.
import { useEffect, useRef, useState } from 'react';
import { inspectTiff, previewMip } from '../lib/tiff';
import { createSession, submit, uploadFile } from '../lib/api';
import { store } from '../lib/data';

const LIMITS = { fileBytes: 900 * 2 ** 20, ivVoxels: 400e6, exPixels: 120e6 };
const MODELS = [['cyto2_ft', 'fine-tuned cyto2 (Cellpose 3)'], ['cyto3_ft', 'fine-tuned cyto3 (Cellpose 3)'], ['cpsam', 'CP-SAM fine-tune (Cellpose 4)']];
const mb = (b) => `${(b / 2 ** 20).toFixed(0)} MB`;

function Preview({ info, channel, z0, z1, label }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!info || channel == null || channel === '') return; let live = true;
    previewMip(info, Number(channel), { z0, z1 }).then(({ canvas }) => { if (live && ref.current) { ref.current.replaceChildren(canvas); } }).catch((e) => console.error(e));
    return () => { live = false; };
  }, [info, channel, z0, z1]);
  return <figure><div ref={ref} className="hint">rendering…</div><figcaption>{label}</figcaption></figure>;
}

function FilePick({ side, info, setInfo, error }) {
  return (
    <div className="field">
      <label>{side === 'iv' ? 'In vivo stack (2-photon), TIFF / OME-TIFF' : 'Ex vivo image or stack (confocal), TIFF / OME-TIFF'}</label>
      <input type="file" accept=".tif,.tiff,image/tiff" onChange={async (e) => {
        const f = e.target.files?.[0]; if (!f) return; setInfo(null, null);
        try { setInfo(await inspectTiff(f), null); } catch (err) { setInfo(null, `Could not read ${f.name}: ${err.message || err}`); }
      }} />
      {info && <div className="hint">{info.file.name} · {mb(info.file.size)} · {info.Z} plane(s) × {info.C} channel(s) × {info.H} × {info.W} px · {info.dtype}</div>}
      {error && <div className="err">{error}</div>}
    </div>
  );
}

export default function Upload() {
  const [iv, setIv] = useState(null), [ex, setEx] = useState(null), [errs, setErrs] = useState({});
  const [f, setF] = useState({ title: '', invite: '', iv_um: '', iv_um_z: '', iv_ch: 0, iv_d: 11, ex_um: '', ex_ch: 0, dapi: '', ex_z0: 0, ex_z1: '', ex_d: 11, models: MODELS.map((m) => m[0]), match: 'cyto3_ft' });
  const [phase, setPhase] = useState(null), [prog, setProg] = useState({ iv: 0, ex: 0 }), [fail, setFail] = useState(null), [sid, setSid] = useState(null);
  const set = (k) => (e) => setF((s) => ({ ...s, [k]: e.target.value }));
  useEffect(() => { if (iv) setF((s) => ({ ...s, iv_um: iv.um_xy ? +iv.um_xy.toFixed(5) : s.iv_um, iv_um_z: iv.um_z ? +iv.um_z.toFixed(4) : s.iv_um_z, iv_ch: 0 })); }, [iv]);
  useEffect(() => { if (ex) setF((s) => ({ ...s, ex_um: ex.um_xy ? +ex.um_xy.toFixed(5) : s.ex_um, ex_ch: 0, dapi: ex.C > 1 ? 1 : '', ex_z0: 0, ex_z1: ex.Z })); }, [ex]);

  const problems = [];
  if (!iv) problems.push('Choose the IV file.'); if (!ex) problems.push('Choose the EX file.');
  if (iv && iv.file.size > LIMITS.fileBytes) problems.push(`The IV file is ${mb(iv.file.size)}; the pilot accepts up to ${mb(LIMITS.fileBytes)}.`);
  if (ex && ex.file.size > LIMITS.fileBytes) problems.push(`The EX file is ${mb(ex.file.size)}; the pilot accepts up to ${mb(LIMITS.fileBytes)}.`);
  if (iv && iv.Z * iv.H * iv.W > LIMITS.ivVoxels) problems.push(`The IV stack has ${((iv.Z * iv.H * iv.W) / 1e6).toFixed(0)} M voxels per channel; the limit is ${LIMITS.ivVoxels / 1e6} M.`);
  if (ex && ex.H * ex.W > LIMITS.exPixels) problems.push(`The EX image is ${((ex.H * ex.W) / 1e6).toFixed(0)} M pixels; the limit is ${LIMITS.exPixels / 1e6} M.`);
  if (!(Number(f.iv_um) > 0)) problems.push('Give the IV pixel size (µm).'); if (!(Number(f.ex_um) > 0)) problems.push('Give the EX pixel size (µm).');
  if (!f.models.length) problems.push('Choose at least one mask model.'); if (!f.invite.trim()) problems.push('Enter your invite code.');
  if (ex && f.dapi !== '' && Number(f.dapi) === Number(f.ex_ch)) problems.push('The DAPI channel must differ from the matching channel.');

  const spec = () => ({
    title: f.title || `${iv.file.name} ↔ ${ex.file.name}`, models: f.models, match_model: f.models.includes(f.match) ? f.match : f.models[0],
    iv: { um_xy: Number(f.iv_um), um_z: f.iv_um_z === '' ? null : Number(f.iv_um_z), channel: Number(f.iv_ch), diameter_um: Number(f.iv_d), bytes: iv.file.size, shape: iv.shape, name: iv.file.name },
    ex: { um_xy: Number(f.ex_um), channel: Number(f.ex_ch), dapi_channel: f.dapi === '' ? null : Number(f.dapi), z_range: ex.Z > 1 && (Number(f.ex_z0) > 0 || Number(f.ex_z1) < ex.Z) ? `${f.ex_z0}:${f.ex_z1}` : '',
      diameter_um: Number(f.ex_d), bytes: ex.file.size, shape: ex.shape, name: ex.file.name },
  });

  const go = async () => {
    setFail(null);
    try {
      let id = sid;
      if (!id) { setPhase('session'); const s = spec(); ({ sid: id } = await createSession(f.invite.trim(), s)); setSid(id);
        store.set('ivex:sessions', [{ sid: id, title: s.title, created: Date.now() / 1000 }, ...store.get('ivex:sessions', [])].slice(0, 30)); }
      for (const [side, info] of [['iv', iv], ['ex', ex]]) {
        if (prog[side] >= 1) continue; setPhase(`upload-${side}`);
        await uploadFile(id, side, info.file, (p) => setProg((s) => ({ ...s, [side]: p }))); setProg((s) => ({ ...s, [side]: 1 }));
      }
      setPhase('submit'); await submit(id); window.location.search = `?s=${id}`;
    } catch (e) { setFail(String(e.message || e)); setPhase(null); }
  };

  const chans = (info) => Array.from({ length: info?.C || 1 }, (_, c) => c);
  return (
    <div className="page">
      <a href="./">← ivex</a>
      <h1>New analysis</h1>
      <p className="lead">Your browser reads both files first and shows what will be analysed. Nothing is uploaded until you press the button.</p>
      <div className="grid2">
        <div className="card">
          <h2 style={{ marginTop: 0 }}>In vivo (IV)</h2>
          <FilePick side="iv" info={iv} setInfo={(i, e) => { setIv(i); setErrs((s) => ({ ...s, iv: e })); }} error={errs.iv} />
          <div className="inline">
            <div className="field"><label>pixel size (µm)</label><input type="number" step="any" value={f.iv_um} onChange={set('iv_um')} /></div>
            <div className="field"><label>z step (µm)</label><input type="number" step="any" value={f.iv_um_z} onChange={set('iv_um_z')} /></div>
          </div>
          <div className="inline">
            <div className="field"><label>channel to match on</label><select value={f.iv_ch} onChange={set('iv_ch')}>{chans(iv).map((c) => <option key={c} value={c}>channel {c}{iv?.names?.[c] ? ` · ${iv.names[c]}` : ''}</option>)}</select></div>
            <div className="field"><label>cell diameter (µm)</label><input type="number" step="any" value={f.iv_d} onChange={set('iv_d')} /></div>
          </div>
          {iv && <div className="preview"><Preview info={iv} channel={f.iv_ch} label={`max projection, channel ${f.iv_ch}`} /></div>}
        </div>
        <div className="card">
          <h2 style={{ marginTop: 0 }}>Ex vivo (EX)</h2>
          <FilePick side="ex" info={ex} setInfo={(i, e) => { setEx(i); setErrs((s) => ({ ...s, ex: e })); }} error={errs.ex} />
          <div className="inline">
            <div className="field"><label>pixel size (µm)</label><input type="number" step="any" value={f.ex_um} onChange={set('ex_um')} /></div>
            <div className="field"><label>cell diameter (µm)</label><input type="number" step="any" value={f.ex_d} onChange={set('ex_d')} /></div>
          </div>
          <div className="inline">
            <div className="field"><label>GFP / matching channel</label><select value={f.ex_ch} onChange={set('ex_ch')}>{chans(ex).map((c) => <option key={c} value={c}>channel {c}{ex?.names?.[c] ? ` · ${ex.names[c]}` : ''}</option>)}</select></div>
            <div className="field"><label>DAPI channel (optional)</label><select value={f.dapi} onChange={set('dapi')}><option value="">none</option>{chans(ex).map((c) => <option key={c} value={c}>channel {c}</option>)}</select></div>
          </div>
          {ex && ex.Z > 1 && (
            <div className="inline">
              <div className="field"><label>max-project planes from</label><input type="number" min={0} max={ex.Z - 1} value={f.ex_z0} onChange={set('ex_z0')} /></div>
              <div className="field"><label>to (exclusive)</label><input type="number" min={1} max={ex.Z} value={f.ex_z1} onChange={set('ex_z1')} /></div>
            </div>
          )}
          {ex && <div className="preview">
            <Preview info={ex} channel={f.ex_ch} z0={Number(f.ex_z0)} z1={Number(f.ex_z1) || ex.Z} label={`GFP, planes ${f.ex_z0}–${(Number(f.ex_z1) || ex.Z) - 1}`} />
            {f.dapi !== '' && <Preview info={ex} channel={f.dapi} z0={Number(f.ex_z0)} z1={Number(f.ex_z1) || ex.Z} label="DAPI" />}
          </div>}
        </div>
      </div>
      <div className="card">
        <div className="field"><label>Session title (optional)</label><input type="text" maxLength={120} value={f.title} onChange={set('title')} placeholder="e.g. mouse 12, region 3" /></div>
        <div className="field"><label>Mask models</label>
          {MODELS.map(([k, label]) => <label key={k} className="row"><input type="checkbox" checked={f.models.includes(k)} onChange={(e) => setF((s) => ({ ...s, models: e.target.checked ? [...s.models, k] : s.models.filter((m) => m !== k) }))} />{label}</label>)}
        </div>
        <div className="field"><label>Mask set that drives matching (you can re-match with another later)</label>
          <select value={f.match} onChange={set('match')}>{MODELS.filter(([k]) => f.models.includes(k)).map(([k, label]) => <option key={k} value={k}>{label}</option>)}</select></div>
        <div className="field"><label>Invite code</label><input type="text" value={f.invite} onChange={set('invite')} autoComplete="off" /></div>
        <p className="hint">The orientation is found automatically (PMAC blind pose); you check it in the viewer. By uploading you confirm the data are not human or identifiable,
          and that results stay online for 7 days (see <a href="?page=about">privacy</a>).</p>
        {problems.length > 0 && !phase && <ul className="hint">{problems.map((p) => <li key={p}>{p}</li>)}</ul>}
        <button className="cta" style={{ border: 0 }} disabled={problems.length > 0 || !!phase} onClick={go}>{sid ? 'Retry upload' : 'Upload and analyze'}</button>
        {phase && (
          <div style={{ marginTop: 10 }}>
            {phase === 'session' && <div className="hint">Creating the session…</div>}
            {['iv', 'ex'].map((side) => (
              <div key={side} className="row"><span style={{ width: 28 }}>{side.toUpperCase()}</span><div className="progress" style={{ flex: 1 }}><div style={{ width: `${prog[side] * 100}%` }} /></div><span className="hint">{(prog[side] * 100).toFixed(0)}%</span></div>
            ))}
            {phase === 'submit' && <div className="hint">Queuing the analysis…</div>}
          </div>
        )}
        {fail && <div className="err">Failed: {fail}{sid ? ' — press Retry to continue the upload.' : ''}</div>}
      </div>
    </div>
  );
}
