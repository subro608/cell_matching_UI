// New analysis: drop the two files (read locally for size, channels, pixel sizes and previews), describe them, upload, submit.
// No data: the JY195 example card links to the finished example (Explore) and its files.
import { useEffect, useRef, useState } from 'react';
import Shell from '../components/Shell';
import Icon from '../components/Icon';
import { inspectTiff, previewMip } from '../lib/tiff';
import { createSession, getExample, submit, uploadFile } from '../lib/api';
import { store } from '../lib/data';

const LIMITS = { fileBytes: 900 * 2 ** 20, ivVoxels: 400e6, exPixels: 120e6 };
const MODELS = [['cyto2_ft', 'Fine-tuned cyto2', 'Cellpose 3'], ['cyto3_ft', 'Fine-tuned cyto3', 'Cellpose 3'], ['cpsam', 'CP-SAM fine-tune', 'Cellpose 4']];
// formats: TIFF is read here in the browser; the others are read on the server after upload (ivexmatch.readers)
const ACCEPT = '.tif,.tiff,.nd2,.zip,.npy,.npz,.png,.jpg,.jpeg,.czi,.lif,.oib,.oir,.ims,.lsm';
const SERVER_READ = /\.(nd2|zip|npy|npz|png|jpe?g|czi|lif|oib|oir|ims|lsm)$/i;
const mb = (b) => `${(b / 2 ** 20).toFixed(b < 10 * 2 ** 20 ? 1 : 0)} MB`;
const remember = (sid, title) => store.set('ivex:sessions', [{ sid, title, created: Date.now() / 1000 }, ...store.get('ivex:sessions', [])].slice(0, 30));

function Preview({ info, channel, z0, z1, label }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!info || channel == null || channel === '') return; let live = true;
    previewMip(info, Number(channel), { z0, z1 }).then(({ canvas }) => { if (live && ref.current) ref.current.replaceChildren(canvas); }).catch((e) => console.error(e));
    return () => { live = false; };
  }, [info, channel, z0, z1]);
  return <figure><div ref={ref} className="hint">rendering…</div><figcaption>{label}</figcaption></figure>;
}

function DropZone({ side, info, setInfo, error }) {
  const [over, setOver] = useState(false);
  const take = async (f) => {
    if (!f) return; setInfo(null, null);
    if (/\.pkl$|\.pickle$/i.test(f.name)) return setInfo(null, `${f.name}: Python pickles are not accepted (opening one can run code). Convert it with tools/convert_lut_pickle.py first.`);
    if (SERVER_READ.test(f.name)) return setInfo({ file: f, deferred: true }, null);                 // read on the server after upload
    try { setInfo(await inspectTiff(f), null); } catch (err) { setInfo({ file: f, deferred: true }, null); }    // unusual TIFF: let the server try
  };
  return (
    <>
      <div className={`drop ${over ? 'over' : ''} ${info ? 'has' : ''}`} onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); take(e.dataTransfer.files?.[0]); }}>
        <input type="file" accept={ACCEPT} onChange={(e) => take(e.target.files?.[0])} aria-label={`${side.toUpperCase()} file`} />
        {info?.deferred ? (
          <div className="row" style={{ alignItems: 'flex-start' }}>
            <Icon name="check" size={18} style={{ color: 'var(--accent)', marginTop: 2 }} />
            <div><div className="big" style={{ margin: 0 }}>{info.file.name}</div>
              <div className="hint">{mb(info.file.size)} · read on the server after upload: dimensions, channels, positions and pixel sizes are shown for you to confirm before the analysis starts.</div></div>
          </div>
        ) : info ? (
          <div className="row" style={{ alignItems: 'flex-start' }}>
            <Icon name="check" size={18} style={{ color: 'var(--accent)', marginTop: 2 }} />
            <div><div className="big" style={{ margin: 0 }}>{info.file.name}</div>
              <div className="hint">{mb(info.file.size)} · {info.Z} plane{info.Z > 1 ? 's' : ''} × {info.C} channel{info.C > 1 ? 's' : ''} × {info.H} × {info.W} px · {info.dtype}</div>
              <div className="hint">Drop another file to replace it.</div></div>
          </div>
        ) : (
          <><Icon name="upload" size={22} /><div className="big">{side === 'iv' ? 'Drop the in vivo stack' : 'Drop the ex vivo image or stack'}</div>
            <div className="hint">TIFF, OME-TIFF, Nikon ND2, zip of per-plane TIFFs (e.g. Bruker), CZI, LIF, NPY / NPZ, PNG · or click to choose</div></>
        )}
      </div>
      {error && <div className="err" style={{ marginTop: 6 }}>{error}</div>}
    </>
  );
}

function ExampleCard() {
  const [ex, setEx] = useState(null);
  useEffect(() => { getExample().then(setEx).catch(() => setEx(false)); }, []);
  if (ex === false) return null;
  const dl = ex ? Object.entries(ex.info.downloads || {}) : [];
  return (
    <div className="card example-card">
      {ex?.info?.thumbs?.iv && <img src={ex.urls[ex.base + ex.info.thumbs.iv]} alt="JY195 in vivo max projection" />}
      <div style={{ flex: 1 }}>
        <div className="kicker">No data at hand?</div>
        <h2 style={{ margin: '4px 0 6px' }}>{ex?.info?.title || 'The JY195 example'}</h2>
        <p className="hint" style={{ margin: 0 }}>{ex?.info?.description || 'Loading…'}</p>
        {dl.length > 0 && <div className="hint" style={{ marginTop: 6 }}>Files: {dl.map(([k, d], i) => <span key={k}>{i ? ' · ' : ''}<a href={ex.urls[ex.base + d.path]} download>{d.label || k.toUpperCase()}</a> ({d.mb} MB)</span>)}</div>}
      </div>
      <a className="btn primary" href="?page=example"><Icon name="eye" />Explore it</a>
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
  if (!iv) problems.push('Add the IV file.'); if (!ex) problems.push('Add the EX file.');
  if (iv && iv.file.size > LIMITS.fileBytes) problems.push(`The IV file is ${mb(iv.file.size)}; the pilot accepts up to ${mb(LIMITS.fileBytes)}.`);
  if (ex && ex.file.size > LIMITS.fileBytes) problems.push(`The EX file is ${mb(ex.file.size)}; the pilot accepts up to ${mb(LIMITS.fileBytes)}.`);
  if (iv && !iv.deferred && iv.Z * iv.H * iv.W > LIMITS.ivVoxels) problems.push(`The IV stack has ${((iv.Z * iv.H * iv.W) / 1e6).toFixed(0)} M voxels per channel; the limit is ${LIMITS.ivVoxels / 1e6} M.`);
  if (ex && !ex.deferred && ex.H * ex.W > LIMITS.exPixels) problems.push(`The EX image is ${((ex.H * ex.W) / 1e6).toFixed(0)} M pixels; the limit is ${LIMITS.exPixels / 1e6} M.`);
  if (!iv?.deferred && !(Number(f.iv_um) > 0)) problems.push('Give the IV pixel size (µm).'); if (!ex?.deferred && !(Number(f.ex_um) > 0)) problems.push('Give the EX pixel size (µm).');
  if (!f.models.length) problems.push('Choose at least one mask model.'); if (!f.invite.trim()) problems.push('Enter your invite code.');
  if (ex && !ex.deferred && f.dapi !== '' && Number(f.dapi) === Number(f.ex_ch)) problems.push('The DAPI channel must differ from the matching channel.');
  const stage = !iv || !ex ? 0 : !(Number(f.iv_um) > 0 && Number(f.ex_um) > 0) ? 1 : phase ? 3 : 2;

  const deferred = !!(iv?.deferred || ex?.deferred);
  const later = (info, um, umz) => ({ deferred: true, bytes: info.file.size, name: info.file.name, um_xy: Number(um) > 0 ? Number(um) : null, um_z: Number(umz) > 0 ? Number(umz) : null });
  const spec = () => (deferred ? {
    title: f.title || `${iv.file.name} ↔ ${ex.file.name}`, models: f.models, match_model: f.models.includes(f.match) ? f.match : f.models[0],
    iv: iv.deferred ? later(iv, f.iv_um, f.iv_um_z) : { ...full().iv }, ex: ex.deferred ? later(ex, f.ex_um, null) : { ...full().ex },
  } : full());
  const full = () => ({
    title: f.title || `${iv.file.name} ↔ ${ex.file.name}`, models: f.models, match_model: f.models.includes(f.match) ? f.match : f.models[0],
    iv: { um_xy: Number(f.iv_um), um_z: f.iv_um_z === '' ? null : Number(f.iv_um_z), channel: Number(f.iv_ch), diameter_um: Number(f.iv_d), bytes: iv.file.size, shape: iv.shape, name: iv.file.name },
    ex: { um_xy: Number(f.ex_um), channel: Number(f.ex_ch), dapi_channel: f.dapi === '' ? null : Number(f.dapi), z_range: ex.Z > 1 && (Number(f.ex_z0) > 0 || Number(f.ex_z1) < ex.Z) ? `${f.ex_z0}:${f.ex_z1}` : '',
      diameter_um: Number(f.ex_d), bytes: ex.file.size, shape: ex.shape, name: ex.file.name },
  });

  const go = async () => {
    setFail(null);
    try {
      let id = sid;
      if (!id) { setPhase('session'); const s = spec(); ({ sid: id } = await createSession(f.invite.trim(), s)); setSid(id); remember(id, s.title); }
      for (const [side, info] of [['iv', iv], ['ex', ex]]) {
        if (prog[side] >= 1) continue; setPhase(`upload-${side}`);
        await uploadFile(id, side, info.file, (p) => setProg((s) => ({ ...s, [side]: p }))); setProg((s) => ({ ...s, [side]: 1 }));
      }
      setPhase('submit'); await submit(id); window.location.search = `?s=${id}`;          // deferred files: the session page asks to confirm
    } catch (e) { setFail(String(e.message || e)); setPhase(null); }
  };

  const chans = (info) => Array.from({ length: info?.C || 1 }, (_, c) => c);
  return (
    <Shell active="new">
      <div className="page">
        <div className="kicker">New analysis</div>
        <h1>Upload a pair</h1>
        <p className="lead">Your browser reads both files first and shows what will be analysed. Nothing leaves your computer until you press Upload.</p>
        <div className="stepper">{['1 · Files', '2 · Describe', '3 · Models', '4 · Upload'].map((s, i) => <span key={s} className={i <= stage ? 'on' : ''}>{s}</span>)}</div>
        <ExampleCard />
        <div className="grid2">
          <div className="card">
            <h2>In vivo · 2-photon stack</h2>
            <DropZone side="iv" info={iv} setInfo={(i, e) => { setIv(i); setErrs((s) => ({ ...s, iv: e })); }} error={errs.iv} />
            <div className="inline">
              <div className="field"><label>Pixel size (µm)</label><input type="number" step="any" value={f.iv_um} onChange={set('iv_um')} placeholder={iv?.deferred ? 'from the file, if stated' : ''} /></div>
              <div className="field"><label>Z step (µm)</label><input type="number" step="any" value={f.iv_um_z} onChange={set('iv_um_z')} /></div>
            </div>
            <div className="inline">
              <div className="field" hidden={iv?.deferred}><label>Channel to match on</label><select value={f.iv_ch} onChange={set('iv_ch')}>{chans(iv).map((c) => <option key={c} value={c}>channel {c}{iv?.names?.[c] ? ` · ${iv.names[c]}` : ''}</option>)}</select></div>
              <div className="field"><label>Cell diameter (µm)</label><input type="number" step="any" value={f.iv_d} onChange={set('iv_d')} /></div>
            </div>
            {iv && !iv.deferred && <div className="preview"><Preview info={iv} channel={f.iv_ch} label={`max projection · channel ${f.iv_ch}`} /></div>}
          </div>
          <div className="card">
            <h2>Ex vivo · confocal</h2>
            <DropZone side="ex" info={ex} setInfo={(i, e) => { setEx(i); setErrs((s) => ({ ...s, ex: e })); }} error={errs.ex} />
            <div className="inline">
              <div className="field"><label>Pixel size (µm)</label><input type="number" step="any" value={f.ex_um} onChange={set('ex_um')} placeholder={ex?.deferred ? 'from the file, if stated' : ''} /></div>
              <div className="field"><label>Cell diameter (µm)</label><input type="number" step="any" value={f.ex_d} onChange={set('ex_d')} /></div>
            </div>
            <div className="inline">
              <div className="field" hidden={ex?.deferred}><label>GFP / matching channel</label><select value={f.ex_ch} onChange={set('ex_ch')}>{chans(ex).map((c) => <option key={c} value={c}>channel {c}{ex?.names?.[c] ? ` · ${ex.names[c]}` : ''}</option>)}</select></div>
              <div className="field" hidden={ex?.deferred}><label>DAPI channel (optional)</label><select value={f.dapi} onChange={set('dapi')}><option value="">none</option>{chans(ex).map((c) => <option key={c} value={c}>channel {c}{ex?.names?.[c] ? ` · ${ex.names[c]}` : ''}</option>)}</select></div>
            </div>
            {ex && !ex.deferred && ex.Z > 1 && (
              <div className="inline">
                <div className="field"><label>Max-project planes from</label><input type="number" min={0} max={ex.Z - 1} value={f.ex_z0} onChange={set('ex_z0')} /></div>
                <div className="field"><label>To (exclusive)</label><input type="number" min={1} max={ex.Z} value={f.ex_z1} onChange={set('ex_z1')} /></div>
              </div>
            )}
            {ex && !ex.deferred && <div className="preview">
              <Preview info={ex} channel={f.ex_ch} z0={Number(f.ex_z0)} z1={Number(f.ex_z1) || ex.Z} label={`GFP · planes ${f.ex_z0}–${(Number(f.ex_z1) || ex.Z) - 1}`} />
              {f.dapi !== '' && <Preview info={ex} channel={f.dapi} z0={Number(f.ex_z0)} z1={Number(f.ex_z1) || ex.Z} label="DAPI" />}
            </div>}
          </div>
        </div>
        <div className="card">
          <h2>Models and session</h2>
          <div className="field"><label>Mask models</label>
            <div className="chips">{MODELS.map(([k, label, v]) => (
              <label key={k} className={`chip ${f.models.includes(k) ? 'on' : ''}`}>
                <input type="checkbox" checked={f.models.includes(k)} onChange={(e) => setF((s) => ({ ...s, models: e.target.checked ? [...s.models, k] : s.models.filter((m) => m !== k) }))} />
                {label} <span className="hint">{v}</span></label>))}</div>
          </div>
          <div className="inline">
            <div className="field"><label>Mask set that drives matching (re-match with another later)</label>
              <select value={f.match} onChange={set('match')}>{MODELS.filter(([k]) => f.models.includes(k)).map(([k, label]) => <option key={k} value={k}>{label}</option>)}</select></div>
            <div className="field"><label>Session title (optional)</label><input type="text" maxLength={120} value={f.title} onChange={set('title')} placeholder="e.g. mouse 12, region 3" /></div>
          </div>
          <div className="inline">
            <div className="field"><label>Invite code</label><input type="text" value={f.invite} onChange={set('invite')} autoComplete="off" /></div>
            <div className="field" style={{ justifyContent: 'flex-end' }}>
              <button className="primary" style={{ padding: '10px 16px' }} disabled={problems.length > 0 || !!phase} onClick={go}><Icon name="upload" />{sid ? 'Retry upload' : deferred ? 'Upload and read the files' : 'Upload and analyze'}</button>
            </div>
          </div>
          <p className="hint">The orientation is found automatically (PMAC blind pose); you check it in the viewer. By uploading you confirm the data are not human or identifiable,
            and that results stay online for 7 days (<a href="?page=about">privacy</a>).</p>
          {problems.length > 0 && !phase && (iv || ex) && <ul className="hint" style={{ margin: '4px 0 0', paddingLeft: 18 }}>{problems.map((p) => <li key={p}>{p}</li>)}</ul>}
          {phase && (
            <div style={{ marginTop: 10 }}>
              {phase === 'session' && <div className="hint">Creating the session…</div>}
              {['iv', 'ex'].map((side) => (
                <div key={side} className="row"><b style={{ width: 28 }}>{side.toUpperCase()}</b><div className="progress" style={{ flex: 1 }}><div style={{ width: `${prog[side] * 100}%` }} /></div><span className="hint">{(prog[side] * 100).toFixed(0)}%</span></div>
              ))}
              {phase === 'submit' && <div className="hint">Queuing the analysis…</div>}
            </div>
          )}
          {fail && <div className="err">Failed: {fail}{sid ? ' — press Retry to continue the upload.' : ''}</div>}
        </div>
      </div>
    </Shell>
  );
}
