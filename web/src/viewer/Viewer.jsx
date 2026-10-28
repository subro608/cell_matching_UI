// The session viewer: masks (three models, agreement), PMAC orientation card, matches with crop cards and filters, verification
// (keys), slice-wise registration (similarity / affine / TPS, leave-one-out) with blend / swipe / checkerboard overlays, exports.
// Works on a local folder bundle (?data=) and on a session (signed Blob URLs; verdicts saved to the server).
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ImageView, { OverlayCanvas, fitView } from '../components/ImageView';
import { autoContrast, getOutlines, hasPlane, imageSize, loadRun, store } from '../lib/data';
import { MIN_PAIRS, apply, fit, invert, isMirrored, linear, looErrors, median, modelMatrix, residuals, rotationOf, scaleOf, sliceFits, toNative } from '../lib/fit';
import PoseCard from './PoseCard';
import CropCards from './CropCards';
import Icon from '../components/Icon';
import { agreement } from './agreement';

const RANK_COLORS = [[60, 255, 120, 255], [255, 230, 60, 255], [255, 160, 40, 255]];
const rankColor = (k) => RANK_COLORS[Math.min(k, 2)];
const GREEN = [40, 255, 90, 255], CYAN = [0, 230, 255, 255];
const pct = (logp) => `${(100 * Math.exp(logp)).toFixed(1)}%`;
const affineOf = (m) => (m ? { kind: 'pose', A: [m[0].slice(0, 2), m[1].slice(0, 2)], t: [m[0][2], m[1][2]] } : null);
const toRows = (T) => [[T.A[0][0], T.A[0][1], T.t[0]], [T.A[1][0], T.A[1][1], T.t[1]]];
const download = (name, text, type) => { const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name; a.click(); };

// manual orientation <-> native affine: EX = s_px R(theta) F (p - c_iv) + c_ex + d
function manualInit(T, ivSize, exSize, ivUm, exUm) {
  const c = [ivSize.width / 2, ivSize.height / 2];
  if (!T) return { theta: 0, flip: false, scale: 1, dx: 0, dy: 0, c, cEx: [exSize.width / 2, exSize.height / 2] };
  const [[a, b], [cc, d]] = T.A, flip = a * d - b * cc < 0, L = flip ? [[-a, b], [-cc, d]] : T.A;
  return { theta: Math.round((Math.atan2(L[1][0], L[0][0]) * 180) / Math.PI * 2) / 2, flip, scale: +(scaleOf(T) * exUm / ivUm).toFixed(3), dx: 0, dy: 0, c, cEx: apply(T, c) };
}
function manualAffine(m, ivUm, exUm) {
  const s = (m.scale * ivUm) / exUm, th = (m.theta * Math.PI) / 180, f = m.flip ? -1 : 1;
  const A = [[s * Math.cos(th) * f, -s * Math.sin(th)], [s * Math.sin(th) * f, s * Math.cos(th)]];
  const t = [m.cEx[0] + m.dx / exUm - (A[0][0] * m.c[0] + A[0][1] * m.c[1]), m.cEx[1] + m.dy / exUm - (A[1][0] * m.c[0] + A[1][1] * m.c[1])];
  return { kind: 'manual', A, t };
}

export default function Viewer({ b, session = null, downloads = [] }) {
  const { manifest: M, pose } = b;
  const ivSize = useMemo(() => imageSize(b.iv), [b]), exSize = useMemo(() => imageSize(b.ex), [b]);
  const key = `ivex:${session?.sid || M.title}`;
  const saved0 = useMemo(() => {
    const local = store.get(key, null), server = session?.verify;
    const pick = server && (!local || (server.saved || 0) >= (local.saved || 0)) ? server : local;
    return pick || {};
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const [runIdx, setRunIdx] = useState(() => Math.max(0, M.runs.findIndex((r) => r.name === saved0.run)));
  const run = M.runs[runIdx];
  const [matches, setMatches] = useState(null);
  useEffect(() => { let live = true; setMatches(null); loadRun(b.src, run).then((m) => live && setMatches(m)); return () => { live = false; }; }, [b, run]);

  const [z, setZ] = useState(M.iv.default_z ?? 0);
  const [ivContrast, setIvContrast] = useState([0, 1000]);
  const [exContrast, setExContrast] = useState(M.ex.channels.map(() => [0, 1000]));
  const [exVisible, setExVisible] = useState(M.ex.channels.map((_, i) => i === 0));
  const [maskVis, setMaskVis] = useState(() => ({
    iv: Object.fromEntries(M.iv.mask_sets.map((s, i) => [s.name, s.name === (run.match_model || M.match_model) || (!M.match_model && i === 0)])),
    ex: Object.fromEntries(M.ex.mask_sets.map((s, i) => [s.name, s.name === (run.match_model || M.match_model) || (!M.match_model && i === 0)])),
  }));
  const [agree, setAgree] = useState({ k: 1, r_um: 5 });
  const [ivView, setIvView] = useState(null), [exView, setExView] = useState(null);
  const [sel, setSel] = useState(null);
  const [rank, setRank] = useState(0);
  // verdicts per run ({ run: { 'z:id': verdict } }); older saves were one flat map for a single run
  const [verdictsAll, setVerdictsAll] = useState(() => (Object.values(saved0.verdicts || {})[0]?.status ? { [M.runs[0].name]: saved0.verdicts } : saved0.verdicts || {}));
  const verdicts = verdictsAll[run.name] || {};
  const setVerdicts = useCallback((fn) => setVerdictsAll((all) => ({ ...all, [run.name]: fn(all[run.name] || {}) })), [run.name]);
  const [poseOk, setPoseOk] = useState(saved0.pose_ok ?? null);
  const [mode, setMode] = useState('side');
  const [ovl, setOvl] = useState({ style: 'blend', amount: 0.6, cell: 64 });
  const [link, setLink] = useState(true);
  const [regKind, setRegKind] = useState('similarity');
  const [scope, setScope] = useState('slice');
  const [dz, setDz] = useState(9);
  const [source, setSource] = useState('auto');
  const [filters, setFilters] = useState({ planeOnly: true, minP: 0, mutual: false, z0: 0, z1: M.iv.nz - 1 });
  const [showInliers, setShowInliers] = useState(false);
  const [msg, setMsg] = useState(null);
  const [busy, setBusy] = useState(false);
  const runT = useMemo(() => affineOf(run.iv_to_ex_native), [run]);
  const [manual, setManual] = useState(() => ({ on: false, ...manualInit(runT, ivSize, exSize, M.iv.um_xy, M.ex.um_xy) }));
  useEffect(() => { const init = manualInit(runT, ivSize, exSize, M.iv.um_xy, M.ex.um_xy); setManual({ ...init, init, on: false }); setSel(null); }, [runT, ivSize, exSize, M]);
  const manualT = useMemo(() => (manual.on ? manualAffine(manual, M.iv.um_xy, M.ex.um_xy) : null), [manual, M]);

  useEffect(() => { autoContrast(b.iv, { z, c: 0, t: 0 }).then(setIvContrast); }, [b, z]);
  useEffect(() => { Promise.all(M.ex.channels.map((_, c) => autoContrast(b.ex, { z: 0, c, t: 0 }))).then(setExContrast); }, [b, M]);
  useEffect(() => {
    const w = (window.innerWidth - 660) / 2, h = window.innerHeight - 60;
    setIvView(fitView({ width: w, height: h }, ivSize)); setExView(fitView({ width: w, height: h }, exSize));
  }, [ivSize, exSize]);

  // --- persistence: localStorage always; the server (verify.json) in a session, throttled (each save is one Blob write)
  const [saveState, setSaveState] = useState({ dirty: false, at: saved0.saved || null, error: null });
  const doc = useMemo(() => ({ verdicts: verdictsAll, pose_ok: poseOk, run: run.name }), [verdictsAll, poseOk, run.name]);
  const first = useRef(true);
  useEffect(() => {
    store.set(key, { ...doc, saved: Date.now() / 1000 });
    if (first.current) { first.current = false; return; }
    setSaveState((s) => ({ ...s, dirty: true }));
  }, [doc, key]);
  const saveNow = useCallback(async () => {
    if (!session) return;
    try { const r = await session.save(doc); setSaveState({ dirty: false, at: r.saved, error: null }); } catch (e) { setSaveState((s) => ({ ...s, error: String(e.message || e) })); }
  }, [session, doc]);
  useEffect(() => { if (!session || !saveState.dirty) return; const t = setTimeout(saveNow, 30000); return () => clearTimeout(t); }, [session, saveState.dirty, saveNow]);
  useEffect(() => {
    if (!session) return;
    const onHide = () => { if (saveState.dirty) navigator.sendBeacon('/api/verify', new Blob([JSON.stringify({ sid: session.sid, ...doc })], { type: 'application/json' })); };
    window.addEventListener('pagehide', onHide); return () => window.removeEventListener('pagehide', onHide);
  }, [session, saveState.dirty, doc]);

  // --- accepted pairs -> fits (micrometres), slice-wise around the current plane
  const pairs = useMemo(() => (!matches ? [] : Object.entries(verdicts).filter(([, v]) => v.status === 'accepted').map(([k, v]) => {
    const c = matches.byKey.get(k), e = matches.ex[v.ex];
    return c && e ? { key: k, z: c.z, iv: [c.x * M.iv.um_xy, c.y * M.iv.um_xy], ex: [e.x * M.ex.um_xy, e.y * M.ex.um_xy], cell: c, exCell: e } : null;
  }).filter(Boolean)), [verdicts, matches, M]);
  const reg = useMemo(() => {
    const use = scope === 'slice' ? pairs.filter((p) => Math.abs(p.z - z) * (M.iv.um_z || 1) <= dz) : pairs;
    const fallback = scope === 'slice' && use.length < MIN_PAIRS[regKind];
    const P = fallback ? pairs : use;
    const T = fit(regKind, P.map((p) => p.iv), P.map((p) => p.ex));
    if (!T) return { n: P.length, fallback };
    const res = residuals(T, P.map((p) => p.iv), P.map((p) => p.ex));
    const loo = looErrors(regKind, P.map((p) => p.iv), P.map((p) => p.ex));
    return { n: P.length, fallback, T, rms: Math.sqrt(res.reduce((s, r) => s + r * r, 0) / res.length), loo: median(loo),
      all: P.map((p, i) => ({ key: p.key, r: res[i], loo: loo[i], p })).sort((a, c) => c.r - a.r) };
  }, [pairs, regKind, scope, dz, z, M]);
  const fitT = useMemo(() => (reg.T ? toNative(linear(reg.T), M.iv.um_xy, M.ex.um_xy) : null), [reg, M]);
  const ovT = manualT || fitT;                     // overlay only from verified pairs (or while setting the orientation by hand)
  const canOverlay = !!ovT;
  useEffect(() => { if (!canOverlay && mode === 'overlay') setMode('side'); }, [canOverlay, mode]);
  const T = useMemo(() => {                       // native IV px -> EX px (linear part for TPS): view linking
    if (manualT) return manualT;
    if (source === 'none') return null;
    if ((source === 'fit' || source === 'auto') && reg.T) return toNative(linear(reg.T), M.iv.um_xy, M.ex.um_xy);
    return source !== 'fit' ? runT : null;
  }, [manualT, source, reg, runT, M]);

  // --- linked views
  const onIvView = useCallback((v) => {
    setIvView(v);
    if (link && T && mode === 'side') { const p = apply(T, v.target); setExView({ target: [p[0], p[1], 0], zoom: v.zoom - Math.log2(scaleOf(T)) }); }
  }, [link, T, mode]);
  const onExView = useCallback((v) => {
    setExView(v);
    if (link && T && mode === 'side') { const p = apply(invert(T), v.target); setIvView({ target: [p[0], p[1], 0], zoom: v.zoom + Math.log2(scaleOf(T)) }); }
  }, [link, T, mode]);

  // --- mask agreement on this plane (all sets of a side, whether shown or not)
  const [only, setOnly] = useState({ iv: null, ex: null });
  useEffect(() => {
    if (agree.k <= 1) { setOnly({ iv: null, ex: null }); return; }
    let live = true;
    const side = (s, zz, um) => Promise.all(M[s].mask_sets.filter((m) => hasPlane(m, zz)).map((m) => getOutlines(b.src, m, zz).then((r) => [m.name, r])))
      .then((pp) => agreement(Object.fromEntries(pp), agree.r_um / um, agree.k));
    Promise.all([side('iv', z, M.iv.um_xy), side('ex', 0, M.ex.um_xy)]).then(([iv, ex]) => live && setOnly({ iv, ex }));
    return () => { live = false; };
  }, [agree, z, M, b]);

  // --- verification queue
  const queue = useMemo(() => {
    if (!matches) return [];
    const base = filters.planeOnly ? matches.byPlane.get(z) || [] : matches.cells.filter((c) => c.z >= filters.z0 && c.z <= filters.z1);
    return base.filter((c) => c.p >= filters.minP && (!filters.mutual || c.mutual)).slice().sort((a, c) => c.p - a.p);
  }, [matches, z, filters]);
  const done = queue.filter((c) => verdicts[`${c.z}:${c.id}`]).length;
  const cand = sel ? sel.combinedIdx : [];
  const chosen = sel ? (typeof rank === 'object' ? rank.manual : cand[rank]) : null;

  const select = useCallback((c, centerEx = true) => {
    setSel(c); setRank(0);
    if (c && c.z !== z) setZ(c.z);
    if (c && centerEx && matches) { const e = matches.ex[c.combinedIdx[0]]; setExView((v) => v && ({ ...v, target: [e.x, e.y, 0] })); }
  }, [matches, z]);
  // every verdict change is recorded so it can be undone (Z / Ctrl+Z): { key, prev } per change, newest last
  const history = useRef([]);
  const [undoCount, setUndoCount] = useState(0);
  const record = (key) => { history.current.push({ run: run.name, key, prev: verdicts[key] }); setUndoCount(history.current.length); };
  const verdict = useCallback((status) => {
    if (!sel) return;
    const key = `${sel.z}:${sel.id}`; record(key);
    setVerdicts((v) => ({ ...v, [key]: { status, ex: chosen, rank: typeof rank === 'object' ? null : rank, t: Date.now() } }));
    const i = queue.indexOf(sel), next = queue.slice(i + 1).find((c) => !verdicts[`${c.z}:${c.id}`]);
    if (next) select(next);
  }, [sel, chosen, rank, queue, verdicts, select, setVerdicts]); // eslint-disable-line react-hooks/exhaustive-deps
  const clearVerdict = useCallback((key) => {
    if (!verdicts[key]) return;
    record(key); setVerdicts((v) => { const o = { ...v }; delete o[key]; return o; });
  }, [verdicts, setVerdicts]); // eslint-disable-line react-hooks/exhaustive-deps
  const undo = useCallback(() => {
    const h = history.current.filter((x) => x.run === run.name).pop(); if (!h) return;
    history.current.splice(history.current.lastIndexOf(h), 1); setUndoCount(history.current.length);
    setVerdicts((v) => { const o = { ...v }; if (h.prev) o[h.key] = h.prev; else delete o[h.key]; return o; });
    const c = matches?.byKey.get(h.key); if (c) select(c);
  }, [run.name, setVerdicts, matches, select]);
  const step = useCallback((d) => {
    if (!queue.length) return;
    const i = sel ? queue.indexOf(sel) : -1; select(queue[(i + d + queue.length) % queue.length]);
  }, [queue, sel, select]);

  useEffect(() => {
    const onKey = (e) => {
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
      const k = e.key.toLowerCase();
      if (k === 'z' || ((e.ctrlKey || e.metaKey) && k === 'z')) undo();
      else if ((k === 'backspace' || k === 'delete') && sel) clearVerdict(`${sel.z}:${sel.id}`);
      else if (k === 'a') verdict('accepted'); else if (k === 'r') verdict('rejected'); else if (k === 'u') verdict('unsure');
      else if (k === 'n' || k === 'arrowright') step(1); else if (k === 'p' || k === 'arrowleft') step(-1);
      else if (k === 'o') setMode((m) => (m === 'side' && canOverlay ? 'overlay' : 'side'));
      else if (/^[1-9]$/.test(k) && sel && Number(k) <= cand.length) setRank(Number(k) - 1);
      else if (k === 'arrowup') setZ((v) => Math.min(M.iv.nz - 1, v + 1)); else if (k === 'arrowdown') setZ((v) => Math.max(0, v - 1));
      else return;
      e.preventDefault();
    };
    window.addEventListener('keydown', onKey); return () => window.removeEventListener('keydown', onKey);
  }, [verdict, step, sel, cand, M, canOverlay, undo, clearVerdict]);

  const onIvClick = ({ x, y }) => {
    const plane = matches?.byPlane.get(z) || [];
    let best = null, bd = 15;
    for (const c of plane) { const d = Math.hypot(c.x - x, c.y - y); if (d < bd) { bd = d; best = c; } }
    if (best) select(best, true);
  };
  const onExClick = ({ x, y }) => {
    if (!sel || !matches) return;
    const k = cand.findIndex((i) => Math.hypot(matches.ex[i].x - x, matches.ex[i].y - y) < 25);
    if (k >= 0) { setRank(k); return; }
    let best = null, bd = 25;
    for (const e of matches.ex) { const d = Math.hypot(e.x - x, e.y - y); if (d < bd) { bd = d; best = e; } }
    if (best) setRank({ manual: best.i });
  };

  // --- rematch (sessions)
  const doRematch = async (body, what) => {
    if (!session) return; setBusy(true); setMsg(null);
    try { await session.rematch(body); setMsg({ ok: true, text: `${what} queued on the HPC worker. This page updates when it is done.` }); }
    catch (e) { setMsg({ ok: false, text: String(e.message || e) }); } finally { setBusy(false); }
  };

  // --- layers' inputs
  const ivSel = useMemo(() => [{ z, c: 0, t: 0 }], [z]);
  const exSel = useMemo(() => M.ex.channels.map((_, c) => ({ z: 0, c, t: 0 })), [M]);
  const exColors = useMemo(() => M.ex.channels.map((_, c) => [[255, 0, 255], [60, 120, 255]][c] || [255, 255, 255]), [M]);      // EX magenta (DAPI blue), IV green
  const acceptedHere = pairs.filter((p) => p.z === z);
  const matchSet = matches?.iv_mask_set;
  const ivHighlight = useMemo(() => !matches ? [] : [
    { set: matchSet, key: 'acc', ids: new Set(acceptedHere.map((p) => p.cell.id)), color: GREEN, width: 2 },
    ...(sel && sel.z === z ? [{ set: matchSet, key: 'sel', ids: new Set([sel.id]), color: [255, 255, 255, 255], width: 3 }] : []),
  ], [matches, matchSet, acceptedHere, sel, z]);
  const exHighlight = useMemo(() => (!matches ? [] : cand.slice(0, 3).map((i, k) => ({ set: matches.ex[i].set, key: `c${k}`, ids: new Set([matches.ex[i].maskId]), color: rankColor(k), width: k === rank ? 4 : 2 }))), [cand, matches, rank]);
  const inl = pose?.inlier_pairs;
  const exMarkers = useMemo(() => !matches ? [] : [
    ...cand.map((i, k) => ({ x: matches.ex[i].x, y: matches.ex[i].y, color: rankColor(k), label: String(k + 1), r: k === rank ? 10 : 6, w: k === rank ? 3 : 1.5 })),
    ...(typeof rank === 'object' ? [{ x: matches.ex[rank.manual].x, y: matches.ex[rank.manual].y, color: [0, 255, 255, 255], label: 'pick', r: 10, w: 3 }] : []),
    ...pairs.map((p) => ({ x: p.exCell.x, y: p.exCell.y, color: GREEN, r: 4, w: 2 })),
    ...(sel && sel.truth != null ? [{ x: matches.ex[sel.truth].x, y: matches.ex[sel.truth].y, color: [255, 255, 255, 255], label: 'true', r: 14, w: 1.5 }] : []),
    ...(showInliers && inl ? inl.ex.map(([x, y]) => ({ x, y, color: CYAN, r: 5, w: 1.5 })) : []),
  ], [cand, matches, rank, pairs, showInliers, inl, sel]);
  const ivMarkers = useMemo(() => [
    ...(sel && sel.z === z ? [{ x: sel.x, y: sel.y, color: [255, 255, 255, 255], r: 12, w: 2 }] : []),
    ...(showInliers && inl ? inl.iv.filter((p) => Math.abs(p[2] - z) <= 1).map(([x, y]) => ({ x, y, color: CYAN, r: 5, w: 1.5 })) : []),
  ], [sel, z, showInliers, inl]);
  const exLines = useMemo(() => {
    if (mode !== 'overlay' || !ovT || !matches) return [];
    const out = [];
    if (sel) { const p = apply(ovT, [sel.x, sel.y]); cand.slice(0, 3).forEach((i, k) => out.push({ from: [p[0], p[1]], to: [matches.ex[i].x, matches.ex[i].y], color: rankColor(k), w: k === rank ? 2.5 : 1.2 })); }
    if (showInliers && inl) inl.iv.forEach((q, i) => { const p = apply(ovT, q); out.push({ from: [p[0], p[1]], to: inl.ex[i], color: CYAN, w: 1 }); });
    return out;
  }, [mode, ovT, sel, cand, matches, rank, showInliers, inl]);
  const ivMatrix = useMemo(() => (ovT ? modelMatrix(ovT) : null), [ovT]);
  const counts = Object.values(verdicts).reduce((m, v) => ({ ...m, [v.status]: (m[v.status] || 0) + 1 }), {});

  // --- exports
  const exportJSON = () => {
    const out = { bundle: M.title, session: session?.sid || null, run: run.name, match_model: run.match_model || M.match_model, exported: new Date().toISOString(),
      pose_ok: poseOk, units: { iv_um_xy: M.iv.um_xy, iv_um_z: M.iv.um_z, ex_um_xy: M.ex.um_xy, coordinates: 'native pixels, x right, y down' },
      pairs: pairs.map((p) => ({ iv: { z: p.cell.z, id: p.cell.id, x: p.cell.x, y: p.cell.y }, ex: { index: p.exCell.i, x: p.exCell.x, y: p.exCell.y, mask_set: p.exCell.set, mask_id: p.exCell.maskId } })),
      verdicts, registration: reg.T ? { kind: regKind, scope, dz_um: dz, z, rms_um: reg.rms, loo_median_um: reg.loo, n: reg.n, ...(reg.T.kind === 'tps' ? {} : { A: reg.T.A, t_um: reg.T.t }) } : null };
    download('ivex_verification.json', JSON.stringify(out, null, 1), 'application/json');
  };
  const exportCSV = () => {
    const rows = [['iv_z', 'iv_id', 'iv_x_px', 'iv_y_px', 'iv_x_um', 'iv_y_um', 'ex_mask_set', 'ex_id', 'ex_x_px', 'ex_y_px', 'ex_x_um', 'ex_y_um', 'p_top1', 'rank']];
    for (const [k, v] of Object.entries(verdicts)) {
      if (v.status !== 'accepted' || !matches) continue; const c = matches.byKey.get(k), e = matches.ex[v.ex]; if (!c || !e) continue;
      rows.push([c.z, c.id, c.x, c.y, (c.x * M.iv.um_xy).toFixed(2), (c.y * M.iv.um_xy).toFixed(2), e.set, e.maskId, e.x, e.y, (e.x * M.ex.um_xy).toFixed(2), (e.y * M.ex.um_xy).toFixed(2), c.p, v.rank ?? 'manual']);
    }
    download('ivex_pairs.csv', rows.map((r) => r.join(',')).join('\n'), 'text/csv');
  };
  const exportTransforms = () => {
    const out = { bundle: M.title, session: session?.sid || null, exported: new Date().toISOString(), iv_um_xy: M.iv.um_xy, ex_um_xy: M.ex.um_xy, iv_um_z: M.iv.um_z,
      ...sliceFits(regKind, pairs, M.iv.nz, M.iv.um_z, dz) };
    download('ivex_transforms.json', JSON.stringify(out, null, 1), 'application/json');
  };

  const masksPanel = (side) => M[side].mask_sets.map((s) => (
    <label key={s.name} className="row">
      <input type="checkbox" checked={!!maskVis[side][s.name]} onChange={(e) => setMaskVis((v) => ({ ...v, [side]: { ...v[side], [s.name]: e.target.checked } }))} />
      <span className="swatch" style={{ background: `rgb(${(s.color || [255, 255, 0]).join(',')})` }} />
      <span title={s.name}>{s.label}</span>
      {only[side]?.[s.name] && <span className="hint">· {only[side][s.name].size}</span>}
    </label>
  ));
  const isConfident = (c) => c.p >= 0.5 && c.mutual;
  const setF = (k) => (e) => setFilters((f) => ({ ...f, [k]: e.target.type === 'checkbox' ? e.target.checked : Number(e.target.value) }));

  if (!matches) return <div className="splash"><div><div className="spinner" />Loading matches ({run.label})…</div></div>;
  const zDock = (
    <div className="zdock">
      <span className="hint">IV plane</span>
      <input type="range" min={0} max={M.iv.nz - 1} value={z} onChange={(e) => setZ(Number(e.target.value))} aria-label="IV plane" />
      <b>z {z}</b><span className="hint">{M.iv.um_z ? `${(z * M.iv.um_z).toFixed(0)} µm · ` : ''}{(matches.byPlane.get(z) || []).length} cells</span>
    </div>
  );
  const saveText = !session ? (M.example ? 'example · saved in this browser' : 'saved in this browser')
    : saveState.error ? `not saved: ${saveState.error}` : saveState.dirty ? 'unsaved changes' : saveState.at ? `saved ${new Date(saveState.at * 1000).toLocaleTimeString()}` : 'nothing to save yet';
  return (
    <div className="app">
      <header className="vbar">
        <a className="brand" href="./" title="ivex home" />
        <span className="title" title={M.title}>{M.title}</span>
        {M.example && <span className="pill">example</span>}
        <span className="sep" />
        <div className="segctl">{['side', 'overlay'].map((m) => <button key={m} className={mode === m ? 'on' : ''} disabled={m === 'overlay' && !canOverlay}
          title={m === 'overlay' && !canOverlay ? `Accept at least ${MIN_PAIRS[regKind]} pairs for a ${regKind} fit first` : ''} onClick={() => setMode(m)}>{m === 'side' ? 'Side by side' : 'Overlay'}</button>)}</div>
        {!canOverlay && <span className="hint nowrap">overlay: {pairs.length}/{MIN_PAIRS[regKind]} pairs</span>}
        {mode === 'overlay' && (
          <>
            <div className="segctl">{['blend', 'swipe', 'checker'].map((s) => <button key={s} className={ovl.style === s ? 'on' : ''} onClick={() => setOvl((o) => ({ ...o, style: s, amount: s === 'swipe' ? 0.5 : o.amount }))}>{s === 'checker' ? 'checkerboard' : s}</button>)}</div>
            {ovl.style !== 'checker'
              ? <input style={{ width: 110 }} type="range" min={0} max={1} step={0.01} value={ovl.amount} onChange={(e) => setOvl((o) => ({ ...o, amount: Number(e.target.value) }))} aria-label={ovl.style === 'blend' ? 'IV opacity' : 'divider'} />
              : <input style={{ width: 110 }} type="range" min={16} max={256} step={8} value={ovl.cell} onChange={(e) => setOvl((o) => ({ ...o, cell: Number(e.target.value) }))} aria-label="square size" />}
          </>
        )}
        <label className="row" style={{ margin: 0, whiteSpace: 'nowrap' }}><input type="checkbox" checked={link} onChange={(e) => setLink(e.target.checked)} />link views</label>
        <span className="spacer" />
        <span className={`hint nowrap ${saveState.error ? 'bad' : ''}`} title={saveText}>{!session ? 'saved in this browser' : saveState.error ? 'not saved' : saveState.dirty ? 'unsaved' : 'saved'}</span>
        {session && saveState.dirty && <button onClick={saveNow}><Icon name="save" />Save</button>}
        {downloads.map((d) => <a key={d.href} className="btn ghost" href={d.href} download title={`download the ${d.label}`}><Icon name="download" />{d.label.split(' (')[0]}</a>)}
        <span className="sep" />
        <button onClick={exportCSV} title="accepted pairs (CSV)"><Icon name="download" />Pairs CSV</button>
        <button onClick={exportTransforms} title={`one ${regKind} per IV plane (± ${dz} µm)`}>transforms.json</button>
        <button onClick={exportJSON} title="verdicts, pairs and the current fit">Export JSON</button>
      </header>

      <aside className="panel left">
        <h1>ivex · IV ↔ EX</h1>
        {M.example && <div className="msg ok" style={{ marginTop: 10 }}>{M.example_note || 'Example data.'} Your verdicts stay in this browser.</div>}
        <PoseCard pose={pose} runs={M.runs} runIdx={runIdx} setRunIdx={setRunIdx} poseOk={poseOk} setPoseOk={setPoseOk} manual={manual}
          setManual={(f) => { setManual(f); setMode('overlay'); }} canRematch={!!session && session.canRematch} busy={busy}
          onMatchManual={() => doRematch({ affine: toRows(manualT || manualAffine(manual, M.iv.um_xy, M.ex.um_xy)) }, 'Matching at your orientation')}
          onRunnerUp={(ru) => doRematch({ flip: ru.flip, theta: ru.theta }, 'Matching at the runner-up orientation')} />
        {pose?.inlier_pairs && <label className="row"><input type="checkbox" checked={showInliers} onChange={(e) => setShowInliers(e.target.checked)} />show the pose's inlier pairs (cyan)</label>}
        {msg && <div className={`msg ${msg.ok ? 'ok' : 'bad'}`}>{msg.text}</div>}
        <details className="sec" open>
          <summary>Masks</summary>
          <div className="row"><select style={{ flex: 1 }} value={agree.k} onChange={(e) => setAgree((a) => ({ ...a, k: Number(e.target.value) }))}>
            <option value={1}>Every cell</option><option value={2}>Cells ≥ 2 models agree on</option><option value={3}>Cells all 3 models agree on</option>
          </select></div>
          {agree.k > 1 && <div className="row hint">same cell = centroids within <input className="num" type="number" min={1} max={20} value={agree.r_um} onChange={(e) => setAgree((a) => ({ ...a, r_um: Number(e.target.value) }))} /> µm</div>}
          <div className="sub" style={{ marginTop: 8 }}>In vivo</div>{masksPanel('iv')}
          <div className="sub" style={{ marginTop: 8 }}>Ex vivo</div>{masksPanel('ex')}
          <div className="hint" style={{ marginTop: 6 }}>Matching uses <b>{run.match_model || M.match_model || matchSet}</b>.</div>
          {session && M.models && (
            <div className="row" style={{ marginTop: 6 }}>
              <select style={{ flex: 1 }} disabled={!session.canRematch || busy} value="" onChange={(e) => e.target.value && doRematch({ match_model: e.target.value, ...(run.iv_to_ex_native ? { affine: run.iv_to_ex_native } : {}) }, `Matching with ${e.target.value}`)}>
                <option value="">Re-match with another mask set…</option>{M.models.filter((m) => m !== (run.match_model || M.match_model)).map((m) => <option key={m} value={m}>{m}</option>)}
              </select>
            </div>
          )}
        </details>
        <details className="sec" open>
          <summary>Ex vivo channels</summary>
          {M.ex.channels.map((ch, c) => (
            <label key={ch} className="row"><input type="checkbox" checked={exVisible[c]} onChange={(e) => setExVisible((v) => v.map((x, i) => (i === c ? e.target.checked : x)))} />{ch}</label>
          ))}
          <div className="hint">{M.ex.projection}</div>
        </details>
        <details className="sec">
          <summary>Transform</summary>
          <div className="row"><select style={{ flex: 1 }} value={source} onChange={(e) => setSource(e.target.value)} disabled={!!manualT}>
            <option value="auto">Fit if enough pairs, else pose</option><option value="fit">Fit only</option><option value="pose">Pose only</option><option value="none">None</option>
          </select></div>
          {manualT && <div className="hint warn">Using your manual orientation (Set manually is on).</div>}
        </details>
        <details className="sec">
          <summary>Keyboard</summary>
          <div className="keys">click an IV cell · <b>1–9</b> candidate · click EX to pick<br /><b>A</b> accept · <b>R</b> reject · <b>U</b> unsure<br /><b>Z</b> undo last · <b>⌫</b> clear this cell's verdict<br /><b>N</b>/<b>P</b> next / previous · <b>O</b> overlay · <b>↑</b>/<b>↓</b> plane</div>
        </details>
      </aside>

      <main className={`views ${mode}`}>
        {mode === 'side' && (
          <ImageView title={`In vivo · z ${z}`} src={b.src} image={b.iv} selection={ivSel} contrast={[ivContrast]} colors={[[0, 255, 0]]} channelsVisible={[true]}
            maskSets={M.iv.mask_sets} maskVisible={maskVis.iv} z={z} viewState={ivView} onViewState={onIvView} onClick={onIvClick}
            highlight={ivHighlight} markers={ivMarkers} only={only.iv}>{zDock}</ImageView>
        )}
        <ImageView title={mode === 'overlay' ? `IV z ${z} (green) on EX (magenta) · ${ovl.style}` : 'Ex vivo'} src={b.src} image={b.ex} selection={exSel} contrast={exContrast}
          colors={exColors} channelsVisible={exVisible} maskSets={M.ex.mask_sets} maskVisible={maskVis.ex} viewState={exView} onViewState={onExView}
          onClick={onExClick} highlight={exHighlight} markers={exMarkers} lines={exLines} only={only.ex}>
          {mode === 'overlay' && ovT && <OverlayCanvas image={b.iv} selection={ivSel} contrast={[ivContrast]} colors={[[0, 255, 0]]} matrix={ivMatrix} viewState={exView} mode={ovl.style} amount={ovl.amount} cell={ovl.cell} />}
          {mode === 'overlay' && ovT && ovl.style === 'swipe' && <div className="swipe-line" style={{ left: `${(ovl.amount * 100).toFixed(1)}%` }} />}
          {mode === 'overlay' && reg.T?.kind === 'tps' && !manualT && <div className="view-note">The overlay shows the affine part of the TPS.</div>}
          {mode === 'overlay' && zDock}
        </ImageView>
      </main>

      <aside className="panel right">
        <section>
          <h2>Verification</h2>
          <div className="counts sr" aria-live="polite">accepted <b>{counts.accepted || 0}</b> · rejected <b>{counts.rejected || 0}</b> · unsure <b>{counts.unsure || 0}</b></div>
          <div className="stat3"><div className="a"><b>{counts.accepted || 0}</b><span>accepted</span></div><div className="r"><b>{counts.rejected || 0}</b><span>rejected</span></div><div className="u"><b>{counts.unsure || 0}</b><span>unsure</span></div></div>
          <div className="progress" title={`${done} of ${queue.length} in this queue`}><div style={{ width: `${queue.length ? (100 * done) / queue.length : 0}%` }} /></div>
          <div className="row" style={{ justifyContent: 'space-between' }}><span className="hint">{done} / {queue.length} in queue · {matches.cells.length} queries</span>
            <button onClick={undo} disabled={!history.current.some((x) => x.run === run.name)} title="Undo the last verdict (Z)" data-n={undoCount}>Undo</button>
            <span className="seg" style={{ margin: 0 }}><button onClick={() => step(-1)} title="previous (P)">◀</button><button onClick={() => step(1)} title="next (N)">Next ▶</button></span></div>
          <details className="sec" style={{ borderBottom: 0, paddingBottom: 0 }}>
            <summary>Queue filters</summary>
            <label className="row"><input type="checkbox" checked={filters.planeOnly} onChange={setF('planeOnly')} />this plane only</label>
            {!filters.planeOnly && <div className="row">planes <input className="num" type="number" min={0} max={M.iv.nz - 1} value={filters.z0} onChange={setF('z0')} /> to <input className="num" type="number" min={0} max={M.iv.nz - 1} value={filters.z1} onChange={setF('z1')} /></div>}
            <div className="row">P(top1) ≥ <input type="range" min={0} max={0.95} step={0.05} value={filters.minP} onChange={setF('minP')} /><b>{(filters.minP * 100).toFixed(0)}%</b></div>
            <label className="row"><input type="checkbox" checked={filters.mutual} onChange={setF('mutual')} />mutual best only</label>
          </details>
        </section>
        {sel ? (
          <section>
            <h2>IV cell {sel.id} · z {sel.z} {isConfident(sel) && <span className="badge">confident</span>}</h2>
            <div className="hint">({sel.x.toFixed(0)}, {sel.y.toFixed(0)}) px · P(top1) {(100 * sel.p).toFixed(1)}% {sel.mutual ? '· mutual best' : ''}
            </div>
            {verdicts[`${sel.z}:${sel.id}`] && (
              <div className={`msg ${verdicts[`${sel.z}:${sel.id}`].status === 'accepted' ? 'ok' : verdicts[`${sel.z}:${sel.id}`].status === 'rejected' ? 'bad' : ''}`} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ flex: 1 }}>Marked <b>{verdicts[`${sel.z}:${sel.id}`].status}</b>{verdicts[`${sel.z}:${sel.id}`].status === 'accepted' && matches.ex[verdicts[`${sel.z}:${sel.id}`].ex] ? ` with EX ${matches.ex[verdicts[`${sel.z}:${sel.id}`].ex].maskId}` : ''}. Choose again to change it.</span>
                <button onClick={() => clearVerdict(`${sel.z}:${sel.id}`)} title="Backspace">Clear</button>
              </div>
            )}
            <div className="seg verdict">
              <button className="acc" onClick={() => verdict('accepted')}><Icon name="check" />Accept (A)</button>
              <button className="rej" onClick={() => verdict('rejected')}><Icon name="x" />Reject (R)</button>
              <button onClick={() => verdict('unsure')}>Unsure (U)</button>
            </div>
            <CropCards b={b} sel={sel} cand={cand} matches={matches} ivContrast={ivContrast} exContrast={exContrast} rank={typeof rank === 'number' ? rank : -1} onPick={setRank} pct={pct} />
            {typeof rank === 'object' && <div className="hint">Manual pick: EX {matches.ex[rank.manual].set} {matches.ex[rank.manual].maskId}</div>}
            {sel.truth != null && <div className={`msg ${cand[0] === sel.truth ? 'ok' : 'bad'}`}>{cand[0] === sel.truth ? 'PMAC ranks the annotated match first.'
              : cand.includes(sel.truth) ? `The annotated match is PMAC's #${cand.indexOf(sel.truth) + 1}.` : "The annotated match is not in PMAC's top 10."}</div>}
            <div className="hint" style={{ marginTop: 6 }}>PMAC's scores are suggestions; nothing is accepted for you.</div>
          </section>
        ) : <section><div className="empty"><div><Icon name="target" size={22} /><div style={{ marginTop: 6 }}>Click an in vivo cell, or press <kbd>N</kbd>,<br />to see PMAC's candidates.</div></div></div></section>}
        <section>
          <h2>Registration</h2>
          <div className="row">
            <select value={regKind} onChange={(e) => setRegKind(e.target.value)}><option value="similarity">similarity</option><option value="affine">affine</option><option value="tps">thin-plate spline</option></select>
            <select value={scope} onChange={(e) => setScope(e.target.value)}><option value="slice">slice-wise</option><option value="global">global</option></select>
            {scope === 'slice' && <>± <input className="num" type="number" value={dz} min={0} onChange={(e) => setDz(Number(e.target.value))} /> µm</>}
          </div>
          {reg.T ? (
            <div className="reg">
              <div>{reg.n} pairs{reg.fallback ? ' (too few near this plane → global)' : ''}{reg.T.kind !== 'tps' ? ` · ${isMirrored(reg.T) ? 'mirrored, ' : ''}rot ${rotationOf(reg.T).toFixed(1)}° · scale ${scaleOf(reg.T).toFixed(3)}` : ''}</div>
              <div>RMS <b>{reg.rms.toFixed(1)} µm</b> · leave-one-out median <b>{Number.isFinite(reg.loo) ? `${reg.loo.toFixed(1)} µm` : '–'}</b></div>
              <div className="residuals">
                {reg.all.map((w) => (
                  <div key={w.key} className="res-row" onClick={() => select(w.p.cell)}><span>{w.key}</span><span>{w.r.toFixed(1)} µm</span><span className="hint">LOO {Number.isFinite(w.loo) ? w.loo.toFixed(1) : '–'}</span>
                    <button className="x" title="Remove this pair" onClick={(e) => { e.stopPropagation(); clearVerdict(w.key); }}>×</button></div>
                ))}
              </div>
            </div>
          ) : <div className="hint">Accept at least {MIN_PAIRS[regKind]} pairs to fit ({pairs.length} so far).</div>}
        </section>
      </aside>
    </div>
  );
}
