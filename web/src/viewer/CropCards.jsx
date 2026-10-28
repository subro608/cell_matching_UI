// Candidate panel for the selected IV cell: the IV query's soma and neighbourhood crops, then every top-10 EX candidate with its own
// soma and neighbourhood crops, PMAC score and rank in each term. IV is drawn green, EX magenta (as in the views). Click a row to pick.
import { useEffect, useRef, useState } from 'react';
import { readWindow } from '../lib/data';

export const SOMA_UM = 50, NBHD_UM = 250;                 // physical fields, roughly PMAC's soma (40 px) and neighbourhood (150 px) crops
const PX = 72;
const GREEN = [0, 255, 0], MAGENTA = [255, 0, 255];

function sample(win, contrast) {
  const [lo, hi] = contrast, k = 255 / Math.max(1, hi - lo);
  return (x, y) => {
    const u = x - win.x0, v = y - win.y0, i = Math.floor(u), j = Math.floor(v);
    if (i < 0 || j < 0 || i + 1 >= win.w || j + 1 >= win.h) return 0;
    const fu = u - i, fv = v - j, d = win.data, w = win.w;
    const val = d[j * w + i] * (1 - fu) * (1 - fv) + d[j * w + i + 1] * fu * (1 - fv) + d[(j + 1) * w + i] * (1 - fu) * fv + d[(j + 1) * w + i + 1] * fu * fv;
    return Math.max(0, Math.min(255, (val - lo) * k));
  };
}

function Crop({ win, contrast, cx, cy, fieldPx, color, label, ring }) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current; if (!c || !win) return;
    const ctx = c.getContext('2d'), im = ctx.createImageData(PX, PX), f = sample(win, contrast), step = fieldPx / PX;
    for (let j = 0; j < PX; j++) for (let i = 0; i < PX; i++) {
      const v = f(cx + (i - PX / 2) * step, cy + (j - PX / 2) * step) / 255, o = (j * PX + i) * 4;
      im.data[o] = color[0] * v; im.data[o + 1] = color[1] * v; im.data[o + 2] = color[2] * v; im.data[o + 3] = 255;
    }
    ctx.putImageData(im, 0, 0);
    if (ring) { ctx.strokeStyle = '#ffffffaa'; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(PX / 2, PX / 2, (ring / step) / 2, 0, 2 * Math.PI); ctx.stroke(); }
  }, [win, contrast, cx, cy, fieldPx, color, ring]);
  return <figure className="crop"><canvas ref={ref} width={PX} height={PX} />{label && <figcaption>{label}</figcaption>}</figure>;
}

export default function CropCards({ b, sel, cand, matches, ivContrast, exContrast, rank, onPick, pct }) {
  const [wins, setWins] = useState(null);
  const M = b.manifest, ivUm = M.iv.um_xy, exUm = M.ex.um_xy;
  useEffect(() => {
    if (!sel) return; let live = true; setWins(null);
    const ivSide = Math.ceil(NBHD_UM / ivUm) + 8, exSide = Math.ceil(NBHD_UM / exUm) + 8, top = cand.slice(0, 10).map((i) => matches.ex[i]);
    Promise.all([readWindow(b.iv, { z: sel.z, c: 0, t: 0 }, sel.x, sel.y, ivSide, ivSide), ...top.map((e) => readWindow(b.ex, { z: 0, c: 0, t: 0 }, e.x, e.y, exSide, exSide))])
      .then(([iv, ...ex]) => { if (live) setWins({ iv, ex, top }); }).catch((e) => console.error(e));
    return () => { live = false; };
  }, [b, sel, cand, matches, ivUm, exUm]);
  if (!sel) return null;
  const exC = exContrast[0] || [0, 255];
  return (
    <div className="cands-panel">
      <div className="query">
        <Crop win={wins?.iv} contrast={ivContrast} cx={sel.x} cy={sel.y} fieldPx={SOMA_UM / ivUm} color={GREEN} label="IV soma" />
        <Crop win={wins?.iv} contrast={ivContrast} cx={sel.x} cy={sel.y} fieldPx={NBHD_UM / ivUm} color={GREEN} label="IV neighbourhood" ring={SOMA_UM / ivUm} />
        <div className="hint">query: IV cell {sel.id}, plane {sel.z}<br />soma {SOMA_UM} µm · neighbourhood {NBHD_UM} µm</div>
      </div>
      <div className="cand-head"><span>#</span><span>EX soma</span><span>EX neighbourhood</span><span>PMAC</span></div>
      {(wins?.top || cand.slice(0, 10).map((i) => matches.ex[i])).map((e, k) => {
        const i = cand[k], ai = sel.appearanceIdx.indexOf(i), li = sel.locationIdx.indexOf(i);
        return (
          <div key={e.i} className={`cand-row ${k === rank ? 'on' : ''}`} onClick={() => onPick(k)}>
            <span className="rank">{k + 1}</span>
            <Crop win={wins?.ex[k]} contrast={exC} cx={e.x} cy={e.y} fieldPx={SOMA_UM / exUm} color={MAGENTA} />
            <Crop win={wins?.ex[k]} contrast={exC} cx={e.x} cy={e.y} fieldPx={NBHD_UM / exUm} color={MAGENTA} ring={SOMA_UM / exUm} />
            <div className="cand-meta">
              <b>{pct(sel.combined[k])}</b>
              <span className="hint">EX {e.maskId}</span>
              <span className="hint">appear. {ai >= 0 ? `#${ai + 1}` : '–'} · loc. {li >= 0 ? `#${li + 1}` : '–'}</span>
              {sel.truth === i && <span className="badge">true</span>}
            </div>
          </div>
        );
      })}
    </div>
  );
}
