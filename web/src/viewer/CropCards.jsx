// Crop cards for the selected IV cell: the IV crop, each top candidate's EX crop, and the IV crop warped onto that EX crop through
// the current transform (green IV over magenta EX). Same physical size on both sides (FIELD_UM), level-0 pixels.
import { useEffect, useRef, useState } from 'react';
import { readWindow } from '../lib/data';
import { apply, invert, linear } from '../lib/fit';

const FIELD_UM = 50, PX = 112;

const stretch = (d, [lo, hi]) => (v) => Math.max(0, Math.min(255, ((v - lo) / Math.max(1, hi - lo)) * 255));

function paint(canvas, { gray, green, magenta }) {
  const ctx = canvas.getContext('2d'); const im = ctx.createImageData(PX, PX);
  for (let i = 0; i < PX * PX; i++) {
    const g = gray ? gray[i] : 0, gr = green ? green[i] : 0, m = magenta ? magenta[i] : 0;
    im.data[i * 4] = gray ? g : m; im.data[i * 4 + 1] = gray ? g : gr; im.data[i * 4 + 2] = gray ? g : m; im.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(im, 0, 0);
}

// sample a window (data, w, h, x0, y0) at native coordinates -> Uint8 PX x PX, bilinear
function sampler(win, contrast) {
  const s = stretch(null, contrast);
  return (x, y) => {
    const u = x - win.x0, v = y - win.y0, i = Math.floor(u), j = Math.floor(v);
    if (i < 0 || j < 0 || i + 1 >= win.w || j + 1 >= win.h) return 0;
    const fu = u - i, fv = v - j, d = win.data, w = win.w;
    return s(d[j * w + i] * (1 - fu) * (1 - fv) + d[j * w + i + 1] * fu * (1 - fv) + d[(j + 1) * w + i] * (1 - fu) * fv + d[(j + 1) * w + i + 1] * fu * fv);
  };
}

function Card({ label, draw, deps }) {
  const ref = useRef(null);
  useEffect(() => { if (ref.current) draw(ref.current); }, deps); // eslint-disable-line react-hooks/exhaustive-deps
  return <figure className="crop"><canvas ref={ref} width={PX} height={PX} /><figcaption>{label}</figcaption></figure>;
}

export default function CropCards({ b, sel, cand, matches, T, ivContrast, exContrast, rank, onPick }) {
  const [wins, setWins] = useState(null);
  const M = b.manifest, ivUm = M.iv.um_xy, exUm = M.ex.um_xy;
  useEffect(() => {
    if (!sel) return; let live = true;
    const ivSide = Math.ceil(FIELD_UM / ivUm) + 8, exSide = Math.ceil(FIELD_UM / exUm) + 8, top = cand.slice(0, 3).map((i) => matches.ex[i]);
    Promise.all([readWindow(b.iv, { z: sel.z, c: 0, t: 0 }, sel.x, sel.y, ivSide * 2, ivSide * 2),
      ...top.map((e) => readWindow(b.ex, { z: 0, c: 0, t: 0 }, e.x, e.y, exSide, exSide))])
      .then(([iv, ...ex]) => { if (live) setWins({ iv, ex, top, key: `${sel.z}:${sel.id}:${cand.slice(0, 3).join(',')}` }); })
      .catch((e) => console.error(e));
    return () => { live = false; };
  }, [b, sel, cand, matches, ivUm, exUm]);
  if (!sel) return null;
  if (!wins) return <div className="hint">Loading crops…</div>;
  const ivS = sampler(wins.iv, ivContrast), Tl = linear(T), Ti = Tl ? invert(Tl) : null;
  const grid = (cx, cy, um) => (fn) => { const out = new Uint8ClampedArray(PX * PX), step = um / PX;
    for (let j = 0; j < PX; j++) for (let i = 0; i < PX; i++) out[j * PX + i] = fn(cx + (i - PX / 2) * step, cy + (j - PX / 2) * step); return out; };
  const ivCrop = grid(sel.x, sel.y, FIELD_UM / ivUm)(ivS);
  return (
    <div className="crops">
      <Card label={`IV ${sel.id} · z ${sel.z}`} deps={[wins.key, ivContrast]} draw={(c) => paint(c, { gray: ivCrop })} />
      {wins.top.map((e, k) => {
        const exS = sampler(wins.ex[k], exContrast[0] || [0, 255]); const exCrop = grid(e.x, e.y, FIELD_UM / exUm)(exS);
        // overlay: for every EX pixel, the IV pixel the transform sends there (only when a transform exists)
        const warped = Ti ? grid(e.x, e.y, FIELD_UM / exUm)((x, y) => { const p = apply(Ti, [x, y]); return ivS(p[0], p[1]); }) : null;
        const dist = Tl ? Math.hypot(...((p) => [p[0] - e.x, p[1] - e.y])(apply(Tl, [sel.x, sel.y]))) * exUm : null;
        return (
          <div key={e.i} className={`cardpair ${k === rank ? 'on' : ''}`} onClick={() => onPick(k)}>
            <Card label={`#${k + 1} EX ${e.maskId}`} deps={[wins.key, exContrast, k]} draw={(c) => paint(c, { gray: exCrop })} />
            <Card label={warped ? `overlay · ${dist.toFixed(0)} µm off` : 'no transform yet'} deps={[wins.key, exContrast, ivContrast, T, k]}
              draw={(c) => paint(c, warped ? { green: warped, magenta: exCrop } : { gray: new Uint8ClampedArray(PX * PX) })} />
          </div>
        );
      })}
      <div className="hint">{FIELD_UM} µm fields. Overlay: IV (green) through the current transform on EX (magenta).</div>
    </div>
  );
}
