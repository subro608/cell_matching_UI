// Web bundle loading (layout: core/ivexmatch/webexport.py). All coordinates are native pixels of each side's image.
// A source resolves bundle paths to URLs: a local folder (?data=...) or a session's signed Blob URLs (lib/api.js).
import { loadOmeTiff } from '@hms-dbmi/viv';

export const folderSource = (base) => ({ kind: 'folder', base, resolve: (p) => base + p });
export const signedSource = (urls, prefix) => ({ kind: 'session', resolve: (p) => { const u = urls[prefix + p]; if (!u) throw new Error(`no URL for ${p}`); return u; } });

async function fetchJSON(url, gz) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url.split('?')[0]}: HTTP ${r.status}`);
  if (!gz) return r.json();
  return new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).json();
}
const getJSON = (src, path) => fetchJSON(src.resolve(path), path.endsWith('.gz'));

async function openImage(url) {
  const res = await loadOmeTiff(url);
  return Array.isArray(res) ? res[0] : res; // { data: PixelSource[] (pyramid levels), metadata }
}

// matches.json -> cells with named fields, lookups by "z:id" and by plane
function indexMatches(m) {
  const col = Object.fromEntries(m.columns.map((c, i) => [c, i]));
  const cells = m.cells.map((r, i) => ({
    i, id: r[col.id], z: r[col.z], x: r[col.x], y: r[col.y], p: r[col.p_top1], mutual: r[col.mutual],
    combinedIdx: r[col.combined_idx], combined: r[col.combined],
    appearanceIdx: r[col.appearance_idx], appearance: r[col.appearance],
    locationIdx: r[col.location_idx], location: r[col.location],
    truth: col.truth_idx != null ? r[col.truth_idx] : null,          // annotated match (benchmark examples only)
  }));
  const byKey = new Map(cells.map((c) => [`${c.z}:${c.id}`, c]));
  const byPlane = new Map();
  for (const c of cells) (byPlane.get(c.z) || byPlane.set(c.z, []).get(c.z)).push(c);
  const ex = m.ex.map(([x, y, set, maskId], i) => ({ i, x, y, set: m.ex_mask_sets[set], maskId }));
  return { ...m, cells, byKey, byPlane, ex };
}

// manifest v1 (local JY369 demo: one matches file, per-plane mask files) and v2 (sessions: runs, one file per mask set)
function normalize(manifest, pose) {
  const runs = manifest.runs || [{ name: 'run0', label: 'PMAC matches', kind: 'given', matches: manifest.matches, iv_to_ex_native: pose?.iv_to_ex_native || null }];
  return { ...manifest, runs };
}

export async function loadBundle(src, manifest = null) {
  manifest = manifest || await getJSON(src, 'manifest.json');
  const [pose, iv, ex] = await Promise.all([
    manifest.pose ? getJSON(src, manifest.pose).catch(() => null) : null,
    openImage(src.resolve(manifest.iv.image.file)),
    openImage(src.resolve(manifest.ex.image.file)),
  ]);
  return { src, manifest: normalize(manifest, pose), pose, iv, ex };
}

const runCache = new Map();
export function loadRun(src, run) {
  const key = `${src.kind}:${run.matches}`;
  if (!runCache.has(key)) runCache.set(key, getJSON(src, run.matches).then(indexMatches));
  return runCache.get(key);
}

const rowsToCells = (rows) => rows.map((r) => {
  const polygon = [];
  for (let k = 3; k + 1 < r.length; k += 2) polygon.push([r[k], r[k + 1]]);
  return { id: r[0], cx: r[1], cy: r[2], polygon };
});

const outlineCache = new Map();
// Outlines of one mask set on plane z: [{ id, cx, cy, polygon: [[x, y], ...] }]. v2 sets are one gzipped file (all planes).
export function getOutlines(src, set, z) {
  const key = `${src.kind}:${set.file || set.planes?.[String(z)]}`;
  if (set.file) {
    if (!outlineCache.has(key)) outlineCache.set(key, getJSON(src, set.file).then((all) => new Map(Object.entries(all).map(([k, rows]) => [k, rowsToCells(rows)]))));
    return outlineCache.get(key).then((m) => m.get(String(z)) || []);
  }
  const path = set.planes?.[String(z)];
  if (!path) return Promise.resolve([]);
  if (!outlineCache.has(key)) outlineCache.set(key, getJSON(src, path).then(rowsToCells));
  return outlineCache.get(key);
}
export const hasPlane = (set, z) => (set.file ? (set.counts?.[String(z)] ?? 0) > 0 : !!set.planes?.[String(z)]);

// Contrast limits from the coarsest pyramid level: [p1, p99.8] of the selected plane / channel.
export async function autoContrast(image, selection) {
  const src = image.data[image.data.length - 1];
  const { data } = await src.getRaster({ selection });
  const step = Math.max(1, Math.floor(data.length / 200000));
  const sample = [];
  for (let i = 0; i < data.length; i += step) sample.push(data[i]);
  sample.sort((a, b) => a - b);
  const q = (p) => sample[Math.min(sample.length - 1, Math.floor(p * sample.length))];
  const lo = q(0.01), hi = q(0.998);
  return [lo, Math.max(hi, lo + 1)];
}

export function imageSize(image) {
  const s = image.data[0];
  const shape = s.shape, labels = s.labels;
  return { width: shape[labels.indexOf('x')], height: shape[labels.indexOf('y')] };
}

// A w x h window of level 0 around (cx, cy), composed from the source's tiles: { data: Float32Array, w, h } (0 outside the image).
export async function readWindow(image, selection, cx, cy, w, h) {
  const src = image.data[0], ts = src.tileSize, { width, height } = imageSize(image);
  const x0 = Math.round(cx - w / 2), y0 = Math.round(cy - h / 2), out = new Float32Array(w * h);
  const tiles = [];
  for (let ty = Math.max(0, Math.floor(y0 / ts)); ty <= Math.min(Math.ceil(height / ts) - 1, Math.floor((y0 + h - 1) / ts)); ty++)
    for (let tx = Math.max(0, Math.floor(x0 / ts)); tx <= Math.min(Math.ceil(width / ts) - 1, Math.floor((x0 + w - 1) / ts)); tx++) tiles.push([tx, ty]);
  await Promise.all(tiles.map(async ([tx, ty]) => {
    const t = await src.getTile({ x: tx, y: ty, selection });
    for (let j = 0; j < t.height; j++) {
      const y = ty * ts + j - y0; if (y < 0 || y >= h) continue;
      for (let i = 0; i < t.width; i++) { const x = tx * ts + i - x0; if (x >= 0 && x < w) out[y * w + x] = t.data[j * t.width + i]; }
    }
  }));
  return { data: out, w, h, x0, y0 };
}

// localStorage wrappers: per-viewer convenience only (in a session the server copy, verify.json, is authoritative)
export const store = {
  get(key, fallback) { try { const v = localStorage.getItem(key); return v ? JSON.parse(v) : fallback; } catch { return fallback; } },
  set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage unavailable */ } },
};
