// Read a local TIFF in the browser before uploading: dimensions (Z, C, Y, X), pixel sizes when the file states them, and a small
// max-projection preview per channel. Understands OME-TIFF, ImageJ hyperstacks and tifffile's {"shape": [...]} description;
// otherwise every page is a plane. Nothing is uploaded here.
import { fromBlob } from 'geotiff';

function parseOme(desc) {
  const doc = new DOMParser().parseFromString(desc, 'application/xml'); const px = doc.getElementsByTagName('Pixels')[0];
  if (!px) return null;
  const a = (k) => px.getAttribute(k), unit = (u) => (!u || /µm|um|micron/i.test(u) ? 1 : /nm/i.test(u) ? 1e-3 : /mm/i.test(u) ? 1e3 : 1);
  return { Z: +a('SizeZ') || 1, C: +a('SizeC') || 1, T: +a('SizeT') || 1, order: a('DimensionOrder') || 'XYCZT',
    um_xy: a('PhysicalSizeX') ? +a('PhysicalSizeX') * unit(a('PhysicalSizeXUnit')) : null, um_z: a('PhysicalSizeZ') ? +a('PhysicalSizeZ') * unit(a('PhysicalSizeZUnit')) : null,
    names: [...doc.getElementsByTagName('Channel')].map((c) => c.getAttribute('Name')).filter(Boolean) };
}

function parseImageJ(desc, img) {
  if (!desc.startsWith('ImageJ=')) return null;
  const kv = Object.fromEntries(desc.split('\n').map((l) => l.split('=')).filter((p) => p.length === 2));
  const res = img.fileDirectory.XResolution, perUnit = Array.isArray(res) ? res[0] / res[1] : res;
  const micron = /micron|µm|um/i.test(kv.unit || '');
  return { Z: +kv.slices || (+kv.images || 1) / (+kv.channels || 1), C: +kv.channels || 1, T: +kv.frames || 1, order: 'XYCZT',
    um_xy: micron && perUnit ? 1 / perUnit : null, um_z: micron && kv.spacing ? +kv.spacing : null, names: [] };
}

function parseTifffile(desc) {
  try {
    const j = JSON.parse(desc); if (!Array.isArray(j.shape)) return null; const s = j.shape;
    if (s.length === 2) return { Z: 1, C: 1, T: 1, order: 'XYCZT' };
    if (s.length === 3) return { Z: s[0], C: 1, T: 1, order: 'XYCZT' };
    if (s.length === 4) return { Z: s[0], C: s[1], T: 1, order: 'XYCZT' };          // tifffile (Z, C, Y, X): pages z-major, c fastest
  } catch { /* not JSON */ }
  return null;
}

export async function inspectTiff(file) {
  const tiff = await fromBlob(file); const img = await tiff.getImage(0);
  const desc = String(img.fileDirectory.ImageDescription || '').replace(/\0+$/, '').trim();   // TIFF ASCII fields end in NUL
  let dims = (desc.includes('<OME') && parseOme(desc)) || parseImageJ(desc, img) || parseTifffile(desc);
  const spp = img.getSamplesPerPixel();
  if (!dims) { const n = await tiff.getImageCount(); dims = { Z: n, C: 1, T: 1, order: 'XYCZT' }; }
  if (spp > 1) dims = { ...dims, C: spp, interleaved: true };
  const W = img.getWidth(), H = img.getHeight();
  return { tiff, file, W, H, ...dims, dtype: `${img.getBitsPerSample()}-bit`, shape: [dims.Z, dims.C, H, W] };
}

// page index of (z, c) for the file's dimension order (t = 0)
function page(info, z, c) {
  if (info.interleaved) return z;
  return info.order.indexOf('C') < info.order.indexOf('Z') ? z * info.C + c : c * info.Z + z;
}

// max projection over up to maxPlanes planes of channel c, resampled to <= side px; returns { canvas, lo, hi }
export async function previewMip(info, c, { side = 256, maxPlanes = 12, z0 = 0, z1 = null } = {}) {
  const last = Math.min(info.Z, z1 ?? info.Z), n = Math.max(1, last - z0), step = Math.max(1, Math.floor(n / maxPlanes));
  const s = Math.min(1, side / Math.max(info.W, info.H)), w = Math.max(1, Math.round(info.W * s)), h = Math.max(1, Math.round(info.H * s));
  const mip = new Float32Array(w * h);
  for (let z = z0; z < last; z += step) {
    const img = await info.tiff.getImage(page(info, z, c));
    const r = await img.readRasters({ width: w, height: h, resampleMethod: 'nearest', samples: info.interleaved ? [c] : [0] });
    const d = r[0]; for (let i = 0; i < d.length; i++) if (d[i] > mip[i]) mip[i] = d[i];
  }
  const sorted = Float32Array.from(mip).sort(), lo = sorted[Math.floor(sorted.length * 0.01)], hi = Math.max(sorted[Math.floor(sorted.length * 0.998)], lo + 1);
  const canvas = document.createElement('canvas'); canvas.width = w; canvas.height = h; const ctx = canvas.getContext('2d'); const im = ctx.createImageData(w, h);
  for (let i = 0; i < mip.length; i++) { const v = Math.max(0, Math.min(255, ((mip[i] - lo) / (hi - lo)) * 255)); im.data.set([v, v, v, 255], i * 4); }
  ctx.putImageData(im, 0, 0); return { canvas, lo, hi };
}
