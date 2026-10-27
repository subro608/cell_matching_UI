// POST /api/session  { invite, spec }  -> { sid }           create a session (invite-gated) before uploading
// GET  /api/session?sid=...           -> { state, spec, worker, queue, bundle? }   status page + viewer bootstrap
//      bundle (when done) = { manifest, urls: { path: signed GET URL (12 h) }, verify }
import { handler, HttpError, LIMITS, MODELS, OFFLINE_S, checkSid, newId, readJSON, writeJSON, setState, signGet, useInvite } from './_lib.js';

const num = (v, lo, hi, name) => { const x = Number(v); if (!Number.isFinite(x) || x < lo || x > hi) throw new HttpError(400, `${name} must be between ${lo} and ${hi}`); return x; };
const int = (v, lo, hi, name) => Math.round(num(v, lo, hi, name));

function cleanSpec(s) {
  if (!s || !s.iv || !s.ex) throw new HttpError(400, 'Missing the IV / EX description');
  const shape = (a, name) => { if (!Array.isArray(a) || a.length < 2 || a.length > 4 || a.some((d) => !Number.isInteger(d) || d < 1)) throw new HttpError(400, `${name} shape is not valid`); return a; };
  const iv = { um_xy: num(s.iv.um_xy, 0.02, 20, 'IV pixel size'), um_z: s.iv.um_z == null || s.iv.um_z === '' ? null : num(s.iv.um_z, 0.05, 100, 'IV z step'),
    channel: int(s.iv.channel ?? 0, 0, 16, 'IV channel'), diameter_um: num(s.iv.diameter_um ?? 11, 2, 80, 'IV cell diameter'),
    bytes: int(s.iv.bytes, 1, LIMITS.fileBytes, 'IV file size'), shape: shape(s.iv.shape, 'IV'), name: String(s.iv.name || 'iv.tif').slice(0, 120) };
  const ex = { um_xy: num(s.ex.um_xy, 0.02, 20, 'EX pixel size'), channel: int(s.ex.channel ?? 0, 0, 16, 'EX channel'),
    dapi_channel: s.ex.dapi_channel == null || s.ex.dapi_channel === '' ? null : int(s.ex.dapi_channel, 0, 16, 'DAPI channel'),
    z_range: /^\d+:\d+$/.test(String(s.ex.z_range || '')) ? s.ex.z_range : '', diameter_um: num(s.ex.diameter_um ?? 11, 2, 80, 'EX cell diameter'),
    bytes: int(s.ex.bytes, 1, LIMITS.fileBytes, 'EX file size'), shape: shape(s.ex.shape, 'EX'), name: String(s.ex.name || 'ex.tif').slice(0, 120) };
  const vox = (sh) => sh.slice(-3).reduce((a, b) => a * b, 1);
  if (vox(iv.shape) > LIMITS.ivVoxels) throw new HttpError(400, `The IV stack is too large for the pilot (${(vox(iv.shape) / 1e6).toFixed(0)} M voxels; limit ${LIMITS.ivVoxels / 1e6} M)`);
  const models = (Array.isArray(s.models) && s.models.length ? s.models : MODELS).filter((m) => MODELS.includes(m));
  if (!models.length) throw new HttpError(400, 'Choose at least one mask model');
  const match_model = models.includes(s.match_model) ? s.match_model : models.includes('cyto3_ft') ? 'cyto3_ft' : models[0];
  return { title: String(s.title || 'Untitled session').slice(0, 120), iv, ex, models, match_model };
}

export default handler(async ({ req, body, query }) => {
  if (req.method === 'POST') {
    let spec, example = null;
    if (body.example) {            // the worker takes the inputs from its own copy of the example; nothing to upload
      example = String(body.example); if (!/^[a-z0-9_-]{1,32}$/.test(example)) throw new HttpError(400, 'Bad example name');
      const info = await readJSON(`examples/${example}/info.json`); if (!info) throw new HttpError(404, 'No such example');
      spec = { ...info.data.spec, title: `${info.data.title} (your run)` };
    } else spec = cleanSpec(body.spec);
    const invite = await useInvite(body.invite); const sid = newId(18);
    await writeJSON(`sessions/${sid}/request.json`, { spec, example, created: Date.now() / 1000, invite, rematches: 0 }, { maxAge: 3600 });
    await setState(sid, { status: 'created', text: 'Waiting for the uploads' });
    return { sid };
  }
  const sid = checkSid(query.sid);
  const [req0, state, hb, queue] = await Promise.all([readJSON(`sessions/${sid}/request.json`), readJSON(`sessions/${sid}/state.json`), readJSON('worker/heartbeat.json'), readJSON('queue.json')]);
  if (!req0) throw new HttpError(404, 'No such session (it may have expired)');
  const age = hb ? Date.now() / 1000 - hb.data.t : null;
  const jobs = queue?.data.jobs || []; const mine = jobs.findIndex((j) => j.sid === sid);
  const out = { sid, spec: req0.data.spec, created: req0.data.created, state: state?.data || { status: 'created' },
    worker: { online: age != null && age < OFFLINE_S, last_seen_s: age, pending: hb?.data.pending ?? null }, queue: { length: jobs.length, index: mine } };
  if (!['created', 'uploading', 'expired', 'failed'].includes(out.state.status) && !(out.state.kind === 'analyze' && out.state.status !== 'done')) {   // a bundle exists
    const man = await readJSON(`sessions/${sid}/web/manifest.json`);
    if (man) {
      const files = man.data.files.filter((f) => f !== 'manifest.json');
      out.bundle = { manifest: man.data, urls: await signGet(files.map((f) => `sessions/${sid}/web/${f}`)), prefix: `sessions/${sid}/web/` };
      out.bundle.verify = (await readJSON(`sessions/${sid}/verify.json`, { fresh: true }))?.data || null;
    }
  }
  return out;
});
