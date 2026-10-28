// POST /api/describe { sid, spec } -> after the worker has read the files (state 'probed'), the user's confirmed description replaces
// the deferred one; the client then calls /api/submit to queue the analysis.
import { handler, HttpError, readJSON, session, writeJSON } from './_lib.js';
import { cleanSpec } from './session.js';

export default handler(async ({ body }) => {
  const sid = body.sid; const s = await session(sid, { fresh: true });
  const st = (await readJSON(`sessions/${sid}/state.json`, { fresh: true }))?.data;
  if (!s.needs_probe || st?.status !== 'probed') throw new HttpError(409, 'Nothing to describe yet');
  const probe = (await readJSON(`sessions/${sid}/probe/probe.json`, { fresh: true }))?.data; if (!probe) throw new HttpError(409, 'The files have not been read yet');
  const spec = body.spec || {};
  for (const side of ['iv', 'ex']) {                // shapes come from what the worker read, not from the client
    const m = probe[side]; spec[side] = { ...spec[side], name: s.spec[side].name, bytes: s.spec[side].bytes, shape: [m.Z, m.C, m.Y, m.X], deferred: false };
  }
  const clean = cleanSpec(spec);
  await writeJSON(`sessions/${sid}/request.json`, { ...s, spec: clean, described: true }, { maxAge: 3600 });
  return { ok: true, spec: clean };
});
