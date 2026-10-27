// POST /api/verify { sid, verdicts, registration?, pose_ok? } -> sessions/<sid>/verify.json (one write per save; the viewer throttles)
import { handler, HttpError, session, writeJSON } from './_lib.js';

export default handler(async ({ body }) => {
  const sid = body.sid; await session(sid);
  const doc = { verdicts: body.verdicts || {}, registration: body.registration || null, pose_ok: body.pose_ok ?? null, run: body.run || null, saved: Date.now() / 1000 };
  const s = JSON.stringify(doc); if (s.length > 3e6) throw new HttpError(413, 'Too large');
  if (typeof doc.verdicts !== 'object' || Array.isArray(doc.verdicts)) throw new HttpError(400, 'verdicts must be an object');
  await writeJSON(`sessions/${sid}/verify.json`, doc, { maxAge: 0 }); return { ok: true, saved: doc.saved };
});
