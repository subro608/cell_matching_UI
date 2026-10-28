// POST /api/rematch { sid, kind: 'rematch', affine?: [[a, b, tx], [c, d, ty]] | flip?, theta?, match_model? }
//                   { sid, kind: 'restore' }   (files expired from the cloud; the worker uploads them again from NYU scratch)
import { handler, HttpError, LIMITS, MODELS, enqueue, readJSON, session, setState, writeJSON } from './_lib.js';

export default handler(async ({ body }) => {
  const sid = body.sid; const s = await session(sid, { fresh: true }); const kind = body.kind === 'restore' ? 'restore' : 'rematch';
  const st = (await readJSON(`sessions/${sid}/state.json`, { fresh: true }))?.data;
  if (kind === 'restore') {
    if (st?.status !== 'expired') throw new HttpError(400, 'Nothing to restore');
  } else {
    if (st?.status !== 'done') throw new HttpError(409, 'Wait until the current job has finished');
    if ((s.rematches || 0) >= LIMITS.rematchesPerSession) throw new HttpError(429, `At most ${LIMITS.rematchesPerSession} re-matches per session`);
  }
  const spec = {};
  if (kind === 'rematch') {
    if (Array.isArray(body.affine)) {
      const A = body.affine; if (A.length !== 2 || A.some((r) => !Array.isArray(r) || r.length !== 3 || r.some((v) => !Number.isFinite(Number(v))))) throw new HttpError(400, 'affine must be 2 x 3');
      spec.affine = A.map((r) => r.map(Number));
    } else if (body.theta != null) { spec.flip = !!body.flip; spec.theta = Number(body.theta) % 360; }
    if (body.match_model) { if (!MODELS.includes(body.match_model) || !s.spec.models.includes(body.match_model)) throw new HttpError(400, 'Unknown mask set'); spec.match_model = body.match_model; }
    if (!spec.affine && spec.theta == null && !spec.match_model) throw new HttpError(400, 'Nothing to change');
    await writeJSON(`sessions/${sid}/request.json`, { ...s, rematches: (s.rematches || 0) + 1 }, { maxAge: 3600 });
  }
  const { ahead } = await enqueue({ sid, kind, spec });
  await setState(sid, { status: 'queued', kind, text: kind === 'restore' ? 'Restoring the files' : 'Re-matching at your orientation', ahead });
  return { ok: true, ahead };
});
