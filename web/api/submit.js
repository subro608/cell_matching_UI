// POST /api/submit { sid } -> both uploads present -> queued for the HPC worker.
import { handler, HttpError, enqueue, exists, rawPath, readJSON, session, setState } from './_lib.js';

export default handler(async ({ body }) => {
  const sid = body.sid; const s = await session(sid, { fresh: true });
  const st = (await readJSON(`sessions/${sid}/state.json`, { fresh: true }))?.data;
  if (st && ['queued', 'running', 'done', 'probing'].includes(st.status)) return { ok: true, already: st.status };
  const fresh = !st || ['created', 'uploading'].includes(st.status);
  if (!s.example && fresh) for (const side of ['iv', 'ex']) if (!(await exists(rawPath(sid, side, s.spec)))) throw new HttpError(400, `The ${side.toUpperCase()} upload is missing`);
  if (s.needs_probe && !s.described) {               // a format the browser cannot read: the worker reads it first, then the user confirms
    const { ahead } = await enqueue({ sid, kind: 'probe' });
    await setState(sid, { status: 'queued', kind: 'probe', text: 'Waiting for the worker to read your files', ahead }); return { ok: true, ahead, probe: true };
  }
  const { ahead } = await enqueue({ sid, kind: 'analyze' });
  await setState(sid, { status: 'queued', kind: 'analyze', text: 'Waiting for the HPC worker', ahead });
  return { ok: true, ahead };
});
