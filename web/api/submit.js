// POST /api/submit { sid } -> both uploads present -> queued for the HPC worker.
import { handler, HttpError, enqueue, exists, readJSON, session, setState } from './_lib.js';

export default handler(async ({ body }) => {
  const sid = body.sid; const s = await session(sid);
  const st = (await readJSON(`sessions/${sid}/state.json`, { fresh: true }))?.data;
  if (st && ['queued', 'running', 'done'].includes(st.status)) return { ok: true, already: st.status };
  if (!s.example) for (const side of ['iv', 'ex']) if (!(await exists(`sessions/${sid}/raw/${side}.tif`))) throw new HttpError(400, `The ${side.toUpperCase()} upload is missing`);
  const { ahead } = await enqueue({ sid, kind: 'analyze' });
  await setState(sid, { status: 'queued', kind: 'analyze', text: 'Waiting for the HPC worker', ahead });
  return { ok: true, ahead };
});
