// GET /api/status -> worker liveness and queue length (home page banner).
import { handler, OFFLINE_S, readJSON } from './_lib.js';

export default handler(async () => {
  const [hb, q] = await Promise.all([readJSON('worker/heartbeat.json'), readJSON('queue.json')]);
  const age = hb ? Date.now() / 1000 - hb.data.t : null;
  return { worker: { online: age != null && age < OFFLINE_S, last_seen_s: age, pending: hb?.data.pending ?? null }, queue: { length: q?.data.jobs.length || 0 } };
});
