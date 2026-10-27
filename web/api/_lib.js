// Shared helpers for the API (Vercel Node functions). Files starting with "_" are not routes.
// Blob layout (private store; the HPC worker reads and writes the same paths, worker/poller.py):
//   queue.json                         { jobs: [{ id, sid, kind, spec?, created }] }   written here (ETag-guarded), read by the worker
//   sessions/<sid>/request.json        the upload form (spec) + created + invite hash
//   sessions/<sid>/raw/{iv,ex}.tif     uploads (browser -> Blob through presigned PUT URLs; the worker deletes them once copied)
//   sessions/<sid>/state.json          created | uploading | queued | running | done | failed | expired  (+ step, text, slurm, error)
//   sessions/<sid>/web/...             viewer bundle (worker)          sessions/<sid>/verify.json   saved verdicts (viewer)
//   worker/heartbeat.json              worker liveness                 invites/<sha>.json           per-invite daily usage
// Free-tier budget (Hobby): 2,000 uploads / lists, 10,000 cache-miss reads, 1 GB stored, 10 GB downloaded a month; over that the
// store is blocked for 30 days. So: no list() calls, CDN-cached reads, few writes per session.
import { get, put, head, issueSignedToken, presignUrl, BlobPreconditionFailedError } from '@vercel/blob';
import { createHash, randomBytes } from 'node:crypto';

export const LIMITS = { fileBytes: 900 * 2 ** 20, ivVoxels: 400e6, exPixels: 2 * 8000 * 8000, rematchesPerSession: 6, perInvitePerDay: 3 };
export const MODELS = ['cyto2_ft', 'cyto3_ft', 'cpsam'];
export const SID = /^[A-Za-z0-9_-]{16,64}$/;
export const OFFLINE_S = 4 * 3600;                       // the worker writes a heartbeat every 3 h

export class HttpError extends Error { constructor(status, message) { super(message); this.status = status; } }

export function handler(fn) {
  return async (req, res) => {
    try {
      const body = req.method === 'POST' ? (typeof req.body === 'string' ? JSON.parse(req.body || '{}') : req.body || {}) : {};
      const out = await fn({ req, body, query: req.query || {} });
      res.setHeader('Cache-Control', 'no-store'); res.status(200).json(out);
    } catch (e) {
      const status = e instanceof HttpError ? e.status : 500;
      if (status === 500) console.error(e);
      res.status(status).json({ error: status === 500 ? 'Server error' : e.message });
    }
  };
}

export const newId = (n = 18) => randomBytes(n).toString('base64url');
export const sha = (s) => createHash('sha256').update(String(s)).digest('hex').slice(0, 24);

export async function readJSON(pathname, { fresh = false } = {}) {
  const r = await get(pathname, { access: 'private', useCache: !fresh }).catch((e) => { if (e.name === 'BlobNotFoundError') return null; throw e; });
  if (!r || r.statusCode !== 200) return null;
  return { data: JSON.parse(await new Response(r.stream).text()), etag: r.blob.etag };
}

export async function writeJSON(pathname, data, { ifMatch, maxAge = 60 } = {}) {
  return put(pathname, JSON.stringify(data), { access: 'private', allowOverwrite: true, addRandomSuffix: false, contentType: 'application/json',
    cacheControlMaxAge: maxAge, ...(ifMatch ? { ifMatch } : {}) });
}

export async function exists(pathname) { return head(pathname).then(() => true, () => false); }

export function checkSid(sid) { if (!SID.test(String(sid || ''))) throw new HttpError(400, 'Bad session id'); return sid; }

export async function session(sid) {
  checkSid(sid); const r = await readJSON(`sessions/${sid}/request.json`);
  if (!r) throw new HttpError(404, 'No such session'); return r.data;
}

export async function setState(sid, state) { await writeJSON(`sessions/${sid}/state.json`, { ...state, sid, updated: Date.now() / 1000 }); }

// queue.json: read-modify-write guarded by the ETag; retried on a concurrent write. Jobs older than 14 days are pruned here.
export async function enqueue(job) {
  for (let attempt = 0; attempt < 5; attempt++) {
    const cur = await readJSON('queue.json', { fresh: true });
    const jobs = (cur?.data.jobs || []).filter((j) => Date.now() / 1000 - j.created < 14 * 86400);
    const entry = { id: newId(9), created: Date.now() / 1000, ...job };
    try {
      await writeJSON('queue.json', { jobs: [...jobs, entry] }, { ifMatch: cur?.etag, maxAge: 3600 });
      return { entry, ahead: jobs.length };
    } catch (e) { if (!(e instanceof BlobPreconditionFailedError)) throw e; }
  }
  throw new HttpError(503, 'The queue is busy; try again');
}

// Signed GET URLs for a session's files. The signing secret never leaves the server; one token per warm instance.
let signer = null;
export async function signGet(pathnames, hours = 12) {
  if (!signer || signer.validUntil - Date.now() < 3 * 3600e3) signer = await issueSignedToken({ pathname: '*', operations: ['get'], validUntil: Date.now() + 7 * 24 * 3600e3 });
  const until = Date.now() + hours * 3600e3;
  return Object.fromEntries(await Promise.all(pathnames.map(async (p) => [p, (await presignUrl(signer, { operation: 'get', pathname: p, access: 'private', validUntil: until })).presignedUrl])));
}

// Invite codes: INVITE_CODES="code1,code2" (Vercel env). Usage per code per UTC day in invites/<sha>.json.
export async function useInvite(code) {
  const codes = String(process.env.INVITE_CODES || '').split(',').map((s) => s.trim()).filter(Boolean);
  if (!codes.includes(String(code || '').trim())) throw new HttpError(403, 'Unknown invite code');
  const key = `invites/${sha(code)}.json`, day = new Date().toISOString().slice(0, 10);
  const cur = await readJSON(key, { fresh: true }); const used = cur?.data.day === day ? cur.data.count : 0;
  if (used >= LIMITS.perInvitePerDay) throw new HttpError(429, `This invite code has used today's ${LIMITS.perInvitePerDay} analyses`);
  await writeJSON(key, { day, count: used + 1 }); return sha(code);
}
