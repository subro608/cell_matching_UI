// Client for the Vercel API (web/api/*). Uploads go straight from the browser to Vercel Blob through presigned PUT URLs.
import { uploadPresigned } from '@vercel/blob/client';

async function call(path, body) {
  const r = await fetch(`/api/${path}`, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {});
  const j = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
  if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
}

export const createSession = (invite, spec) => call('session', { invite, spec });
export const getSession = (sid) => call(`session?sid=${encodeURIComponent(sid)}`);
export const submit = (sid) => call('submit', { sid });
export const saveVerify = (sid, doc) => call('verify', { sid, ...doc });
export const rematch = (sid, body) => call('rematch', { sid, kind: 'rematch', ...body });
export const restore = (sid) => call('rematch', { sid, kind: 'restore' });
export const status = () => call('status');

// one file -> sessions/<sid>/raw/<side>.tif; onProgress(fraction)
export async function uploadFile(sid, side, file, onProgress) {
  return uploadPresigned(`sessions/${sid}/raw/${side}.tif`, file, {
    access: 'private', handleUploadUrl: '/api/upload', clientPayload: JSON.stringify({ sid, side }), contentType: 'application/octet-stream',
    multipart: false, onUploadProgress: (e) => onProgress?.(e.percentage / 100),
  });
}
export const getExample = (name = 'jy369') => call(`example?name=${encodeURIComponent(name)}`);
export const startExample = (invite, name = 'jy369') => call('session', { invite, example: name });
