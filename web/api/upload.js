// POST /api/upload  (@vercel/blob/client uploadPresigned protocol) -> a presigned PUT URL for exactly
// sessions/<sid>/raw/{iv,ex}<ext>, capped at the size the session declared. The browser uploads straight to Blob.
import { issueSignedToken } from '@vercel/blob';
import { handleUploadPresigned } from '@vercel/blob/client';
import { HttpError, rawPath, session } from './_lib.js';

export default async function upload(req, res) {
  try {
    const body = typeof req.body === 'string' ? JSON.parse(req.body) : req.body;
    // only presigned-URL requests: no upload-completed callbacks (submit checks the files), so the SDK's webhook key is never used
    if (body?.type !== 'blob.generate-presigned-url') throw new HttpError(400, 'Unsupported upload event');
    const json = await handleUploadPresigned({
      body, request: req, webhookPublicKey: process.env.BLOB_WEBHOOK_PUBLIC_KEY || 'unused-no-callbacks',
      getSignedToken: async (pathname, clientPayload) => {
        const { sid, side } = JSON.parse(clientPayload || '{}'); const s = await session(sid);
        if (!['iv', 'ex'].includes(side) || pathname !== rawPath(sid, side, s.spec)) throw new HttpError(400, 'Upload path does not match the session');
        const max = s.spec[side].bytes;
        const token = await issueSignedToken({ pathname, operations: ['put'], maximumSizeInBytes: max, validUntil: Date.now() + 2 * 3600e3 });
        return { token, urlOptions: { maximumSizeInBytes: max, validUntil: Date.now() + 2 * 3600e3, addRandomSuffix: false, allowOverwrite: true, cacheControlMaxAge: 60 } };
      },
    });
    res.status(200).json(json);
  } catch (e) {
    res.status(e instanceof HttpError ? e.status : 400).json({ error: e.message });
  }
}
