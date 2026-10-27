// Vercel Blob for the HPC worker (private store; token from BLOB_READ_WRITE_TOKEN, never on the command line).
//   node blob.mjs get PATH [OUT] [--fresh]      -> file or stdout; exit 3 if missing. --fresh bypasses the CDN cache (counts as an operation)
//   node blob.mjs put PATH FILE [--max-age S] [--if-match ETAG] [--type T]   -> {"pathname","etag"}; always overwrites
//   node blob.mjs del PATH...                    deletes are free on every plan
// Every put is one "advanced operation" of the Hobby plan's 2,000 a month: the poller writes few, large files.
import { get, put, del, BlobNotFoundError } from '@vercel/blob';
import { createReadStream, createWriteStream, statSync } from 'node:fs';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';

const [cmd, ...rest] = process.argv.slice(2);
const flags = Object.fromEntries(rest.filter((a) => a.startsWith('--')).map((a, i, all) => [a.slice(2), true]));
const args = []; for (let i = 0; i < rest.length; i++) { if (rest[i].startsWith('--')) { const k = rest[i].slice(2); if (['max-age', 'if-match', 'type'].includes(k)) flags[k] = rest[++i]; } else args.push(rest[i]); }

try {
  if (cmd === 'get') {
    const r = await get(args[0], { access: 'private', useCache: !flags.fresh });
    if (!r || r.statusCode !== 200) process.exit(3);
    await pipeline(Readable.fromWeb(r.stream), args[1] ? createWriteStream(args[1]) : process.stdout);
    if (args[1]) process.stderr.write(JSON.stringify({ etag: r.blob.etag, size: r.blob.size }) + '\n');
  } else if (cmd === 'put') {
    const size = statSync(args[1]).size;
    const r = await put(args[0], size > 0 ? createReadStream(args[1]) : '', {
      access: 'private', allowOverwrite: true, addRandomSuffix: false, multipart: size > 400 * 2 ** 20,
      contentType: flags.type, cacheControlMaxAge: flags['max-age'] ? Number(flags['max-age']) : undefined, ifMatch: flags['if-match'],
    });
    console.log(JSON.stringify({ pathname: r.pathname, etag: r.etag }));
  } else if (cmd === 'del') {
    await del(args);
  } else {
    console.error('usage: node blob.mjs get|put|del ...'); process.exit(2);
  }
} catch (e) {
  if (e instanceof BlobNotFoundError) process.exit(3);
  console.error(`${e.name}: ${e.message}`); process.exit(1);
}
