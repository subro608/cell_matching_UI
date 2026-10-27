// Local stand-in for Vercel: serves dist/ and runs api/*.js with the Vercel Node helpers (req.query, req.body, res.status().json()),
// against the real Blob store. Binds 127.0.0.1 only. Needs BLOB_READ_WRITE_TOKEN and INVITE_CODES in the environment.
// usage: node dev_api_server.mjs [PORT]
import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, join, normalize } from 'node:path';

const PORT = Number(process.argv[2] || 8790), DIST = new URL('./dist/', import.meta.url).pathname;
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.json': 'application/json', '.wasm': 'application/wasm' };

createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);
  if (url.pathname.startsWith('/api/')) {
    const name = url.pathname.slice(5).replace(/[^a-z]/g, '');
    let raw = ''; for await (const c of req) raw += c;
    req.query = Object.fromEntries(url.searchParams); try { req.body = raw ? JSON.parse(raw) : {}; } catch { req.body = raw; }
    res.status = (s) => { res.statusCode = s; return res; };
    res.json = (o) => { if (res.statusCode >= 400) console.log('  error:', JSON.stringify(o).slice(0, 400)); res.setHeader('Content-Type', 'application/json'); res.end(JSON.stringify(o)); };
    try { const mod = await import(`./api/${name}.js`); await mod.default(req, res); }
    catch (e) { console.error(e); if (!res.writableEnded) res.status(500).json({ error: String(e) }); }
    console.log(req.method, url.pathname, res.statusCode); return;
  }
  const p = normalize(join(DIST, url.pathname === '/' ? 'index.html' : url.pathname));
  if (!p.startsWith(DIST)) { res.statusCode = 403; return res.end(); }
  try {
    const st = await stat(p); res.setHeader('Content-Type', TYPES[extname(p)] || 'application/octet-stream'); res.setHeader('Accept-Ranges', 'bytes');
    const m = /^bytes=(\d+)-(\d*)$/.exec(req.headers.range || '');                       // Viv reads OME-TIFF tiles by range
    if (!m) return res.end(await readFile(p));
    const start = Number(m[1]), end = Math.min(m[2] ? Number(m[2]) : st.size - 1, st.size - 1), buf = (await readFile(p)).subarray(start, end + 1);
    res.statusCode = 206; res.setHeader('Content-Range', `bytes ${start}-${end}/${st.size}`); res.end(buf);
  }
  catch { res.statusCode = 404; res.end('not found'); }
}).listen(PORT, '127.0.0.1', () => console.log(`dev server http://127.0.0.1:${PORT}/`));
