#!/usr/bin/env python3
"""Local server for the viewer: web/dist at /, a bundle directory at /data/, with HTTP Range (Viv reads OME-TIFF tiles by range).
Binds to 127.0.0.1 only. usage: python serve.py --data /scratch/sd5963/ivex-match-data/demo_jy369 [--port 8765]"""
import argparse, os, re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DIST = Path(__file__).resolve().parent / 'dist'


class Handler(SimpleHTTPRequestHandler):
    data_dir = None

    def translate_path(self, path):
        path = path.split('?', 1)[0].split('#', 1)[0]
        if path.startswith('/data/'):
            root, rel = Path(self.data_dir), path[len('/data/'):]
        else:
            root, rel = DIST, path.lstrip('/')
        target = (root / rel).resolve()
        return str(target) if str(target).startswith(str(root.resolve())) else str(root / '__forbidden__')

    def send_head(self):
        rng = self.headers.get('Range'); path = self.translate_path(self.path)
        if not rng or not os.path.isfile(path): return super().send_head()
        m = re.match(r'bytes=(\d*)-(\d*)$', rng.strip()); size = os.path.getsize(path)
        if not m: self.send_error(416); return None
        start = int(m.group(1)) if m.group(1) else size - int(m.group(2)); end = int(m.group(2)) if m.group(1) and m.group(2) else size - 1
        end = min(end, size - 1)
        if start > end: self.send_error(416); return None
        f = open(path, 'rb'); f.seek(start); self._remaining = end - start + 1
        self.send_response(206); self.send_header('Content-Type', self.guess_type(path)); self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}'); self.send_header('Content-Length', str(self._remaining)); self.end_headers()
        return f

    def copyfile(self, source, outputfile):
        n = getattr(self, '_remaining', None)
        if n is None: return super().copyfile(source, outputfile)
        while n > 0:
            chunk = source.read(min(1 << 20, n))
            if not chunk: break
            outputfile.write(chunk); n -= len(chunk)
        self._remaining = None

    def end_headers(self):
        self.send_header('Accept-Ranges', 'bytes'); self.send_header('Cache-Control', 'no-cache'); super().end_headers()

    def log_message(self, *a): pass


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--port', type=int, default=8765); a = ap.parse_args()
    Handler.data_dir = a.data
    print(f'serving {DIST} at http://127.0.0.1:{a.port}/  and {a.data} at /data/', flush=True)
    ThreadingHTTPServer(('127.0.0.1', a.port), partial(Handler)).serve_forever()
