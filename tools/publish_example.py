#!/usr/bin/env python3
"""Publish a finished session as the site's example (the user chose JY369 real data, 10-27): Blob examples/<name>/...
  web/...                 the session's viewer bundle (what "Explore the example" opens)
  raw/{iv,ex}.ome.tif     the inputs as compressed OME-TIFF with pixel sizes (downloads); EX is the GFP + DAPI max projection
  thumbs/{iv,ex}.jpg      hero images (IV max projection; EX GFP green + DAPI blue)
  info.json               title, description, spec for "run the example" sessions, download / thumb paths
The same OME inputs go to IVEX_EXAMPLES/<name>/ on scratch: the worker copies them for "run the example" (nothing is uploaded).
JY data is never written into the GitHub repo. Run inside the container (tifffile, cv2), with BLOB_READ_WRITE_TOKEN set.
usage: python tools/publish_example.py SESSION_DIR NAME --title T --description D [--dry-run]"""
import argparse, json, os, subprocess
from pathlib import Path
import numpy as np
import cv2
import tifffile

REPO = Path(__file__).resolve().parents[1]; NODE = os.environ.get('IVEX_NODE', '/scratch/sd5963/conda_envs/neuro_env_311/bin/node')
EXAMPLES = Path(os.environ.get('IVEX_EXAMPLES', '/scratch/sd5963/ivex-match-data/examples'))
ap = argparse.ArgumentParser(); ap.add_argument('session'); ap.add_argument('name'); ap.add_argument('--title', required=True); ap.add_argument('--description', required=True)
ap.add_argument('--dry-run', action='store_true'); a = ap.parse_args()
S = Path(a.session); job = json.loads((S / 'job.json').read_text()); out = EXAMPLES / a.name; (out / 'thumbs').mkdir(parents=True, exist_ok=True)
iv = np.load(S / 'raw/iv.npy', mmap_mode='r')[:, job['iv'].get('channel', 0)]; ex = np.load(S / 'raw/ex.npy', mmap_mode='r')
z0, z1 = (int(v) for v in job['ex']['z_range'].split(':')) if job['ex'].get('z_range') else (0, ex.shape[0])
chans = [job['ex'].get('channel', 0)] + ([job['ex']['dapi_channel']] if job['ex'].get('dapi_channel') is not None else [])
ex_mip = np.stack([np.asarray(ex[z0:z1, c]).max(0) for c in chans])                                    # (C, H, W)
names = ['GFP', 'DAPI'][:len(chans)]
tifffile.imwrite(out / 'iv.ome.tif', np.asarray(iv), ome=True, compression='zlib', metadata=dict(axes='ZYX', PhysicalSizeX=job['iv']['um_xy'], PhysicalSizeY=job['iv']['um_xy'],
                 PhysicalSizeZ=job['iv'].get('um_z'), PhysicalSizeXUnit='µm', PhysicalSizeYUnit='µm', PhysicalSizeZUnit='µm'))
tifffile.imwrite(out / 'ex.ome.tif', ex_mip, ome=True, compression='zlib', metadata=dict(axes='CYX', PhysicalSizeX=job['ex']['um_xy'], PhysicalSizeY=job['ex']['um_xy'],
                 PhysicalSizeXUnit='µm', PhysicalSizeYUnit='µm', Channel=dict(Name=names)))


def u8(a, p=(0.5, 99.7)):
    lo, hi = np.percentile(a[a > 0] if (a > 0).any() else a, p); return (np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1) * 255).astype(np.uint8)


def small(img, side=900): s = side / max(img.shape[:2]); return cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)


cv2.imwrite(str(out / 'thumbs/iv.jpg'), small(np.stack([u8(np.asarray(iv).max(0))] * 3, -1)), [cv2.IMWRITE_JPEG_QUALITY, 88])
g = small(u8(ex_mip[0])); d = small(u8(ex_mip[1])) if len(ex_mip) > 1 else np.zeros_like(g)
cv2.imwrite(str(out / 'thumbs/ex.jpg'), np.stack([np.maximum(d, g // 4), g, g // 3], -1), [cv2.IMWRITE_JPEG_QUALITY, 88])   # BGR: DAPI blue, GFP green
mb = lambda f: round((out / f).stat().st_size / 2 ** 20, 1)
spec = dict(title=a.title, models=job.get('models'), match_model=job.get('match_model'),
            iv=dict(um_xy=job['iv']['um_xy'], um_z=job['iv'].get('um_z'), channel=0, diameter_um=job['iv'].get('diameter_um', 11)),
            ex=dict(um_xy=job['ex']['um_xy'], channel=0, dapi_channel=1 if len(chans) > 1 else None, z_range='', diameter_um=job['ex'].get('diameter_um', 11)))
info = dict(name=a.name, title=a.title, description=a.description, spec=spec, iv_shape=list(iv.shape), ex_shape=list(ex_mip.shape),
            downloads=dict(iv=dict(path='raw/iv.ome.tif', mb=mb('iv.ome.tif')), ex=dict(path='raw/ex.ome.tif', mb=mb('ex.ome.tif'))),
            thumbs=dict(iv='thumbs/iv.jpg', ex='thumbs/ex.jpg'))
(out / 'info.json').write_text(json.dumps(info, indent=1)); print(json.dumps(info, indent=1))
web = S / 'web'; files = json.loads((web / 'manifest.json').read_text())['files']
uploads = [(web / f, f'web/{f}') for f in files if f != 'manifest.json'] + [(web / 'manifest.json', 'web/manifest.json'), (out / 'iv.ome.tif', 'raw/iv.ome.tif'),
           (out / 'ex.ome.tif', 'raw/ex.ome.tif'), (out / 'thumbs/iv.jpg', 'thumbs/iv.jpg'), (out / 'thumbs/ex.jpg', 'thumbs/ex.jpg'), (out / 'info.json', 'info.json')]
print(f'{len(uploads)} files, {sum(p.stat().st_size for p, _ in uploads) / 2 ** 20:.1f} MB -> examples/{a.name}/')
if not a.dry_run:
    for src, dst in uploads:
        r = subprocess.run([NODE, str(REPO / 'worker/blob.mjs'), 'put', f'examples/{a.name}/{dst}', str(src), '--max-age', str(86400 if dst.endswith(('.tif', '.jpg')) else 300)],
                           capture_output=True, text=True, env=os.environ)
        assert r.returncode == 0, f'{dst}: {r.stderr[-300:]}'
    print('published')
