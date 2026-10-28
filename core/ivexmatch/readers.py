"""Read any uploaded microscopy file as (Z, C, Y, X) + metadata. Runs in venvs/ivex_io_env (bioio + plugins, nd2, tifffile, imagecodecs);
the worker calls it as a subprocess, like the Cellpose environments. Formats are detected from the file's first bytes, not its name.

  TIFF / OME-TIFF / ImageJ / BigTIFF   tifffile; a single file of a multi-file OME set (Bruker, tifffile per-plane) is read page by page
  Nikon ND2                            nd2; one stage position (P) per analysis; voxel size + channel names from the metadata
  ZIP of per-plane images              one TIFF / PNG per plane (and per channel): grouped by the Ch# token and the varying number in
                                       the names (Bruker Prairie View *_Cycle00001_Ch2_000012.ome.tif, z012_Ch1.ome.tif, ...);
                                       pixel size and z step from a Prairie View .xml in the zip when present
  ZIP with one .nd2 / .czi / .lif      extracted (size-checked) and read as that format
  CZI, LIF, OIB/OIR, IMS               bioio (scene = position)
  NPY / NPZ                            numpy, allow_pickle=False; NPZ: the largest array
  PNG / JPEG                           Pillow; RGB becomes channels
  Python pickle                        refused: unpickling an upload could run arbitrary code (tools/convert_lut_pickle.py converts
                                       the lab's LUT pickles locally)
Not supported: stitching tiles / positions into one field (pick one position).

usage: python -m ivexmatch.readers probe FILE --json OUT.json [--previews DIR] [--position P]
       python -m ivexmatch.readers convert FILE --out OUT.npy [--position P]"""
from __future__ import annotations
import argparse, json, re, statistics, sys, tempfile, zipfile
from pathlib import Path
import numpy as np

MAX_UNZIPPED = 24 << 30                                  # refuse archives that would expand beyond this (zip bombs)
MAGIC = [(b'II*\x00', 'tiff'), (b'MM\x00*', 'tiff'), (b'II+\x00', 'tiff'), (b'MM\x00+', 'tiff'), (b'\xda\xce\xbe\n', 'nd2'),
         (b'\x00\x00\x00\x0cjP  ', 'nd2'), (b'PK\x03\x04', 'zip'), (b'\x93NUMPY', 'npy'), (b'\x89PNG', 'image'), (b'\xff\xd8\xff', 'image'),
         (b'ZISRAWFILE', 'bioio'), (b'\x80', 'pickle')]
BIOIO_EXT = {'.czi', '.lif', '.oib', '.oir', '.ims', '.vsi', '.lsm'}


class Unsupported(Exception): pass


def sniff(path):
    head = Path(path).open('rb').read(16)
    for m, kind in MAGIC:
        if head.startswith(m): return kind
    if Path(path).suffix.lower() in BIOIO_EXT: return 'bioio'
    raise Unsupported(f'{Path(path).name}: unrecognised file format (first bytes {head[:8].hex()})')


def to_zcyx(a, axes):
    """array + axes string over Z C Y X (plus anything else, which keeps index 0) -> (Z, C, Y, X)."""
    a = np.asarray(a) if not hasattr(a, 'compute') else a; axes = list(axes)
    for i in [i for i, ax in enumerate(axes) if ax not in 'ZCYX'][::-1]: a = a.take(0, axis=i) if not hasattr(a, 'compute') else a[(slice(None),) * i + (0,)]; axes.pop(i)
    for ax in 'CZ':
        if ax not in axes: a = a[None]; axes.insert(0, ax)
    order = [axes.index(ax) for ax in 'ZCYX']
    return a.transpose(order)


def guess_axes(shape):
    """Axes for an array without metadata: Y X last (or Y X C with C <= 4); a short second axis is C."""
    n = len(shape)
    if n == 2: return 'YX'
    if n == 3: return 'YXC' if shape[2] <= 4 and shape[0] > 4 else 'ZYX'
    if n == 4: return 'ZYXC' if shape[3] <= 4 and shape[1] > 4 else 'ZCYX'
    if n == 5: return 'TZCYX'
    raise Unsupported(f'array with {n} dimensions')


# ---------------------------------------------------------------------------------------------------------------- readers
def _tiff(path, position=0):
    import tifffile
    with tifffile.TiffFile(path) as t:
        ome = t.ome_metadata or ''; meta = dict(format='tiff', positions=[dict(index=i, name=s.name or f'series {i}') for i, s in enumerate(t.series)])
        other_files = set(re.findall(r'FileName="([^"]+)"', ome)) - {Path(path).name}
        if other_files:                                   # one file of a multi-file OME set: read only its own pages
            arr = np.stack([p.asarray() for p in t.pages]); axes = 'ZYX' if arr.ndim == 3 else 'ZYXS'
            meta['warnings'] = [f'this file belongs to a {len(other_files) + 1}-file OME set; only its own {len(t.pages)} page(s) are read (upload a zip of the whole set)']
        else:
            s = t.series[min(position, len(t.series) - 1)]; arr, axes = s.asarray(), s.axes
            ax = list(axes)
            if 'S' in ax:
                if 'C' in ax: arr = arr.take(0, axis=ax.index('S')); ax.remove('S')
                else: ax[ax.index('S')] = 'C'
            for i, name in enumerate(list(ax)):           # tifffile's unknown axes: Q / I (and T without Z) -> Z, then C
                if name in 'ZCYX': continue
                if name in 'QIT' and 'Z' not in ax: ax[i] = 'Z'
                elif name in 'QI' and 'C' not in ax: ax[i] = 'C'
            axes = ''.join(ax)
        um_xy, um_z, src, names = None, None, None, []
        if ome and not other_files:
            m = re.search(r'PhysicalSizeX="([\d.eE+-]+)"', ome); mz = re.search(r'PhysicalSizeZ="([\d.eE+-]+)"', ome)
            um_xy = float(m.group(1)) if m else None; um_z = float(mz.group(1)) if mz else None; src = 'OME-XML' if m else None
            names = re.findall(r'<Channel[^>]*Name="([^"]*)"', ome)
        if um_xy is None and t.imagej_metadata:
            ij = t.imagej_metadata; res = t.pages[0].tags.get('XResolution')
            if res and str(ij.get('unit', '')).lower() in ('micron', 'um', 'µm', '\\u00b5m'):
                num, den = res.value; um_xy = den / num if num else None; um_z = ij.get('spacing'); src = 'ImageJ'
    meta.update(um_xy=um_xy, um_z=um_z, um_source=src, channel_names=names)
    return to_zcyx(arr, axes.replace('S', 'C') if 'S' in axes else axes), meta


def _nd2(path, position=0):
    import nd2
    with nd2.ND2File(path) as f:
        sizes = dict(f.sizes); vs = f.voxel_size(); names = [c.channel.name for c in (f.metadata.channels or [])]
        pos = []
        for loop in (f.experiment or []):
            if type(loop).__name__ == 'XYPosLoop':
                pos = [dict(index=i, name=getattr(p, 'name', None) or f'position {i}', stage_um=list(getattr(p, 'stagePositionUm', []) or [])) for i, p in enumerate(loop.parameters.points)]
        if not pos: pos = [dict(index=i, name=f'position {i}') for i in range(sizes.get('P', 1))]
        d = f.to_dask(); axes = ''.join(sizes)
        if 'P' in sizes: d = d[(slice(None),) * axes.index('P') + (min(position, sizes['P'] - 1),)]; axes = axes.replace('P', '')
        if 'S' in axes and 'C' not in axes: axes = axes.replace('S', 'C')
        arr = to_zcyx(d, axes)
        meta = dict(format='nd2', positions=pos, um_xy=float(vs.x) if vs else None, um_z=float(vs.z) if vs and 'Z' in sizes else None,
                    um_source='ND2 metadata', channel_names=names)
        return np.asarray(arr.compute() if hasattr(arr, 'compute') else arr), meta


def _bioio(path, position=0):
    from bioio import BioImage
    img = BioImage(path); scenes = list(img.scenes); img.set_scene(scenes[min(position, len(scenes) - 1)])
    d = img.get_image_dask_data('ZCYX', T=0); ps = img.physical_pixel_sizes
    meta = dict(format=f'{Path(path).suffix.lower().lstrip(".")} (bioio)', positions=[dict(index=i, name=s) for i, s in enumerate(scenes)],
                um_xy=ps.X, um_z=ps.Z, um_source='file metadata' if ps.X else None, channel_names=list(img.channel_names or []))
    return np.asarray(d.compute()), meta


def _numpy(path, position=0):
    if sniff(path) == 'npy': a = np.load(path, allow_pickle=False)
    else:
        with np.load(path, allow_pickle=False) as z: a = max((z[k] for k in z.files), key=lambda v: v.size)
    return to_zcyx(a, guess_axes(a.shape)), dict(format='npy/npz', positions=[dict(index=0, name='array')], um_xy=None, um_z=None, um_source=None, channel_names=[])


def _image(path, position=0):
    from PIL import Image
    a = np.asarray(Image.open(path)); a = a[..., :3] if a.ndim == 3 and a.shape[2] == 4 else a
    if a.ndim == 3 and np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 0], a[..., 2]): a = a[..., 0]
    return to_zcyx(a, 'YXC' if a.ndim == 3 else 'YX'), dict(format='image', positions=[dict(index=0, name='image')], um_xy=None, um_z=None, um_source=None,
                                                            channel_names=['R', 'G', 'B'] if a.ndim == 3 else [])


def prairie_xml(text):
    """Bruker Prairie View .xml: (um per px, z step um) from micronsPerPixel XAxis and the frames' ZAxis positions."""
    block = re.search(r'key="micronsPerPixel">(.*?)</PVStateValue>', text, re.S); block = block.group(1) if block else ''
    m = re.search(r'index="XAxis"\s+value="([\d.]+)"', block); mz = re.search(r'index="ZAxis"\s+value="([\d.]+)"', block)
    if mz: return (float(m.group(1)) if m else None), float(mz.group(1))
    zs = [float(v) for v in re.findall(r'index="ZAxis">\s*<SubindexedValue subindex="0" value="(-?[\d.]+)"', text)]     # older files: frame positions
    dz = sorted({round(abs(b - a), 4) for a, b in zip(zs, zs[1:]) if abs(b - a) > 1e-6})
    return (float(m.group(1)) if m else None), (statistics.median(dz) if dz else None)


def _zip(path, position=0):
    zf = zipfile.ZipFile(path); infos = [i for i in zf.infolist() if not i.is_dir() and '__MACOSX' not in i.filename and not Path(i.filename).name.startswith('.')]
    if sum(i.file_size for i in infos) > MAX_UNZIPPED: raise Unsupported(f'the archive expands to {sum(i.file_size for i in infos) >> 30} GB (limit {MAX_UNZIPPED >> 30} GB)')
    if all(i.filename.endswith('.npy') for i in infos): return _numpy(path, position)                     # an .npz under another name
    inner = [i for i in infos if Path(i.filename).suffix.lower() in {'.nd2'} | BIOIO_EXT]
    if inner:
        big = max(inner, key=lambda i: i.file_size); tmp = Path(tempfile.mkdtemp(dir=Path(path).parent)) / ('inner' + Path(big.filename).suffix.lower())
        with zf.open(big) as src, tmp.open('wb') as dst:
            while chunk := src.read(1 << 24): dst.write(chunk)
        arr, meta = read(tmp, position); tmp.unlink(); meta['format'] += f' (in zip: {Path(big.filename).name})'; return arr, meta
    planes = [i for i in infos if re.search(r'\.(ome\.)?tiff?$|\.png$', i.filename, re.I) and not re.search(r'reference', i.filename, re.I)]
    if not planes: raise Unsupported('the zip holds no images (TIFF / PNG planes, ND2, CZI, LIF, NPY)')
    chan = lambda n: (lambda m: int(m.group(1)) if m else 0)(re.search(r'(?:^|[_\-. ])Ch(?:annel)?[_-]?(\d+)', Path(n).name, re.I))
    nums = lambda n: [int(v) for v in re.findall(r'\d+', re.sub(r'(?:^|[_\-. ])Ch(?:annel)?[_-]?\d+', '', Path(n).name, flags=re.I))]
    keyed = [(chan(i.filename), nums(i.filename), i) for i in planes]
    width = max(len(k[1]) for k in keyed); vec = lambda k: tuple(k[1] + [0] * (width - len(k[1])))
    varying = [j for j in range(width) if len({vec(k)[j] for k in keyed}) > 1]
    by_ch = {}
    for c, _, i in sorted(keyed, key=lambda k: (k[0], tuple(vec(k)[j] for j in varying))): by_ch.setdefault(c, []).append(i)
    counts = {c: len(v) for c, v in by_ch.items()}
    if len(set(counts.values())) != 1: raise Unsupported(f'channels have different plane counts: {counts}')
    import tifffile
    from PIL import Image
    def plane(info):
        with zf.open(info) as fh:
            data = fh.read()
        if info.filename.lower().endswith('.png'): return np.asarray(Image.open(__import__('io').BytesIO(data)))
        with tifffile.TiffFile(__import__('io').BytesIO(data)) as t: return t.pages[0].asarray()        # this file's own page, never the OME set
    arr = np.stack([np.stack([plane(i) for i in by_ch[c]]) for c in sorted(by_ch)], 1)   # (Z, C, Y, X)
    um_xy = um_z = src = None
    xmls = [i for i in infos if i.filename.lower().endswith('.xml')]
    for x in xmls:
        a, b = prairie_xml(zf.read(x).decode('utf-8', 'replace'))
        if a: um_xy, um_z, src = a, b, f'Prairie View XML ({Path(x.filename).name})'; break
    return arr, dict(format='zip of per-plane images', positions=[dict(index=0, name='stack')], um_xy=um_xy, um_z=um_z, um_source=src,
                     channel_names=[f'Ch{c}' for c in sorted(by_ch)], planes_per_channel=counts[next(iter(counts))])


READERS = dict(tiff=_tiff, nd2=_nd2, zip=_zip, npy=_numpy, image=_image, bioio=_bioio)


def read(path, position=0):
    kind = sniff(path)
    if kind == 'pickle': raise Unsupported('Python pickles are not accepted (opening one can run arbitrary code). Convert it first: tools/convert_lut_pickle.py')
    reader = _numpy if kind == 'zip' and str(path).lower().endswith('.npz') else READERS[kind]
    arr, meta = reader(path, position)
    arr = np.asarray(arr)
    if arr.ndim != 4 or arr.shape[2] < 16 or arr.shape[3] < 16: raise Unsupported(f'unexpected image shape {arr.shape} after reading')
    meta.update(Z=int(arr.shape[0]), C=int(arr.shape[1]), Y=int(arr.shape[2]), X=int(arr.shape[3]), dtype=str(arr.dtype), position=int(position))
    names = meta.get('channel_names') or []
    meta['channel_names'] = [names[c] if c < len(names) and names[c] else f'channel {c}' for c in range(arr.shape[1])]
    meta.setdefault('warnings', [])
    if meta['um_xy'] is None: meta['warnings'].append('no pixel size in the file: enter it by hand')
    if len(meta['positions']) > 1: meta['warnings'].append(f'{len(meta["positions"])} positions in the file: one is analysed at a time (no stitching)')
    return arr, meta


def previews(arr, out_dir, side=384, max_planes=12):
    """Per-channel max projection over <= max_planes planes, p1-p99.8 stretch, PNG <= side px. Returns {channel: filename}."""
    from PIL import Image
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True); Z = arr.shape[0]; step = max(1, Z // max_planes); files = {}
    s = max(1, int(np.ceil(max(arr.shape[2:]) / side)))
    for c in range(arr.shape[1]):
        mip = arr[::step, c, ::s, ::s].astype(np.float32).max(0); lo, hi = np.percentile(mip, [1, 99.8])
        img = (np.clip((mip - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8); name = f'c{c}.png'
        Image.fromarray(img).save(out_dir / name); files[c] = name
    return files


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ivexmatch.readers"); sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('probe'); p.add_argument('file'); p.add_argument('--json', required=True); p.add_argument('--previews'); p.add_argument('--position', type=int, default=0)
    c = sub.add_parser('convert'); c.add_argument('file'); c.add_argument('--out', required=True); c.add_argument('--position', type=int, default=0)
    a = ap.parse_args(argv)
    try:
        arr, meta = read(a.file, a.position)
        if a.cmd == 'probe':
            if a.previews: meta['previews'] = previews(arr, a.previews)
            Path(a.json).write_text(json.dumps(meta, indent=1, default=str))
        else:
            np.save(a.out, arr); print(json.dumps(dict(shape=list(arr.shape), dtype=str(arr.dtype))))
    except Unsupported as e:
        if a.cmd == 'probe': Path(a.json).write_text(json.dumps(dict(error=str(e))))
        print(f'UNSUPPORTED: {e}', file=sys.stderr); sys.exit(3)


if __name__ == '__main__':
    main()
