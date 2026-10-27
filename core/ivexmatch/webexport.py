"""Web bundle for the viewer (static files; the same layout is served locally or from Vercel Blob through signed URLs).

  manifest.json                       sides, voxel sizes, image files, mask sets, runs (matches + pose), files[] (for URL signing)
  iv.ome.tif / ex.ome.tif             pyramidal tiled OME-TIFF (SubIFD levels, deflate) read by Viv over HTTP range requests; uint8
                                      for sessions (each file must stay < 512 MB, the Blob CDN cache limit)
  masks/<side>/<set>.json.gz          {"z": [[id, cx, cy, x0, y0, x1, y1, ...], ...]} outline polygons in native px, one file per set
                                      (the local JY369 demo uses the older one-file-per-plane layout: masks/<side>/<set>/z###.json)
  matches_<run>.json.gz               per IV cell-plane: id, z, x, y, top-k EX candidates with scores, P(top1), mutual-best
  pose.json                           blind pose card (p, inliers vs null, hypotheses, inlier pairs) + native-pixel 2x3 affine IV -> EX
Coordinates are native pixels of each side's image (x right, y down); micrometres via manifest um_xy. Few, large files on purpose:
every Blob upload is one operation of the free tier's 2,000 a month."""
from __future__ import annotations
import gzip, json
from pathlib import Path
import numpy as np
import cv2


def write_pyramid(path, data, axes, um_xy, um_z=None, channel_names=None, tile=256, min_side=512):
    """data (Z, Y, X) or (C, Y, X); writes level 0 + SubIFD levels (2x mean downsampling) until the long side <= min_side."""
    import tifffile
    data = np.asarray(data); levels = []; cur = data
    while max(cur.shape[-2:]) > min_side:
        h, w = cur.shape[-2] // 2, cur.shape[-1] // 2
        cur = np.stack([cv2.resize(p, (w, h), interpolation=cv2.INTER_AREA) for p in cur]).astype(data.dtype); levels.append(cur)
    meta = dict(axes=axes, PhysicalSizeX=float(um_xy), PhysicalSizeY=float(um_xy), PhysicalSizeXUnit='µm', PhysicalSizeYUnit='µm')
    if um_z: meta.update(PhysicalSizeZ=float(um_z), PhysicalSizeZUnit='µm')
    if channel_names: meta['Channel'] = dict(Name=list(channel_names))
    opts = dict(tile=(tile, tile), compression='zlib', photometric='minisblack')
    with tifffile.TiffWriter(path, bigtiff=True, ome=True) as tif:
        tif.write(data, subifds=len(levels), metadata=meta, **opts)
        for lev in levels: tif.write(lev, subfiletype=1, metadata=None, **opts)
    return dict(file=Path(path).name, shape=list(data.shape), axes=axes, levels=1 + len(levels), dtype=str(data.dtype),
                um_xy=float(um_xy), um_z=um_z, channels=list(channel_names) if channel_names else None)


def outlines(labels, scale=1.0, eps=0.6):
    """2D label image -> [[id, cx, cy, x0, y0, ...]] (outer contour, approxPolyDP eps px), rounded to 0.1 px. scale maps this label
    grid onto the image's native grid with the pixel-centre rule (v + .5) * scale - .5."""
    out = []; lab = np.asarray(labels)
    if not lab.any(): return out
    from scipy import ndimage as ndi
    s = lambda v: round((v + .5) * scale - .5, 1)
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None: continue
        sub = (lab[sl] == i).astype(np.uint8)
        cs, _ = cv2.findContours(np.pad(sub, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs: continue
        c = max(cs, key=cv2.contourArea); c = cv2.approxPolyDP(c, eps, True)[:, 0].astype(float)
        c += np.array([sl[1].start - 1, sl[0].start - 1])                         # back to label px (pad removed)
        yy, xx = np.nonzero(sub); cx, cy = xx.mean() + sl[1].start, yy.mean() + sl[0].start
        out.append([int(i), s(cx), s(cy)] + [s(v) for v in c.ravel()])
    return out


def write_mask_set(out_dir, side, name, planes, scale=1.0, color=None, label=None):
    """planes: {z: 2D labels}. Returns the manifest entry."""
    d = Path(out_dir) / 'masks' / side / name; d.mkdir(parents=True, exist_ok=True); counts = {}
    for z, lab in planes.items():
        polys = outlines(lab, scale); counts[int(z)] = len(polys)
        (d / f'z{int(z):03d}.json').write_text(json.dumps(polys, separators=(',', ':')))
    return dict(name=name, label=label or name, color=color, planes={str(z): f'masks/{side}/{name}/z{z:03d}.json' for z in counts}, counts=counts)


def native_affine(fn, pts=((0., 0.), (1000., 0.), (0., 1000.))):
    """2x3 affine [A | t] of a point map fn(xy (n, 2)) -> (n, 2) that is affine; solved exactly from 3 points."""
    P = np.asarray(pts, float); Q = np.asarray(fn(P), float); M, *_ = np.linalg.lstsq(np.c_[P, np.ones(3)], Q, rcond=None); return M.T.tolist()


MODEL_STYLE = dict(cyto2_ft=([0, 220, 255], 'fine-tuned cyto2'), cyto3_ft=([255, 80, 200], 'fine-tuned cyto3'), cpsam=([255, 200, 0], 'CP-SAM (dk5288 fine-tune)'))
MAX_FILE = 500 << 20                                                    # Blob CDN caches blobs up to 512 MB


def to_uint8(a, p=(0.5, 99.8)):
    """Display copy: percentile stretch over a sample of the nonzero voxels (zeros = outside the field)."""
    a = np.asarray(a); s = a.reshape(-1)[:: max(1, a.size // 4_000_000)]; nz = s[s != 0]; lo, hi = np.percentile(nz if nz.size else s, p)
    return (np.clip((a.astype(np.float32) - lo) / max(float(hi - lo), 1e-6), 0, 1) * 255).astype(np.uint8)


def write_json_gz(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', compresslevel=6) as f: json.dump(obj, f, separators=(',', ':'))


def write_mask_set_packed(out_dir, side, name, planes, scale=1.0, color=None, label=None):
    """planes: {z: 2D labels} -> masks/<side>/<name>.json.gz. Returns the manifest entry."""
    data = {str(int(z)): outlines(lab, scale) for z, lab in planes.items()}; rel = f'masks/{side}/{name}.json.gz'
    write_json_gz(Path(out_dir) / rel, data)
    return dict(name=name, label=label or name, color=color, file=rel, counts={z: len(v) for z, v in data.items()})


def _native_pose(fn_grid, iv_meta, ex_meta):
    """2x3 native IV px -> native EX px affine of a working-grid point map fn_grid (IV grid px -> EX grid px)."""
    from .preprocess import to_grid, from_grid
    return native_affine(lambda P: from_grid(fn_grid(to_grid(P, iv_meta['native_hw'], iv_meta['grid_hw'])), ex_meta['native_hw'], ex_meta['grid_hw']))


def matches_record(topk, ex_cells, match_model, note=''):
    """pmac/<run>/topk.json (ids) -> the viewer's matches format (indices into the EX candidate list)."""
    ex_list = [[round(float(c[2]), 1), round(float(c[3]), 1), 0, int(c[1])] for c in ex_cells]; index = {int(c[1]): i for i, c in enumerate(ex_cells)}
    ix = lambda ids: [index[int(i)] for i in ids]; r3 = lambda v: [round(float(x), 3) for x in v]
    cells = [[c['id'], c['z'], c['x'], c['y'], c['p_top1'], c['mutual'], ix(c['top_ex_ids']), r3(c['combined']), ix(c['appearance_top_ex_ids']),
              r3(c['appearance']), ix(c['location_top_ex_ids']), r3(c['location'])] for c in topk['cells']]
    return dict(columns=['id', 'z', 'x', 'y', 'p_top1', 'mutual', 'combined_idx', 'combined', 'appearance_idx', 'appearance', 'location_idx', 'location'],
                ex_columns=['x', 'y', 'mask_set', 'mask_id'], ex_mask_sets=[match_model], iv_mask_set=match_model, ex=ex_list, cells=cells,
                note=note or 'combined = log P(EX cell | IV query); appearance / location = term scores')


def pose_card(pose, iv_meta, ex_meta, iv_cells_grid_to_native, ex_grid_to_native):
    """pose/pose.json (working grids) -> the viewer's pose card with native-pixel geometry."""
    from .pose import pose_to_iv
    best = pose.get('pose'); card = dict(method=pose['method'], label='PMAC blind pose', provisional=False, p_familywise=pose['p_familywise'],
                                         uncertain=pose['uncertain'], n_query=pose['n_query'], n_ex=pose['n_ex'], n_null=pose['n_null'],
                                         null_max=pose['null_max'], best=pose['best'], runner_up=pose['runner_up'], seconds=pose['seconds'],
                                         hypotheses=pose['hypotheses'][:72])
    if best:
        card.update(rotation_deg=best['theta'], scale=best['scale'], flip=best['flip'], n_inliers=best['n_inliers'], median_residual_um=best['median_residual_um'],
                    iv_to_ex_native=_native_pose(lambda P: pose_to_iv(best, P), iv_meta, ex_meta))
        ip = pose.get('inlier_pairs') or {}
        if ip: card['inlier_pairs'] = dict(iv=[[*map(float, iv_cells_grid_to_native(np.asarray(r[:2])[None])[0]), int(r[2])] for r in ip['iv']],
                                           ex=[list(map(float, ex_grid_to_native(np.asarray(r)[None])[0])) for r in ip['ex']])
    return card


def export_session(analysis_dir, web_dir, title=None, runs_only=False, log=print):
    """Analysis bundle (cli.analyze) -> web bundle. Images from work/, masks from masks/<side>/<model>.npz, runs from pmac/run*/.
    runs_only: after a rematch, rewrite only the matches files and the manifest (images, masks and pose card are unchanged)."""
    from .preprocess import from_grid
    A, Wd = Path(analysis_dir), Path(web_dir); Wd.mkdir(parents=True, exist_ok=True); meta = json.loads((A / 'meta.json').read_text()); args = meta['args']
    iv_meta, ex_meta = meta['iv'], meta['ex']; files = []; old = json.loads((Wd / 'manifest.json').read_text()) if runs_only else None
    iv = np.load(A / 'work/iv_native.npy', mmap_mode='r')
    if runs_only: return _export_runs(A, Wd, meta, args, old, log)
    iv_img = write_pyramid(Wd / 'iv.ome.tif', to_uint8(iv), 'ZYX', args['iv_um'], args.get('iv_z_um'))
    gfp = np.load(A / 'work/ex_mip.npy'); dapi = np.load(A / 'work/ex_dapi_mip.npy') if (A / 'work/ex_dapi_mip.npy').exists() else None
    chans = ['GFP'] + (['DAPI'] if dapi is not None else [])
    ex_img = write_pyramid(Wd / 'ex.ome.tif', np.stack([to_uint8(gfp)] + ([to_uint8(dapi)] if dapi is not None else [])), 'CYX', args['ex_um'], channel_names=chans)
    for f in ('iv.ome.tif', 'ex.ome.tif'):
        assert (Wd / f).stat().st_size < MAX_FILE, f'{f} is {(Wd / f).stat().st_size >> 20} MB; the viewer needs < 500 MB per file'
    files += ['iv.ome.tif', 'ex.ome.tif']; log(f'pyramids: IV {iv_img["shape"]}, EX {ex_img["shape"]}')
    sets = dict(iv=[], ex=[])
    for side in ('iv', 'ex'):
        for name in args['models'].split(','):
            m = np.load(A / 'masks' / side / f'{name}.npz'); lab = m['labels']; color, label = MODEL_STYLE.get(name, ([255, 255, 255], name))
            planes = {z: lab[z] for z in range(len(lab))} if lab.ndim == 3 else {0: lab}
            sets[side].append(write_mask_set_packed(Wd, side, name, planes, color=color, label=label + (' · drives matching' if name == args['match_model'] else '')))
            files.append(sets[side][-1]['file'])
    log('masks: ' + ', '.join(f"{s} {e['name']} {sum(e['counts'].values())}" for s in sets for e in sets[s]))
    pose = json.loads((A / 'pose/pose.json').read_text())
    iv_native = lambda P: from_grid(P, iv_meta['native_hw'], iv_meta['grid_hw']); ex_native = lambda P: from_grid(P, ex_meta['native_hw'], ex_meta['grid_hw'])
    card = pose_card(pose, iv_meta, ex_meta, iv_native, ex_native); (Wd / 'pose.json').write_text(json.dumps(card, indent=1)); files.append('pose.json')
    manifest = dict(version='ivex-web-2', title=title or args.get('title') or 'ivex session', private=True, created=meta.get('created'),
                    iv=dict(image=iv_img, um_xy=args['iv_um'], um_z=args.get('iv_z_um'), nz=int(iv.shape[0]), mask_sets=sets['iv'], channel=f"channel {args['iv_channel']}",
                            default_z=_default_z(sets['iv'], int(iv.shape[0]))),
                    ex=dict(image=ex_img, um_xy=args['ex_um'], mask_sets=sets['ex'], channels=chans, projection=f"max over planes {args['ex_z'] or 'all'}"),
                    pose='pose.json', models=args['models'].split(','), files=files)
    return _export_runs(A, Wd, meta, args, manifest, log, card)


def _default_z(iv_sets, nz):
    counts = iv_sets[0]['counts'] if iv_sets else {}
    return int(max(range(nz), key=lambda z: counts.get(str(z), 0))) if counts else nz // 2


def _export_runs(A, Wd, meta, args, manifest, log, card=None):
    from .pose import hypothesis_affine
    iv_meta, ex_meta = meta['iv'], meta['ex']; card = card or json.loads((Wd / 'pose.json').read_text()); ex_cells = {}
    files = [f for f in manifest['files'] if not f.startswith('matches_') and f != 'manifest.json']
    runs = []
    for name in meta['runs']:
        topk = json.loads((A / 'pmac' / name / 'topk.json').read_text()); h = topk['pose']; rel = f'matches_{name}.json.gz'; mm = topk.get('match_model', args['match_model'])
        if mm not in ex_cells: ex_cells[mm] = np.load(A / 'masks/ex' / f'{mm}.npz')['cells']
        write_json_gz(Wd / rel, matches_record(topk, ex_cells[mm], mm)); files.append(rel)
        kind = h.get('kind') or ('none' if h.get('unposed') else 'runner_up' if h.get('runner_up') else 'best')
        aff = card.get('iv_to_ex_native') if kind == 'best' else h.get('iv_to_ex_native') if kind == 'manual' else _native_pose(lambda P: hypothesis_affine(h, iv_meta['grid_hw'], P), iv_meta, ex_meta) if h.get('t') else None
        label = dict(best='best pose', runner_up='runner-up pose', manual='your orientation', none='as uploaded (no pose)')[kind]
        runs.append(dict(name=name, label=f"{label} · flip {'yes' if h.get('flip') else 'no'} · {h.get('theta', 0):.1f}°", kind=kind, matches=rel,
                         flip=bool(h.get('flip')), theta=float(h.get('theta', 0)), iv_to_ex_native=aff, match_model=mm))
    manifest = dict(manifest, runs=runs, match_model=runs[0]['match_model'] if runs else args['match_model'], files=files + ['manifest.json'],
                    analysis=dict(seconds=meta['seconds'], gpu=meta['gpu'], pmac=meta['pmac']))
    (Wd / 'manifest.json').write_text(json.dumps(manifest, indent=1)); log(f'web bundle -> {Wd} ({len(manifest["files"])} files, {len(runs)} runs)')
    return manifest
