"""PMAC inference on arrays: one IV volume + its query cells, one EX image + its candidate cells.

Inputs are already on the checkpoint's working grid (see ivexmatch.preprocess): IV uint8 (Z, H, W) B-normalized, EX uint8
(H, W) grayscale or (H, W, 3) composite (DAPI, GFP, DAPI). Records follow the training code's layout (`a` = cells and lazy
crops, `f` = letterboxed fields), so the vendored model runs unchanged:
  * IV crops: plane q_z, centre q_xy, half sizes 75 / crop_scale and 20 / crop_scale px (data.PixelBank.crops, B100);
  * EX crops: centre ex_xy, half sizes 75 and 20 px;
  * overview: the IV plane cut to `iv_box` (default whole plane) and the whole EX image, each letterboxed.
`score()` is run_pmac.score (JY369, 10-25) and `combined_full` is run_gd.combined_full, both verified against model.forward."""
from __future__ import annotations
import numpy as np
import torch

from .crops import extract_single, to_model_layout
from .vendor import data as vdata

TERMS = ('combined', 'soma+neighbourhood', 'region')


class LazyCrops:
    """Crops of selected cells, cut only when sliced (model.crop_level reads gray[s:s + chunk]), like data.GpuCrops; np.asarray()
    materializes all of them (tests)."""
    def __init__(self, view, sel): self.view, self.sel = view, sel

    def __len__(self): return len(self.sel)

    def __getitem__(self, sl):
        ids = self.sel[sl]; scalar = np.ndim(ids) == 0; ids = np.atleast_1d(ids)
        out = to_model_layout(np.stack([self.view.crop(i) for i in ids])) if len(ids) else np.zeros((0, 224, 224), np.uint8)
        return out[0] if scalar else out

    def __array__(self, dtype=None, copy=None):
        out = self[0:len(self)]; return out if dtype is None else out.astype(dtype)


class CropView:
    """Crops for describe(): view[ids] -> LazyCrops of uint8 (n, 224, 224) or (n, 3, 224, 224); local=True is the soma window."""
    def __init__(self, planes, xy, z, half_global, half_local, local):
        self.planes, self.xy, self.z, self.local = planes, np.asarray(xy, np.float64), np.asarray(z, np.int64), local
        self.half = (np.asarray(half_global, float), np.asarray(half_local, float))

    def __len__(self): return len(self.xy)

    def crop(self, i): return extract_single(self.planes[int(self.z[i])], self.xy[i], *self.half, 224)[int(self.local)]

    def __getitem__(self, ids):
        sel = np.arange(len(self))[ids] if isinstance(ids, slice) else np.asarray(ids)
        if np.ndim(sel) == 0: return to_model_layout(self.crop(int(sel))[None])[0]
        return LazyCrops(self, sel.astype(np.int64))


def build_record(iv, q_xy, q_z, ex, ex_xy, crop_scale, iv_box=None):
    """(a, f) in the training code's layout. iv: (Z, H, W) uint8; q_xy (N, 2) IV px; q_z (N,) planes; ex: (H, W[, 3]) uint8;
    ex_xy (M, 2) EX px; iv_box = [x0, y0, x1, y1] of the IV overview (None = whole plane)."""
    iv = np.asarray(iv); ex = np.asarray(ex); assert iv.dtype == np.uint8 and ex.dtype == np.uint8 and iv.ndim == 3 and ex.ndim in (2, 3)
    q_xy = np.asarray(q_xy, np.float32).reshape(-1, 2); q_z = np.asarray(q_z, np.int64).reshape(-1); ex_xy = np.asarray(ex_xy, np.float32).reshape(-1, 2)
    assert len(q_xy) == len(q_z) and len(ex_xy) and len(q_xy); scale = np.asarray(crop_scale, float)
    x0, y0, x1, y1 = iv_box if iv_box is not None else (0, 0, iv.shape[2], iv.shape[1])
    raw_iv = {int(z): np.asarray(iv[int(z)]) for z in np.unique(q_z)}
    padded_ex, eg = vdata.letterbox(ex); padded_iv = {}; ig = None
    for z, image in raw_iv.items():
        padded_iv[z], g = vdata.letterbox(image[y0:y1, x0:x1])
        assert ig is None or ig == g; ig = g
    f = dict(ex=padded_ex, iv=padded_iv, ex_geom=eg, iv_geom=ig, q_xy=vdata.map_points(q_xy - np.asarray([x0, y0], np.float32), ig),
             ex_xy=vdata.map_points(ex_xy, eg), ex_valid=vdata.content_mask(eg), iv_scale=scale, raw_ex=ex, raw_iv=raw_iv,
             iv_full_xy=q_xy, iv_overview_box=[x0, y0, x1, y1])
    a = dict(n_gt=np.asarray(len(q_xy), np.int64), _subject='ivex', _region='upload', _domain='real', q_xy=q_xy, q_z=q_z, ex_xy=ex_xy,
             background=np.zeros(len(ex_xy), bool))
    planes_ex = {0: ex}; zeros = np.zeros(len(ex_xy), np.int64)
    a['q_global'] = CropView(raw_iv, q_xy, q_z, np.full(2, 75.) / scale, np.full(2, 20.) / scale, False)
    a['q_local'] = CropView(raw_iv, q_xy, q_z, np.full(2, 75.) / scale, np.full(2, 20.) / scale, True)
    a['ex_global'] = CropView(planes_ex, ex_xy, zeros, np.full(2, 75.), np.full(2, 20.), False)
    a['ex_local'] = CropView(planes_ex, ex_xy, zeros, np.full(2, 75.), np.full(2, 20.), True)
    return a, f


class PMAC:
    """Frozen PMAC. checkpoint: a training `best*.pt` ({state, arm, ...}) or an ivex release file ({state, arm, crop_scale, ...})."""
    def __init__(self, checkpoint, crop_scale=None, device=None):
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False) if not isinstance(checkpoint, dict) else checkpoint
        scale = ck.get('crop_scale', crop_scale); assert scale is not None, 'crop_scale missing: pass the fold/run scale.json value'
        if crop_scale is not None: assert np.allclose(scale, crop_scale), (scale, crop_scale)
        self.arm, self.crop_scale = ck['arm'], np.asarray(scale, float)
        vdata.set_crop_scale(self.crop_scale)
        from .vendor import model as vmodel, resolution_base as base
        self.vmodel, self.base = vmodel, base
        self.model = vmodel.ResolutionPMAC(self.arm)
        self.model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck['state'].items()}, strict=True)
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device).eval(); self.model.iv_scale = self.crop_scale
        self.composite = self.vmodel.CONFIGS[self.arm].get('candidates') == 'gfp+dapi'

    def record(self, iv, q_xy, q_z, ex, ex_xy, iv_box=None):
        return build_record(iv, q_xy, q_z, ex, ex_xy, self.crop_scale, iv_box)

    def combined_full(self, local, heat, spatial_valid, ex_xy, block=2048, rows=4):
        """run_gd.combined_full: combined[q, c] = local[q, c] + logsumexp_g( log P(g | q) + member[g, c] - logsumexp_c' (local[q, c'] + member[g, c']) )."""
        model, base = self.model, self.base
        g = model.geom; centres = base.spatial_centres(model.grid, g, local.device)[spatial_valid]
        xy = (ex_xy - ex_xy.new_tensor(g['offset_xy']) + .5) / (ex_xy.new_tensor(g['resized_wh']) / ex_xy.new_tensor(g['native_wh'])) - .5
        logp = heat[:, spatial_valid].log_softmax(-1); acc = torch.full_like(local, -torch.inf)
        for s in range(0, len(centres), block):
            member = -((centres[s:s + block, None] - xy[None]) ** 2).sum(-1) / (2 * base.SIGMA_NATIVE_PX ** 2)
            for k in range(0, len(local), rows):
                norm = torch.logsumexp(local[k:k + rows, None] + member[None], -1)
                acc[k:k + rows] = torch.logaddexp(acc[k:k + rows], torch.logsumexp((logp[k:k + rows, s:s + block] - norm)[:, :, None] + member[None], 1))
        return local + acc

    @torch.no_grad()
    def full_terms(self, a, f, ids=None, cids=None):
        """run_gd.full_terms through model.forward (small galleries / parity tests): score matrices per term, (n_q, n_c)."""
        ids = np.arange(len(a['q_xy'])) if ids is None else np.asarray(ids); cids = np.arange(len(a['ex_xy'])) if cids is None else np.asarray(cids)
        q, e, heat, mask = self.model(a, f, ids, cids)
        assert torch.isfinite(q).all() and torch.isfinite(e).all()
        terms = {n: q[:, i] @ e[:, i].T for i, n in enumerate(self.vmodel.LEVELS)}
        terms['soma+neighbourhood'] = (terms['soma'] + terms['neighbourhood']) / 2
        terms['combined'] = self.combined_full(terms['soma+neighbourhood'] / self.vmodel.TAU, heat, mask, torch.as_tensor(f['ex_xy'][cids], device=q.device))
        return terms

    @torch.no_grad()
    def score(self, a, f, k=10):
        """run_pmac.score: EX side once, then per IV plane; top-k per term over the whole gallery, P(top1), mutual-best flag."""
        model, LEVELS, TAU = self.model, self.vmodel.LEVELS, self.vmodel.TAU
        dev = model.device; nc = len(a['ex_xy']); k = min(k, nc)
        e, maps = model.describe(a, f, 'ex', np.arange(nc), True); overview = model.overview.flatten(1).T
        model.geom = f['ex_geom']; model.grid = maps[0].shape[-1]
        valid = torch.as_tensor(vdata.content_mask(model.geom, model.grid).reshape(-1), device=dev)
        coarse_valid = torch.as_tensor(vdata.content_mask(model.geom, 14).reshape(-1), device=dev); ex_xy = torch.as_tensor(f['ex_xy'], device=dev)
        res = {f'{t}_{s}': np.zeros((len(a['q_xy']), k), np.int32 if s == 'idx' else np.float32) for t in TERMS for s in ('idx', 'score')}
        res['mutual'] = np.zeros(len(a['q_xy']), bool)
        for z in np.unique(a['q_z']):
            ids = np.flatnonzero(a['q_z'] == z)
            q = model.describe(a, f, 'q', ids); region = model.conditioner(q, overview, torch.as_tensor(f['q_xy'][ids], device=dev), coarse_valid)
            q = torch.stack([q[:, 0], q[:, 1], region], 1); heat = q[:, 2] @ maps[0].flatten(1) / TAU
            terms = {n: q[:, i] @ e[:, i].T for i, n in enumerate(LEVELS)}
            terms['soma+neighbourhood'] = (terms['soma'] + terms['neighbourhood']) / 2
            terms['combined'] = self.combined_full(terms['soma+neighbourhood'] / TAU, heat, valid, ex_xy)
            for t in TERMS:
                top = torch.topk(terms[t], k, dim=1); res[f'{t}_idx'][ids] = top.indices.cpu().numpy(); res[f'{t}_score'][ids] = top.values.cpu().numpy()
            best_q = terms['combined'].argmax(0).cpu().numpy(); top1 = res['combined_idx'][ids, 0]; res['mutual'][ids] = best_q[top1] == np.arange(len(ids))
            del terms
            if dev.type == 'cuda': torch.cuda.empty_cache()
        res['p_top1'] = np.exp(res['combined_score'][:, 0])
        return res
