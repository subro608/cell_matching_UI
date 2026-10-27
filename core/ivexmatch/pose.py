"""PMAC blind pose: find the flip / rotation / scale / translation that maps IV onto EX, with no prior, using PMAC's own appearance
descriptors (soma + neighbourhood crops -> shared UNICOM + heads; the region term is skipped, it depends on orientation).

Method = the C4 blind pose (jy369_twostage_2026-10-12/search2.py, jy369_fov_blind_pose_2026-10-26/fov_blind_pose.py) with the
embedder swapped:
  for flip in (False, True), theta on a grid:
      rotate (and mirror) every IV query plane on an expanded zero canvas; cut soma / neighbourhood crops with PMAC's own
      extract_single (so theta = 0, flip = False gives exactly PMAC's matching descriptors);
      S = (soma cos + neighbourhood cos) / 2 against the EX candidates (EX embedded once);
      mutual nearest neighbours -> RANSAC  E_um ~ s * P_um + t  over a scale band (fov_blind_pose.ransac: every pair x scale is a
      hypothesis, top-5 rescored as one-to-one inliers within 15 um);
  coarse grid (step 10 deg) for the decision and the null; then 1-deg refinement within +-10 deg of the best 3 coarse hypotheses;
  null: K permutations of the EX positions, same pairs, max over the coarse grid -> family-wise p;
  final: Umeyama similarity (register.fit_similarity) on the best hypothesis' inliers in the mirrored-but-unrotated IV frame.
Pose convention (pixels on the ivex working grids, um via IV_UM / EX_UM):
  F(xy) = (W - 1 - x, y) if flip else xy;  EX_um = s * R(theta) @ (F(IV_xy) * IV_UM) + t,  R = [[c, -s], [s, c]] (image y down)."""
from __future__ import annotations
import math, time
import numpy as np
import cv2
import torch

from .pmac.crops import extract_single, to_model_layout
from .preprocess import IV_UM, EX_UM
from .register import fit_similarity

INLIER_UM = 15.0
SCALES = np.arange(0.80, 1.201, 0.05)            # tissue shrinkage / magnification band around the user's voxel sizes (fov_blind_pose)


def canvas_matrix(H, W, theta_deg):
    """Rotation about the plane centre onto the expanded canvas: (M 2x3, (W2, H2))."""
    th = math.radians(theta_deg); c, s = math.cos(th), math.sin(th)
    W2, H2 = int(math.ceil(abs(c) * W + abs(s) * H)), int(math.ceil(abs(s) * W + abs(c) * H))
    R = np.array([[c, -s], [s, c]]); src_c = np.array([(W - 1) / 2, (H - 1) / 2]); dst_c = np.array([(W2 - 1) / 2, (H2 - 1) / 2])
    return np.c_[R, dst_c - R @ src_c], (W2, H2)


def rotate_plane(img, theta_deg, flip=False):
    """Mirror (x) then rotate by theta on an expanded canvas. Returns (out, M) with dst_xy = M[:, :2] @ F(src_xy) + M[:, 2]."""
    if flip: img = img[:, ::-1]
    H, W = img.shape[:2]; M, (W2, H2) = canvas_matrix(H, W, theta_deg)
    out = cv2.warpAffine(np.ascontiguousarray(img), M, (W2, H2), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out, M


def flip_xy(xy, W): xy = np.asarray(xy, float).copy(); xy[:, 0] = W - 1 - xy[:, 0]; return xy


@torch.no_grad()
def embed(model, crops_global, crops_local, chunk=16):
    """PMAC appearance descriptors: (n, 768) soma and neighbourhood, L2-normalized (model.crop_level)."""
    return model.crop_level(to_model_layout(crops_local), 'soma', chunk), model.crop_level(to_model_layout(crops_global), 'neighbourhood', chunk)


def iv_crops(planes, xyz, theta, flip, half_g, half_l):
    """Rotated crops for IV cells xyz = (x, y, z) on the working grid -> (global, local, rotated positions in the canvas px)."""
    W = planes.shape[2]; G, L, P = [], [], np.zeros((len(xyz), 2))
    for z in np.unique(xyz[:, 2].astype(int)):
        rows = np.flatnonzero(xyz[:, 2].astype(int) == z); img, M = rotate_plane(planes[z], theta, flip)
        src = flip_xy(xyz[rows, :2], W) if flip else xyz[rows, :2]; P[rows] = src @ M[:, :2].T + M[:, 2]
        for i, r in enumerate(rows):
            g, l = extract_single(img, P[r], half_g, half_l, 224); G.append((r, g)); L.append((r, l))
    order = np.argsort([r for r, _ in G]); return np.stack([G[i][1] for i in order]), np.stack([L[i][1] for i in order]), P


def mutual(S):
    a = S.argmax(1); b = S.argmax(0); m = torch.nonzero(b[a] == torch.arange(S.shape[0], device=S.device), as_tuple=True)[0]; return m, a[m]


def ransac(P, E, qi, gi, scales, top=5):
    """fov_blind_pose.ransac: E ~ s * P + t. -> (one-to-one inliers, s, t, inlier mask)."""
    n = len(P)
    if n < 2: return 0, float('nan'), [float('nan')] * 2, None
    cand = []
    for s in scales:
        t = E - float(s) * P; cnt = torch.zeros(n, device=P.device, dtype=torch.int32)
        for c0 in range(0, n, 4096): cnt[c0:c0 + 4096] = (torch.cdist(t[c0:c0 + 4096], t) <= INLIER_UM).sum(1).int()
        cand += [(int(cnt[j]), float(s), t[j]) for j in torch.topk(cnt, min(top, n)).indices.tolist()]
    cand.sort(key=lambda c: -c[0]); best = (0, float('nan'), [float('nan')] * 2, None)
    for raw, s, tj in cand[:top]:
        inl = torch.linalg.norm(E - s * P - tj, dim=1) <= INLIER_UM
        sc = min(len(torch.unique(qi[inl])), len(torch.unique(gi[inl])))
        if sc > best[0]: best = (sc, s, tj.tolist(), inl)
    return best


@torch.no_grad()
def blind_pose(pmac, iv_planes, iv_cells, ex_image, ex_cells, n_query=300, coarse_step=10, fine_halfwidth=10, n_null=50, seed=0, log=print):
    """pmac: ivexmatch.pmac.PMAC. iv_planes uint8 (Z, H, W) on IV_UM (B-normalized, unrotated); iv_cells (N, 3) = (x, y, z) px;
    ex_image uint8 (H, W[, 3]) on EX_UM; ex_cells (M, 2) px. Returns a JSON-able dict (pose, p-value, hypotheses, inlier pairs)."""
    t0 = time.time(); dev = pmac.device; rng = np.random.default_rng(seed); model = pmac.model; scale = pmac.crop_scale
    iv_cells = np.asarray(iv_cells, float); ex_cells = np.asarray(ex_cells, float)
    q = iv_cells if len(iv_cells) <= n_query else iv_cells[np.sort(rng.choice(len(iv_cells), n_query, replace=False))]
    half_g, half_l = np.full(2, 75.) / scale, np.full(2, 20.) / scale
    eg, el = zip(*[extract_single(ex_image, xy, np.full(2, 75.), np.full(2, 20.), 224) for xy in ex_cells])
    Es, En = embed(model, np.stack(eg), np.stack(el)); E_um = torch.as_tensor(ex_cells * EX_UM, dtype=torch.float32, device=dev)
    perms = [torch.from_numpy(np.random.default_rng(1000 + k).permutation(len(ex_cells))).to(dev) for k in range(n_null)]
    hyps = {}

    def evaluate(flip, theta, with_null):
        G, L, P = iv_crops(iv_planes, q, theta, flip, half_g, half_l); Qs, Qn = embed(model, G, L)
        S = (Qs @ Es.T + Qn @ En.T) / 2; qi, gi = mutual(S); Pum = torch.as_tensor(P * IV_UM, dtype=torch.float32, device=dev)
        sc, s, t, inl = ransac(Pum[qi], E_um[gi], qi, gi, SCALES)
        h = dict(flip=bool(flip), theta=float(theta), score=int(sc), scale=s, t=t, n_pairs=int(len(qi)),
                 inliers=None if inl is None else [qi[inl].cpu().numpy(), gi[inl].cpu().numpy()])
        if with_null: h['null'] = [ransac(Pum[qi], E_um[p][gi], qi, gi, SCALES)[0] for p in perms]
        hyps[(bool(flip), float(theta))] = h; return h

    coarse = [(f, th) for f in (False, True) for th in np.arange(0, 360, coarse_step)]
    for i, (f, th) in enumerate(coarse):
        h = evaluate(f, th, True)
        if i % 6 == 0: log(f'coarse flip={f} theta={th:g}: inliers {h["score"]} of {h["n_pairs"]} pairs ({time.time() - t0:.0f}s)')
    best_coarse = max(hyps.values(), key=lambda h: h['score'])
    null_max = np.max([[hyps[(f, float(th))]['null'][k] for f, th in coarse] for k in range(n_null)], axis=1) if n_null else np.zeros(0)
    p = float((1 + (null_max >= best_coarse['score']).sum()) / (n_null + 1)) if n_null else float('nan')
    top3 = sorted([hyps[(f, float(th))] for f, th in coarse], key=lambda h: -h['score'])[:3]
    for h0 in top3:
        for d in range(-fine_halfwidth, fine_halfwidth + 1):
            th = (h0['theta'] + d) % 360
            if (h0['flip'], float(th)) not in hyps: evaluate(h0['flip'], th, False)
    ranked = sorted(hyps.values(), key=lambda h: -h['score']); best = ranked[0]
    runner = next((h for h in ranked[1:] if h['flip'] != best['flip'] or min(abs(h['theta'] - best['theta']) % 360, 360 - abs(h['theta'] - best['theta']) % 360) > 2 * coarse_step), None)
    out = dict(method='pmac_blind_pose_v1', n_query=int(len(q)), n_ex=int(len(ex_cells)), coarse_step=coarse_step, n_null=n_null,
               p_familywise=p, null_max=null_max.tolist(), coarse_best=dict(flip=best_coarse['flip'], theta=best_coarse['theta'], score=best_coarse['score']),
               best=dict(flip=best['flip'], theta_grid=best['theta'], score=best['score'], ransac_scale=best['scale']),
               runner_up=None if runner is None else dict(flip=runner['flip'], theta=runner['theta'], score=runner['score'], scale=runner['scale'], t=runner['t']),
               hypotheses=[dict(flip=h['flip'], theta=h['theta'], score=h['score'], n_pairs=h['n_pairs']) for h in ranked], seconds=None)
    if best['inliers'] is not None and len(best['inliers'][0]) >= 3:
        qi, gi = best['inliers']; X = (flip_xy(q[qi, :2], iv_planes.shape[2]) if best['flip'] else q[qi, :2]) * IV_UM; Y = ex_cells[gi] * EX_UM
        A, t = fit_similarity(X, Y); s = float(np.sqrt(np.linalg.det(A))); theta = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])) % 360)
        res = np.linalg.norm(X @ A.T + t - Y, axis=1)
        out['pose'] = dict(flip=best['flip'], theta=theta, scale=s, t_um=t.tolist(), matrix=A.tolist(), iv_width_px=int(iv_planes.shape[2]),
                           iv_um=IV_UM, ex_um=EX_UM, n_inliers=int(len(qi)), median_residual_um=float(np.median(res)))
        out['inlier_pairs'] = dict(iv=q[qi].tolist(), ex=ex_cells[gi].tolist())
    else:
        out['pose'] = None
    out['uncertain'] = bool(out['pose'] is None or not (p < 0.05))
    out['seconds'] = round(time.time() - t0, 1); log(f'blind pose: {out["best"]} p={p:.3f} ({out["seconds"]}s)')
    return out


def pose_to_iv(pose, iv_xy):
    """Map IV working-grid px -> EX working-grid px with a pose dict."""
    xy = flip_xy(iv_xy, pose['iv_width_px']) if pose['flip'] else np.asarray(iv_xy, float)
    return ((xy * pose['iv_um']) @ np.asarray(pose['matrix']).T + np.asarray(pose['t_um'])) / pose['ex_um']


def hypothesis_affine(h, grid_hw, iv_xy):
    """IV working-grid px -> EX working-grid px for a RANSAC hypothesis (flip, theta, scale, t): EX_um = s * M F(xy) IV_UM + t."""
    M, _ = canvas_matrix(grid_hw[0], grid_hw[1], h['theta']); xy = flip_xy(iv_xy, grid_hw[1]) if h['flip'] else np.asarray(iv_xy, float)
    return (float(h['scale']) * (xy @ M[:, :2].T + M[:, 2]) * IV_UM + np.asarray(h['t'], float)) / EX_UM


def pose_from_native_affine(A, iv_native_hw, iv_grid_hw, ex_native_hw, ex_grid_hw):
    """A user's orientation (2x3 native IV px -> native EX px, from the viewer's Set manually) -> dict(flip, theta) for apply_pose.
    Only the rotation and the mirror matter for PMAC (the location branch is translation invariant within the EX overview)."""
    L = np.asarray(A, float)[:, :2] * (np.asarray(ex_grid_hw[::-1], float) / np.asarray(ex_native_hw[::-1], float))[:, None] \
        / (np.asarray(iv_grid_hw[::-1], float) / np.asarray(iv_native_hw[::-1], float))[None, :]
    flip = bool(np.linalg.det(L) < 0); L = L @ np.diag([-1., 1.]) if flip else L
    return dict(flip=flip, theta=float(np.degrees(np.arctan2(L[1, 0], L[0, 0])) % 360), kind='manual')


def apply_pose(iv_planes, iv_cells, pose):
    """Mirror / rotate the IV volume by the pose so it is upright relative to EX before PMAC matching (location branch tolerance is
    about -16..+12 deg). Returns (planes (Z, H2, W2), cells (N, 3) with x, y in the new frame)."""
    out, xy = [], None; cells = np.asarray(iv_cells, float).copy(); W = iv_planes.shape[2]
    for z in range(len(iv_planes)):
        img, M = rotate_plane(iv_planes[z], pose['theta'], pose['flip']); out.append(img)
    src = flip_xy(cells[:, :2], W) if pose['flip'] else cells[:, :2]; cells[:, :2] = src @ M[:, :2].T + M[:, 2]
    return np.stack(out), cells
