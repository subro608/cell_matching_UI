"""Landmark registration from verified IV -> EX pairs: global and slice-wise similarity / affine / thin-plate spline, with
leave-one-out error. Coordinates in micrometres: IV (x, y, z), EX (x, y).

fit_similarity / ransac are refine_pose.py (jy369_twostage_2026-10-12) verbatim: proper rotations only, so a mirrored IV must be
flipped before fitting (ivexmatch.pose does). The affine is prior_eval_affine.affine_fit; the TPS is oracle_registration_ceiling.tps
(scipy RBFInterpolator, thin_plate_spline, exact interpolation)."""
from __future__ import annotations
import numpy as np

MIN_PAIRS = dict(similarity=2, affine=3, tps=10)


def fit_similarity(X, Y):
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my; U, S, Vt = np.linalg.svd(Yc.T @ Xc)
    D = np.diag([1, np.sign(np.linalg.det(U @ Vt))]); R = U @ D @ Vt; s = (S * np.diag(D)).sum() / (Xc ** 2).sum(); return s * R, my - s * R @ mx


def ransac(X, Y, tol=15.0, iters=3000, seed=0):
    rng = np.random.default_rng(seed); best = np.zeros(len(X), bool)
    for _ in range(iters if len(X) >= 3 else 0):
        i = rng.choice(len(X), 2, replace=False); A, b = fit_similarity(X[i], Y[i]); inl = np.linalg.norm(X @ A.T + b - Y, axis=1) <= tol
        if inl.sum() > best.sum(): best = inl
    return best


def fit_affine(X, Y):
    M, *_ = np.linalg.lstsq(np.c_[X, np.ones(len(X))], Y, rcond=None); return M[:2].T, M[2]          # [x y 1] @ M = [x' y']


class Transform:
    """kind in {similarity, affine, tps}; __call__ maps IV xy (um) -> EX xy (um); to_json() for transforms.json."""
    def __init__(self, kind, X, Y):
        X, Y = np.asarray(X, float).reshape(-1, 2), np.asarray(Y, float).reshape(-1, 2); self.kind, self.n = kind, len(X)
        assert self.n >= MIN_PAIRS[kind], f'{kind} needs >= {MIN_PAIRS[kind]} pairs, got {self.n}'
        if kind == 'tps':
            from scipy.interpolate import RBFInterpolator
            self.f = RBFInterpolator(X, Y, kernel='thin_plate_spline', smoothing=0.0); self.X, self.Y = X, Y
        else:
            self.A, self.b = (fit_similarity if kind == 'similarity' else fit_affine)(X, Y)

    def __call__(self, P):
        P = np.asarray(P, float).reshape(-1, 2)
        return self.f(P) if self.kind == 'tps' else P @ self.A.T + self.b

    def to_json(self):
        if self.kind == 'tps': return dict(kind='tps', n=self.n, src=self.X.tolist(), dst=self.Y.tolist())
        s = float(np.sqrt(abs(np.linalg.det(self.A)))); theta = float(np.degrees(np.arctan2(self.A[1, 0], self.A[0, 0])))
        return dict(kind=self.kind, n=self.n, matrix=self.A.tolist(), offset=self.b.tolist(), scale=s, rotation_deg=theta)


def residuals(T, X, Y): return np.linalg.norm(T(X) - np.asarray(Y, float), axis=1)


def loo_errors(kind, X, Y):
    """Error of each pair when it is left out of the fit (um); NaN where too few pairs remain."""
    X, Y = np.asarray(X, float).reshape(-1, 2), np.asarray(Y, float).reshape(-1, 2); out = np.full(len(X), np.nan)
    if len(X) - 1 < MIN_PAIRS[kind]: return out
    for i in range(len(X)):
        keep = np.arange(len(X)) != i; out[i] = residuals(Transform(kind, X[keep], Y[keep]), X[i:i + 1], Y[i:i + 1])[0]
    return out


def slicewise(iv_xyz, ex_xy, planes_z, kind='affine', dz=6.0):
    """One transform per IV plane from the pairs within |z - plane| <= dz um; falls back to the global fit when a plane has
    fewer than MIN_PAIRS[kind] pairs. Returns dict(global=..., planes=[...]) ready for transforms.json."""
    iv_xyz, ex_xy = np.asarray(iv_xyz, float).reshape(-1, 3), np.asarray(ex_xy, float).reshape(-1, 2)
    glob = Transform(kind, iv_xyz[:, :2], ex_xy) if len(iv_xyz) >= MIN_PAIRS[kind] else None
    out = dict(kind=kind, dz_um=dz, n_pairs=len(iv_xyz), planes=[],
               global_=None if glob is None else dict(glob.to_json(), rms_um=float(np.sqrt(np.mean(residuals(glob, iv_xyz[:, :2], ex_xy) ** 2))),
                                                    loo_median_um=float(np.nanmedian(loo_errors(kind, iv_xyz[:, :2], ex_xy))) if len(iv_xyz) > MIN_PAIRS[kind] else None))
    for z in planes_z:
        near = np.abs(iv_xyz[:, 2] - z) <= dz
        if near.sum() >= MIN_PAIRS[kind]:
            T = Transform(kind, iv_xyz[near, :2], ex_xy[near]); loo = loo_errors(kind, iv_xyz[near, :2], ex_xy[near])
            out['planes'].append(dict(z_um=float(z), source='plane', **T.to_json(), loo_median_um=float(np.nanmedian(loo)) if np.isfinite(loo).any() else None))
        else:
            out['planes'].append(dict(z_um=float(z), source='global' if glob is not None else 'none', n_near=int(near.sum())))
    return out
