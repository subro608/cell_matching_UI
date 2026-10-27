import { Matrix4 } from '@math.gl/core';

// Landmark fits in micrometres, the browser twin of core/ivexmatch/register.py (proper-rotation similarity, least-squares affine,
// thin-plate spline, leave-one-out). A linear transform is { kind, A: [[a, b], [c, d]], t: [tx, ty] } with y = A x + t; a TPS is
// { kind: 'tps', C: control points, W: n x 2 weights, P: 3 x 2 affine part, lin: the affine fit (for overlays and view linking) }.

export const MIN_PAIRS = { similarity: 2, affine: 3, tps: 10 };

const mean = (P) => P.reduce((m, p) => [m[0] + p[0] / P.length, m[1] + p[1] / P.length], [0, 0]);

function fitProperSimilarity(X, Y) {
  const mx = mean(X), my = mean(Y);
  let a = 0, b = 0, n = 0;
  X.forEach((x, i) => {
    const u = [x[0] - mx[0], x[1] - mx[1]], v = [Y[i][0] - my[0], Y[i][1] - my[1]];
    a += u[0] * v[0] + u[1] * v[1]; b += u[0] * v[1] - u[1] * v[0]; n += u[0] * u[0] + u[1] * u[1];
  });
  const th = Math.atan2(b, a), s = Math.hypot(a, b) / n, c = Math.cos(th) * s, d = Math.sin(th) * s;
  const A = [[c, -d], [d, c]];
  return { kind: 'similarity', A, t: [my[0] - (A[0][0] * mx[0] + A[0][1] * mx[1]), my[1] - (A[1][0] * mx[0] + A[1][1] * mx[1])] };
}

// Rotation + uniform scale + translation, and its mirrored twin (x -> -x first); the better fit wins, so a flipped IV still fits.
export function fitSimilarity(X, Y) {
  const P = fitProperSimilarity(X, Y), Mx = fitProperSimilarity(X.map((p) => [-p[0], p[1]]), Y);
  const M = { kind: 'similarity', A: [[-Mx.A[0][0], Mx.A[0][1]], [-Mx.A[1][0], Mx.A[1][1]]], t: Mx.t, mirrored: true };
  const sse = (T) => residuals(T, X, Y).reduce((s, r) => s + r * r, 0);
  return X.length >= 3 && sse(M) < sse(P) ? M : P;
}

export function fitAffine(X, Y) {
  // normal equations for [x y 1] M = [x' y'], 3x3 inverse
  const S = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], R = [[0, 0], [0, 0], [0, 0]];
  X.forEach((x, i) => {
    const r = [x[0], x[1], 1];
    for (let j = 0; j < 3; j++) { for (let k = 0; k < 3; k++) S[j][k] += r[j] * r[k]; R[j][0] += r[j] * Y[i][0]; R[j][1] += r[j] * Y[i][1]; }
  });
  const [[a, b, c], [d, e, f], [g, h, k]] = S;
  const det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g);
  if (Math.abs(det) < 1e-12) return null;
  const inv = [[e * k - f * h, c * h - b * k, b * f - c * e], [f * g - d * k, a * k - c * g, c * d - a * f], [d * h - e * g, b * g - a * h, a * e - b * d]]
    .map((row) => row.map((v) => v / det));
  const M = inv.map((row) => [0, 1].map((o) => row[0] * R[0][o] + row[1] * R[1][o] + row[2] * R[2][o])); // 3x2
  return { kind: 'affine', A: [[M[0][0], M[1][0]], [M[0][1], M[1][1]]], t: [M[2][0], M[2][1]] };
}

// Solve M x = b (n x n, b n x m) by Gaussian elimination with partial pivoting; null if singular.
function solve(M, b) {
  const n = M.length, m = b[0].length, A = M.map((row, i) => [...row, ...b[i]]);
  for (let c = 0; c < n; c++) {
    let piv = c; for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
    if (Math.abs(A[piv][c]) < 1e-12) return null;
    [A[c], A[piv]] = [A[piv], A[c]];
    for (let r = 0; r < n; r++) { if (r === c) continue; const f = A[r][c] / A[c][c]; if (f) for (let k = c; k < n + m; k++) A[r][k] -= f * A[c][k]; }
  }
  return A.map((row, i) => row.slice(n).map((v) => v / row[i]));
}

const U = (r2) => (r2 > 0 ? 0.5 * r2 * Math.log(r2) : 0);         // r^2 log r, from r^2

// Exact-interpolation thin-plate spline (scipy RBFInterpolator thin_plate_spline, smoothing 0), coordinates centred for conditioning.
export function fitTps(X, Y) {
  const n = X.length, mx = X.reduce((m, p) => [m[0] + p[0] / n, m[1] + p[1] / n], [0, 0]);
  const C = X.map((p) => [p[0] - mx[0], p[1] - mx[1]]);
  const L = Array.from({ length: n + 3 }, () => new Array(n + 3).fill(0)), B = Array.from({ length: n + 3 }, () => [0, 0]);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) L[i][j] = U((C[i][0] - C[j][0]) ** 2 + (C[i][1] - C[j][1]) ** 2);
    L[i][n] = L[n][i] = 1; L[i][n + 1] = L[n + 1][i] = C[i][0]; L[i][n + 2] = L[n + 2][i] = C[i][1]; B[i] = [Y[i][0], Y[i][1]];
  }
  const sol = solve(L, B); if (!sol) return null;
  return { kind: 'tps', C, mx, W: sol.slice(0, n), P: sol.slice(n), lin: fitAffine(X, Y) };
}

export const fit = (kind, X, Y) => (X.length < MIN_PAIRS[kind] ? null : kind === 'similarity' ? fitSimilarity(X, Y) : kind === 'tps' ? fitTps(X, Y) : fitAffine(X, Y));
export function apply(T, p) {
  if (T.kind !== 'tps') return [T.A[0][0] * p[0] + T.A[0][1] * p[1] + T.t[0], T.A[1][0] * p[0] + T.A[1][1] * p[1] + T.t[1]];
  const x = p[0] - T.mx[0], y = p[1] - T.mx[1]; let u = T.P[0][0] + T.P[1][0] * x + T.P[2][0] * y, v = T.P[0][1] + T.P[1][1] * x + T.P[2][1] * y;
  T.C.forEach((c, i) => { const k = U((x - c[0]) ** 2 + (y - c[1]) ** 2); u += T.W[i][0] * k; v += T.W[i][1] * k; });
  return [u, v];
}
export const linear = (T) => (T && T.kind === 'tps' ? T.lin : T);     // the affine part a view link / overlay matrix can use
export const scaleOf = (T) => Math.sqrt(Math.abs(T.A[0][0] * T.A[1][1] - T.A[0][1] * T.A[1][0]));
export const isMirrored = (T) => T.A[0][0] * T.A[1][1] - T.A[0][1] * T.A[1][0] < 0;
// rotation in degrees; for a mirrored map, the rotation applied after mirroring x
export const rotationOf = (T) => { const A = isMirrored(T) ? [[-T.A[0][0], T.A[0][1]], [-T.A[1][0], T.A[1][1]]] : T.A; return (Math.atan2(A[1][0], A[0][0]) * 180) / Math.PI; };

export function invert(T) {
  const [[a, b], [c, d]] = T.A, det = a * d - b * c;
  const A = [[d / det, -b / det], [-c / det, a / det]];
  return { kind: T.kind, A, t: [-(A[0][0] * T.t[0] + A[0][1] * T.t[1]), -(A[1][0] * T.t[0] + A[1][1] * T.t[1])] };
}

export function residuals(T, X, Y) { return X.map((x, i) => { const p = apply(T, x); return Math.hypot(p[0] - Y[i][0], p[1] - Y[i][1]); }); }

export function looErrors(kind, X, Y, max = 200) {
  if (X.length - 1 < MIN_PAIRS[kind] || (kind === 'tps' && X.length > max)) return X.map(() => NaN);
  return X.map((x, i) => {
    const T = fit(kind, X.filter((_, j) => j !== i), Y.filter((_, j) => j !== i));
    if (!T) return NaN;
    const p = apply(T, x); return Math.hypot(p[0] - Y[i][0], p[1] - Y[i][1]);
  });
}

export const median = (v) => { const s = v.filter(Number.isFinite).sort((a, b) => a - b); return s.length ? s[Math.floor((s.length - 1) / 2)] : NaN; };

// Native-pixel transform (IV px -> EX px) from a micrometre transform and the two pixel sizes.
export function toNative(T, ivUm, exUm) {
  return { kind: T.kind, A: T.A.map((row) => row.map((v) => (v * ivUm) / exUm)), t: T.t.map((v) => v / exUm) };
}

// deck.gl modelMatrix for a 2D affine: a math.gl Matrix4 (column-major); Viv layers call methods on it, a plain array fails
export const modelMatrix = (T) => new Matrix4([T.A[0][0], T.A[1][0], 0, 0, T.A[0][1], T.A[1][1], 0, 0, 0, 0, 1, 0, T.t[0], T.t[1], 0, 1]);

// Plane-wise fits for transforms.json: for every IV plane, the pairs within dz um (fallback: the global fit).
export function sliceFits(kind, pairs, nz, umZ, dz) {
  const all = pairs.map((p) => [p.iv, p.ex]); const glob = fit(kind, all.map((q) => q[0]), all.map((q) => q[1]));
  const summary = (T, P) => {
    if (!T) return null; const X = P.map((p) => p.iv), Y = P.map((p) => p.ex), res = residuals(T, X, Y);
    const out = { kind: T.kind, n: P.length, rms_um: Math.sqrt(res.reduce((s, r) => s + r * r, 0) / res.length), loo_median_um: median(looErrors(kind, X, Y)) };
    if (T.kind === 'tps') Object.assign(out, { src_um: X, dst_um: Y, affine_part: { A: T.lin?.A, t: T.lin?.t } }); else Object.assign(out, { A: T.A, t_um: T.t });
    return out;
  };
  const planes = [];
  for (let z = 0; z < nz; z++) {
    const near = pairs.filter((p) => Math.abs(p.z - z) * (umZ || 1) <= dz);
    const T = near.length >= MIN_PAIRS[kind] ? fit(kind, near.map((p) => p.iv), near.map((p) => p.ex)) : null;
    planes.push({ z, z_um: z * (umZ || 0), source: T ? 'plane' : glob ? 'global' : 'none', ...(T ? summary(T, near) : { n_near: near.length }) });
  }
  return { kind, dz_um: dz, units: 'IV xy (um) -> EX xy (um), origin at the top-left pixel centre', global: summary(glob, pairs), planes };
}
