// Mask agreement between models on one plane: a cell "agrees" with another model when that model has a cell whose centroid lies
// within r px. Returns { [setName]: Set(ids with >= k models, itself included) }.
export function agreement(outlinesBySet, r, k) {
  const names = Object.keys(outlinesBySet), cell = r;
  const grids = Object.fromEntries(names.map((n) => {
    const g = new Map();
    for (const d of outlinesBySet[n]) { const key = `${Math.floor(d.cx / cell)},${Math.floor(d.cy / cell)}`; (g.get(key) || g.set(key, []).get(key)).push(d); }
    return [n, g];
  }));
  const near = (n, d) => {
    const gx = Math.floor(d.cx / cell), gy = Math.floor(d.cy / cell), g = grids[n];
    for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) for (const e of g.get(`${gx + i},${gy + j}`) || []) if (Math.hypot(e.cx - d.cx, e.cy - d.cy) <= r) return true;
    return false;
  };
  return Object.fromEntries(names.map((n) => [n, new Set(outlinesBySet[n].filter((d) => 1 + names.filter((m) => m !== n && near(m, d)).length >= k).map((d) => d.id))]));
}
