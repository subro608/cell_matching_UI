// GET /api/example?name=jy369 -> the public example: its finished result (viewer bundle) and the input files, as signed URLs.
// Blob: examples/<name>/{info.json, web/..., raw/{iv,ex}.ome.tif, thumbs/{iv,ex}.jpg}   (tools/publish_example.py)
import { handler, HttpError, readJSON, signGet } from './_lib.js';

export default handler(async ({ query }) => {
  const name = String(query.name || 'jy369'); if (!/^[a-z0-9_-]{1,32}$/.test(name)) throw new HttpError(400, 'Bad example name');
  const info = await readJSON(`examples/${name}/info.json`); if (!info) throw new HttpError(404, 'No such example');
  const man = await readJSON(`examples/${name}/web/manifest.json`); const prefix = `examples/${name}/web/`;
  const extra = [...Object.values(info.data.downloads || {}).map((d) => d.path), ...Object.values(info.data.thumbs || {})].map((p) => `examples/${name}/${p}`);
  const urls = await signGet([...(man ? man.data.files.filter((f) => f !== 'manifest.json').map((f) => prefix + f) : []), ...extra], 12);
  return { name, info: info.data, manifest: man?.data || null, prefix, urls, base: `examples/${name}/` };
});
