// The public JY195 example: its finished result in the viewer (verdicts stay in this browser), with the input files to download.
import { useEffect, useState } from 'react';
import Viewer from '../viewer/Viewer';
import { getExample } from '../lib/api';
import { loadBundle, signedSource } from '../lib/data';

export default function ExamplePage() {
  const [b, setB] = useState(null), [ex, setEx] = useState(null), [err, setErr] = useState(null);
  useEffect(() => {
    getExample().then(async (e) => {
      if (!e.manifest) throw new Error('the example result is not published yet');
      setEx(e); setB(await loadBundle(signedSource(e.urls, e.prefix), { ...e.manifest, example: true }));
    }).catch((e) => setErr(String(e.message || e)));
  }, []);
  if (err) return <div className="splash error">Could not open the example: {err}</div>;
  if (!b) return <div className="splash"><div><div className="spinner" />Loading the JY195 example…</div></div>;
  const dl = Object.entries(ex.info.downloads || {}).map(([k, d]) => ({ label: `${k.toUpperCase()} (${d.mb} MB)`, href: ex.urls[ex.base + d.path] }));
  return <Viewer b={b} downloads={dl} />;
}
