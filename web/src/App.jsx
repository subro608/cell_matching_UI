// Routes by query string (works on any static host): ?data=URL (local bundle folder) · ?s=SESSION · ?page=new · ?page=about · home.
import { useEffect, useState } from 'react';
import Viewer from './viewer/Viewer';
import { folderSource, loadBundle } from './lib/data';
import Home from './pages/Home';
import Upload from './pages/Upload';
import SessionPage from './pages/SessionPage';
import About from './pages/About';

const params = new URLSearchParams(window.location.search);

function LocalViewer({ data }) {
  // absolute URL: Viv's loadOmeTiff needs one (relative paths fail in new URL())
  const base = new URL(data.replace(/\/?$/, '/'), window.location.href).href;
  const [b, setB] = useState(null), [err, setErr] = useState(null);
  useEffect(() => { loadBundle(folderSource(base)).then(setB).catch((e) => setErr(String(e))); }, [base]);
  if (err) return <div className="splash error">Could not load bundle from <code>{base}</code>: {err}</div>;
  if (!b) return <div className="splash">Loading bundle from <code>{base}</code>…</div>;
  return <Viewer b={b} />;
}

export default function App() {
  if (params.get('data')) return <LocalViewer data={params.get('data')} />;
  if (params.get('s')) return <SessionPage sid={params.get('s')} />;
  if (params.get('page') === 'new') return <Upload />;
  if (params.get('page') === 'about') return <About />;
  return <Home />;
}
