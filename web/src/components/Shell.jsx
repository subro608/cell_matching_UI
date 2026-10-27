// Page chrome for the non-viewer pages: sticky top bar (brand, nav, worker status) and footer.
import { useEffect, useState } from 'react';
import { status } from '../lib/api';

export function Brand() {
  return (
    <a className="brand" href="./">
      <span className="mark"><svg width="16" height="16" viewBox="0 0 32 32"><circle cx="11" cy="16" r="6" fill="#04110e" /><circle cx="21" cy="16" r="6" fill="none" stroke="#04110e" strokeWidth="3" /></svg></span>
      ivex <small>IV ↔ EX cell matching</small>
    </a>
  );
}

export function WorkerPill({ worker }) {
  if (!worker) return <span className="pill"><span className="dot" />checking worker…</span>;
  return worker.online
    ? <span className="pill ok" title="The NYU HPC worker checks for jobs every minute"><span className="dot" />worker online{worker.pending ? ` · ${worker.pending} waiting` : ''}</span>
    : <span className="pill off" title="Jobs wait until the worker is back"><span className="dot" />worker offline</span>;
}

export function useWorker() {
  const [w, setW] = useState(null);
  useEffect(() => { status().then((s) => setW(s.worker)).catch(() => setW({ online: false })); }, []);
  return w;
}

export default function Shell({ active, children, worker }) {
  const w = useWorker(), shown = worker || w;
  const link = (key, href, label) => <a className={active === key ? 'on' : ''} href={href}>{label}</a>;
  return (
    <div className="shell">
      <header className="topnav">
        <Brand />
        <nav>{link('home', './', 'Overview')}{link('new', '?page=new', 'New analysis')}{link('example', '?page=example', 'Example')}{link('about', '?page=about', 'How it works')}</nav>
        <span className="spacer" />
        <WorkerPill worker={shown} />
        <a className="btn primary" href="?page=new">New analysis</a>
      </header>
      <div style={{ flex: 1 }}>{children}</div>
      <footer className="footer"><span>ivex · pilot</span><span>GPU analysis on NYU HPC</span><a href="?page=about">Privacy and retention</a><a href="https://github.com/subro608/cell_matching_UI">Source</a></footer>
    </div>
  );
}
