"""Upload of a file the browser cannot read, through the server probe (plan > universal inputs): IV TIFF (read in the browser) + EX zip
of per-plane TIFFs (read on the server) -> worker probe (dimensions, channels, previews) -> describe form (pixel size entered) ->
analysis queued. Uses public CellGPR data. usage: python e2e_probe.py BASE IV.tif EX.zip INVITE OUT [--iv-um 0.83 --ex-um 0.65]"""
import argparse, json, re, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser(); [ap.add_argument(k) for k in ('base', 'iv', 'ex', 'invite', 'out')]; ap.add_argument('--iv-um', default='0.83'); ap.add_argument('--ex-um', default='0.65')
a = ap.parse_args(); OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True); errors = []
SHELL = '/scratch/sd5963/.cache/ms-playwright/chromium_headless_shell-1223/chrome-headless-shell-linux64/chrome-headless-shell'
with sync_playwright() as p:
    br = p.chromium.launch(executable_path=SHELL, args=['--no-sandbox']); pg = br.new_page(viewport=dict(width=1300, height=1100)); pg.set_default_timeout(60000)
    pg.on('pageerror', lambda e: errors.append(f'pageerror: {e}')); pg.on('console', lambda m: errors.append(m.text[:200]) if m.type == 'error' else None)
    pg.goto(a.base.rstrip('/') + '/?page=new'); pg.wait_for_selector('input[type=file]')
    files = pg.locator('input[type=file]'); files.nth(0).set_input_files(a.iv); files.nth(1).set_input_files(a.ex); time.sleep(3)
    drops = pg.locator('.drop').all_inner_texts(); print('drop zones:', [d.replace('\n', ' ')[:140] for d in drops])
    pg.locator('.card').nth(1).locator('input[type=number]').nth(0).fill(a.iv_um)
    pg.locator('.card').nth(3).locator('input[type=text]').nth(0).fill('probe e2e (CellGPR, public)'); pg.locator('.card').nth(3).locator('input[type=text]').nth(1).fill(a.invite)
    pg.get_by_text('Upload and read the files').click(); pg.wait_for_url(re.compile(r'\?s='), timeout=300000); sid = re.search(r'\?s=([\w-]+)', pg.url).group(1); print('session', sid)
    t0 = time.time()
    while time.time() - t0 < 900:                                            # the page polls every 15 s; wait for the worker's probe
        if pg.get_by_role('button', name='Start the analysis').count(): break
        time.sleep(10); pg.reload(); time.sleep(4)
    print('describe form after', round(time.time() - t0), 's')
    cards = pg.locator('.card'); print('probe cards:', [c.split('\n')[0] + ' | ' + c.split('\n')[1][:120] for c in cards.all_inner_texts()[:2]])
    pg.screenshot(path=str(OUT / 'probe_form.png'), full_page=True)
    ex_card = pg.locator('.card').nth(1); ex_card.locator('input[type=number]').nth(0).fill(a.ex_um); time.sleep(0.5)
    pg.get_by_role('button', name='Start the analysis').click(); time.sleep(8); pg.reload(); time.sleep(5)
    steps = pg.locator('.steps').inner_text() if pg.locator('.steps').count() else pg.inner_text('body')[:300]
    print('after start:', steps.replace('\n', ' | ')[:300]); br.close()
import urllib.request                                                       # Blob's CDN can lag a minute behind overwrites: poll the API
state = {}
for _ in range(24):
    state = json.load(urllib.request.urlopen(f"{a.base.rstrip('/')}/api/session?sid={sid}"))['state']
    if state.get('kind') == 'analyze' or state.get('status') == 'running' and state.get('kind') != 'probe': break
    time.sleep(10)
print(json.dumps(dict(sid=sid, state=state.get('status'), kind=state.get('kind'), errors=errors[:8])))
ok = state.get('status') in ('queued', 'running', 'done') and state.get('kind') != 'probe'
print('PROBE E2E PASS' if ok and not [e for e in errors if 'favicon' not in e] else 'PROBE E2E FAIL')
