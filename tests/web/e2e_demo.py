"""Viewer click-through on a served bundle (plan > Verification > Viewer): load -> N selects a cell -> candidates listed ->
A accepts 6 PMAC top-1 pairs -> registration fit appears (RMS, leave-one-out) -> overlay mode -> Export JSON round-trips.
Fails on any page error. usage: python e2e_demo.py URL SCREENSHOT_DIR (host python + python_user playwright + headless_libs)."""
import json, sys
from pathlib import Path
from playwright.sync_api import sync_playwright

URL, OUT = sys.argv[1], Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
SHELL = '/scratch/sd5963/.cache/ms-playwright/chromium_headless_shell-1223/chrome-headless-shell-linux64/chrome-headless-shell'
errors = []
with sync_playwright() as p:
    br = p.chromium.launch(executable_path=SHELL, args=['--no-sandbox', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
    pg = br.new_page(viewport=dict(width=1600, height=900), accept_downloads=True); pg.set_default_timeout(120000)   # software WebGL is slow
    pg.on('pageerror', lambda e: errors.append(f'pageerror: {e}'))
    pg.on('console', lambda m: errors.append(f'console.error: {m.text}') if m.type == 'error' else None)
    pg.goto(URL); pg.wait_for_selector('canvas', timeout=60000); pg.wait_for_timeout(4000)
    pg.keyboard.press('n'); pg.wait_for_selector('table.cands tbody tr', timeout=20000)
    rows = pg.locator('table.cands tbody tr').count(); head = pg.locator('.panel.right h2').nth(1).inner_text()
    pg.wait_for_timeout(1500); pg.screenshot(path=str(OUT / 'e2e_1_selected.png'))
    for _ in range(6): pg.keyboard.press('a'); pg.wait_for_timeout(400)
    counts = pg.locator('.counts').inner_text(); reg = pg.locator('.reg').inner_text() if pg.locator('.reg').count() else ''
    pg.screenshot(path=str(OUT / 'e2e_2_accepted.png'))
    pg.keyboard.press('o'); pg.wait_for_timeout(3000); pg.screenshot(path=str(OUT / 'e2e_3_overlay.png'))
    with pg.expect_download() as d: pg.get_by_text('Export JSON').click()
    exported = json.loads(Path(d.value.path()).read_text())
    br.close()
report = dict(candidate_rows=rows, selected=head, counts=counts, registration=reg, exported_pairs=len(exported['pairs']),
              exported_registration=bool(exported.get('registration')), errors=errors[:10])
print(json.dumps(report, indent=1))
ok = rows == 10 and 'accepted 6' in counts and 'RMS' in reg and len(exported['pairs']) == 6 and not errors
print('E2E PASS' if ok else 'E2E FAIL'); sys.exit(0 if ok else 1)
