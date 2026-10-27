"""Session viewer click-through (plan > Verification > Viewer) on a finished session served through the API (signed Blob URLs):
pose card -> N selects a cell -> crop cards -> A on 12 cells -> similarity / affine / TPS fits with leave-one-out -> overlay
blend / swipe / checkerboard -> agreement mode -> exports (CSV, transforms.json, JSON) -> "save now" -> a fresh browser (no
localStorage) reloads the session and gets the same verdicts back from the server. Fails on any page error.
usage: python e2e_session.py BASE_URL SID SCREENSHOT_DIR"""
import json, re, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE, SID, OUT = sys.argv[1].rstrip('/'), sys.argv[2], Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
SHELL = '/scratch/sd5963/.cache/ms-playwright/chromium_headless_shell-1223/chrome-headless-shell-linux64/chrome-headless-shell'
ARGS = ['--no-sandbox', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']
errors, report = [], {}


def page(br):
    pg = br.new_page(viewport=dict(width=1600, height=950), accept_downloads=True)
    pg.on('pageerror', lambda e: errors.append(f'pageerror: {e}'))
    pg.on('console', lambda m: errors.append(f'console.error: {m.text[:200]}') if m.type == 'error' else None)
    pg.goto(f'{BASE}/?s={SID}'); pg.wait_for_selector('canvas', timeout=90000); time.sleep(6); return pg


def shot(pg, name): pg.screenshot(path=str(OUT / f'{name}.png'), timeout=120000)


def download(pg, label):
    with pg.expect_download() as d: pg.get_by_text(label, exact=True).click()
    return Path(d.value.path()).read_text()


with sync_playwright() as p:
    br = p.chromium.launch(executable_path=SHELL, args=ARGS); pg = page(br)
    accepted = lambda: int(re.search(r'accepted\D*(\d+)', pg.locator('.counts').inner_text()).group(1))
    report['pose'] = pg.locator('.posecard .pose-head').inner_text(); n0 = accepted()
    pg.keyboard.press('n'); pg.wait_for_selector('table.cands tbody tr', timeout=30000); pg.wait_for_selector('.crops canvas', timeout=30000)
    report['crop_cards'] = pg.locator('.crops canvas').count(); shot(pg, 'v1_selected')
    for _ in range(12): pg.keyboard.press('a'); time.sleep(0.4)
    report['accepted_before'], report['accepted_after'] = n0, accepted()
    for kind in ('similarity', 'affine', 'tps'):
        pg.locator('.panel.right section').filter(has_text='Registration').locator('select').nth(0).select_option(kind); time.sleep(0.8)
        report[f'reg_{kind}'] = pg.locator('.reg').inner_text().split('\n')[:2] if pg.locator('.reg').count() else None
    pg.locator('.panel.right section').filter(has_text='Registration').locator('select').nth(1).select_option('global'); time.sleep(0.5)
    report['reg_tps_global'] = pg.locator('.reg').inner_text().split('\n')[:2] if pg.locator('.reg').count() else None
    pg.keyboard.press('o'); time.sleep(3); shot(pg, 'v2_overlay_blend')
    for style in ('swipe', 'checkerboard'):
        pg.get_by_role('button', name=style, exact=True).click(); time.sleep(2.5); shot(pg, f'v3_overlay_{style}')
    pg.keyboard.press('o'); time.sleep(1)
    pg.locator('.panel.left details.sec').filter(has_text='Masks').locator('select').nth(0).select_option('2'); time.sleep(3)
    report['agreement_counts'] = [t for t in pg.locator('.panel.left details.sec').filter(has_text='Masks').locator('.row .hint').all_inner_texts()]; shot(pg, 'v4_agreement')
    csv = download(pg, 'Pairs CSV'); tr = json.loads(download(pg, 'transforms.json')); ex = json.loads(download(pg, 'Export JSON'))
    report['csv_rows'] = len(csv.strip().split('\n')) - 1; report['transforms_planes'] = len(tr['planes']); report['transforms_global_n'] = (tr.get('global') or {}).get('n')
    report['export_pairs'] = len(ex['pairs'])
    btn = pg.get_by_role('button', name='Save', exact=True)
    if btn.count(): btn.click()
    pg.wait_for_function("() => /saved \\d/.test(document.querySelector('.vbar').innerText)", timeout=30000)
    report['saved'] = [l for l in pg.locator('.vbar').inner_text().split('\n') if l.startswith('saved')][:1]
    br.close()
    br = p.chromium.launch(executable_path=SHELL, args=ARGS); pg = page(br)                        # fresh profile: server copy only
    report['accepted_reloaded'] = accepted(); br.close()
report['errors'] = errors[:10]; print(json.dumps(report, indent=1))
n = report['accepted_after']                     # idempotent: a session may already hold verdicts from an earlier run
ok = (report['crop_cards'] >= 3 and n > report['accepted_before'] and n >= 10 and all(report[f'reg_{k}'] for k in ('similarity', 'affine', 'tps'))
      and report['csv_rows'] == n and report['export_pairs'] == n and report['transforms_planes'] > 0 and report['accepted_reloaded'] == n and not errors)
print('SESSION E2E PASS' if ok else 'SESSION E2E FAIL'); sys.exit(0 if ok else 1)
