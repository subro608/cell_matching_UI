"""Upload flow in a browser against a running API (local dev_api_server.mjs or a Vercel preview): pick IV / EX TIFFs -> header
read + previews -> create session -> presigned uploads to Blob -> submit -> session page shows the job queued.
usage: python e2e_upload.py BASE_URL IV.tif EX.tif INVITE SCREENSHOT_DIR [--iv-um 1 --iv-z 3 --ex-um 0.6 --dapi 1]  -> prints the session id"""
import argparse, json, re, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser(); ap.add_argument('base'); ap.add_argument('iv'); ap.add_argument('ex'); ap.add_argument('invite'); ap.add_argument('out')
ap.add_argument('--iv-um', default='1'); ap.add_argument('--iv-z', default='3'); ap.add_argument('--ex-um', default='0.6'); ap.add_argument('--dapi', default='1')
a = ap.parse_args(); OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
SHELL = '/scratch/sd5963/.cache/ms-playwright/chromium_headless_shell-1223/chrome-headless-shell-linux64/chrome-headless-shell'
errors = []
with sync_playwright() as p:
    br = p.chromium.launch(executable_path=SHELL, args=['--no-sandbox']); pg = br.new_page(viewport=dict(width=1300, height=1100))
    pg.on('pageerror', lambda e: errors.append(f'pageerror: {e}'))
    pg.on('console', lambda m: errors.append(f'console.error: {m.text}') if m.type == 'error' else None)
    pg.goto(a.base.rstrip('/') + '/?page=new'); pg.wait_for_selector('input[type=file]')
    files = pg.locator('input[type=file]'); files.nth(0).set_input_files(a.iv); files.nth(1).set_input_files(a.ex)
    pg.wait_for_selector('.preview canvas', timeout=30000); time.sleep(2)
    nums = lambda card: pg.locator('.card').nth(card).locator('input[type=number]')
    nums(0).nth(0).fill(a.iv_um); nums(0).nth(1).fill(a.iv_z); nums(1).nth(0).fill(a.ex_um)
    pg.locator('.card').nth(1).locator('select').nth(1).select_option(a.dapi)
    pg.locator('.card').nth(2).locator('input[type=text]').nth(0).fill('synthetic e2e test'); pg.locator('.card').nth(2).locator('input[type=text]').nth(1).fill(a.invite)
    dims = [t for t in pg.locator('.field .hint').all_inner_texts() if 'plane' in t]
    pg.screenshot(path=str(OUT / 'upload_1_form.png'), full_page=True)
    pg.get_by_text('Upload and analyze').click()
    pg.wait_for_url(re.compile(r'\?s='), timeout=300000); time.sleep(3); sid = re.search(r'\?s=([\w-]+)', pg.url).group(1)
    pg.wait_for_selector('.steps', timeout=30000); steps = pg.locator('.steps').inner_text()
    pg.screenshot(path=str(OUT / 'upload_2_status.png'), full_page=True); br.close()
print(json.dumps(dict(sid=sid, dims=dims, steps=steps.split('\n'), errors=errors[:10]), indent=1))
ok = bool(sid) and 'Waiting for the HPC worker' in steps and not [e for e in errors if 'favicon' not in e]
print('UPLOAD E2E PASS' if ok else 'UPLOAD E2E FAIL'); sys.exit(0 if ok else 1)
