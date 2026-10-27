#!/usr/bin/env python3
"""Vendor the PMAC model code into core/ivexmatch/pmac/vendor/ (reproducible; rerun after the source changes).

The model files are copied verbatim from the GFP+DAPI e200 experiment, whose code is a superset of the B100 (grayscale) code:
the only differences are the 3-channel EX branches, and `nearby` in arms.json only affects training. Patches are limited to
  * absolute imports -> package-relative imports, so the vendored copy can sit next to the original training modules;
  * UNICOM import from the vendored vision_transformer instead of sys.path;
  * stock UNICOM weights optional (a PMAC checkpoint overwrites every tensor; env IVEX_UNICOM_STOCK points to the file).
Each patch must match an exact number of times. data.py is NOT copied: vendor/data.py is a hand-written inference-only subset.
Writes vendor/SOURCES.json with source paths and sha256 of every copied file."""
import hashlib, json, re, shutil
from pathlib import Path

EXP = Path('/scratch/sd5963/cell_registration_2026/microscopy_ssl/experiments')
E200 = EXP / 'pmac_gfp_dapi_e200_2026-10-25'
B100 = EXP / 'pmac_synth2_B_train100_loso_2026-10-22'
UNICOM = EXP / 'marco_onestage_B_unfrozen_2026-10-08/vendor_unicom'
OUT = Path(__file__).resolve().parents[1] / 'core/ivexmatch/pmac/vendor'

REL = [(r'^from (data|model_unicom|backbones|base_model|legacy_joint|factorial_model|resolution_base) import', r'from .\1 import'),
       (r'^import (data)$', r'from . import \1'),
       (r'^import resolution_base as base$', 'from . import resolution_base as base')]
PATCHES = {
    'model_unicom.py': [
        ('import sys\n', 'import os, sys\n', 1),
        ("STOCK_CKPT = Path('/scratch/vsp7230/cell_matching/neurobridge/.cache/unicom/FP16-ViT-B-16.pt')",
         "STOCK_CKPT = Path(os.environ.get('IVEX_UNICOM_STOCK', '/scratch/vsp7230/cell_matching/neurobridge/.cache/unicom/FP16-ViT-B-16.pt'))", 1),
        ("VENDOR = BU / 'vendor_unicom/unicom'", 'VENDOR = Path(__file__).resolve().parent  # ivex: vendored unicom_vit.py', 1),
        ('        sys.path.insert(0, str(VENDOR))\n        from unicom.vision_transformer import load_model_and_transform',
         '        from .unicom_vit import load_model_and_transform', 1)],
    'backbones.py': [
        ("   sys.path.insert(0,str(VENDOR))\n   from unicom.vision_transformer import load_model_and_transform\n"
         "   self.net,_=load_model_and_transform('ViT-B/16')\n"
         "   weights=torch.load(STOCK_CKPT,map_location='cpu',weights_only=True)\n"
         "   self.net.load_state_dict(weights,strict=True)\n"
         "   assert all(torch.equal(v,weights[k].to(v.dtype)) for k,v in self.net.state_dict().items())\n",
         "   from .unicom_vit import load_model_and_transform\n"
         "   self.net,_=load_model_and_transform('ViT-B/16')\n"
         "   if STOCK_CKPT.exists():  # ivex: optional; a PMAC checkpoint overwrites every tensor (load_state_dict strict=True)\n"
         "    weights=torch.load(STOCK_CKPT,map_location='cpu',weights_only=True)\n"
         "    self.net.load_state_dict(weights,strict=True)\n"
         "    assert all(torch.equal(v,weights[k].to(v.dtype)) for k,v in self.net.state_dict().items())\n", 1)],
}
FILES = ['model.py', 'resolution_base.py', 'base_model.py', 'factorial_model.py', 'backbones.py', 'model_unicom.py', 'legacy_joint.py']


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True); sources = {}
    for name in FILES:
        text = (E200 / name).read_text()
        for pat, rep in REL:
            text = re.sub(pat, rep, text, flags=re.M)
        for old, new, count in PATCHES.get(name, []):
            assert text.count(old) == count, (name, old[:60], text.count(old))
            text = text.replace(old, new)
        assert not re.search(r'^(from|import) (data|model_unicom|backbones|base_model|legacy_joint|factorial_model|resolution_base)\b', text, re.M), name
        (OUT / name).write_text(text); sources[name] = dict(source=str(E200 / name), sha256=sha(E200 / name))
    vit = UNICOM / 'unicom/unicom/vision_transformer.py'
    shutil.copyfile(vit, OUT / 'unicom_vit.py'); sources['unicom_vit.py'] = dict(source=str(vit), sha256=sha(vit))
    shutil.copyfile(UNICOM / 'LICENSE', OUT / 'LICENSE_UNICOM'); sources['LICENSE_UNICOM'] = dict(source=str(UNICOM / 'LICENSE'), license='MIT')
    arms = json.loads((B100 / 'arms.json').read_text()); e200 = json.loads((E200 / 'arms.json').read_text())
    merged = {k: arms[k] for k in ('base_overview_real_synth', 'base_overview_real2')}
    merged.update({k: e200[k] for k in ('gfpdapi_synth', 'gfponly_synth', 'gfpdapi_real', 'gfponly_real')})
    (OUT / 'arms.json').write_text(json.dumps(merged, indent=1))
    sources['arms.json'] = dict(source=[str(B100 / 'arms.json'), str(E200 / 'arms.json')], sha256=[sha(B100 / 'arms.json'), sha(E200 / 'arms.json')])
    (OUT / 'SOURCES.json').write_text(json.dumps(sources, indent=1))
    print('vendored', len(sources), 'files ->', OUT)


if __name__ == '__main__':
    main()
