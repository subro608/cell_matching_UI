"""Cell masks for the three models the app shows. Run inside the matching Cellpose environment (they cannot share one):
    cyto2_ft, cyto3_ft  Cellpose 3.1 (venvs/cellpose_v3_env)   pyramidal_{iv,ex}_cyto{2,3}_v6public_ft (separate IV / EX fine-tunes)
    cpsam               Cellpose 4.2 (venvs/cellpose_v42_env)  dk5288 all_cpsam_clean_gt_best
IV: 3D with z anisotropy and fixed intensity limits [0, p99 of the stack] (as registration/segmentation/iv/full_cyto23_3d_ft, the
settings the fine-tunes were made for: 2D per plane finds ~1/8 of the cells on JY369); 2D per plane when the z step is unknown.
EX: 2D on the band MIP (as seg_ex.py). Masks stay on the native grid; centroids are converted to the ivex working grids later.

usage: python -m ivexmatch.segment --model cyto3_ft --side iv --input iv.npy --um 0.68 [--um_z 3] --diameter_um 11 --out masks_iv_cyto3_ft.npz
       input .npy: IV (Z, H, W) or EX (H, W) / (Z, H, W) (max-projected); any dtype.
-> npz: labels uint32 (Z, H, W) or (H, W); cells float32 (n, 5) = z, id, x, y, area (native px); meta json."""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np

MODEL_ROOT = Path(os.environ.get('IVEX_CELLPOSE_MODELS', '/scratch/sd5963/cell_registration_2026/cellinvariance/data/new_data_pyramidal/registration/training/cyto23_rebuild_20260912/models'))
CPSAM = Path(os.environ.get('IVEX_CPSAM_MODEL', '/scratch/dk5288/finetuned_cellpose/models_all/all_cpsam_clean_gt/models/all_cpsam_clean_gt_best'))
MODELS = dict(cyto2_ft=dict(version=3, path=lambda side: MODEL_ROOT / f'pyramidal_{side}_cyto2_v6public_ft'),
              cyto3_ft=dict(version=3, path=lambda side: MODEL_ROOT / f'pyramidal_{side}_cyto3_v6public_ft'),
              cpsam=dict(version=4, path=lambda side: CPSAM))
CELLPROB = dict(iv=0.0, ex=-1.0)                    # seg_iv.py / seg_ex.py


def centroids(lab, z=0):
    from scipy import ndimage as ndi
    ids = np.unique(lab); ids = ids[ids > 0]
    if not len(ids): return np.zeros((0, 5), np.float32)
    yx = np.asarray(ndi.center_of_mass(lab > 0, lab, ids), np.float32).reshape(-1, 2); area = ndi.sum(np.ones_like(lab, np.uint8), lab, ids)
    return np.c_[np.full(len(ids), z), ids, yx[:, 1], yx[:, 0], area].astype(np.float32)


def run(model_name, side, image, um, diameter_um, gpu=True, um_z=None, log=print):
    import cellpose
    from cellpose import models
    spec = MODELS[model_name]; major = int(cellpose.version.split('.')[0]) if hasattr(cellpose, 'version') else None
    path = spec['path'](side); assert path.exists(), path
    model = models.CellposeModel(gpu=gpu, pretrained_model=str(path)); diam = float(diameter_um) / float(um)
    kw = dict(diameter=diam, cellprob_threshold=CELLPROB[side])
    if spec['version'] == 3: kw['channels'] = [0, 0]
    t0 = time.time(); three_d = side == 'iv' and image.ndim == 3 and len(image) > 1 and um_z
    if side == 'iv':
        sample = np.asarray(image[:: max(1, len(image) // 8)] if image.ndim == 3 else image, np.float32)
        kw['normalize'] = dict(lowhigh=[0.0, float(np.percentile(sample, 99))])
    if three_d:
        kw.update(do_3D=True, z_axis=0, anisotropy=float(um_z) / float(um), min_size=15)
        if spec['version'] == 3: kw['channel_axis'] = None
        labels = model.eval(np.asarray(image, np.float32), **kw)[0].astype(np.uint32)
        cells = np.concatenate([centroids(labels[z], z) for z in range(len(labels))])
    else:
        planes = [image] if image.ndim == 2 else list(image); labels, cells = [], []
        for z, p in enumerate(planes):
            lab = model.eval(np.asarray(p, np.float32), **kw)[0].astype(np.uint32); labels.append(lab); cells.append(centroids(lab, z))
        labels = labels[0] if image.ndim == 2 else np.stack(labels); cells = np.concatenate(cells)
    meta = dict(model=model_name, side=side, weights=str(path), cellpose_version=getattr(cellpose, 'version', None), cellpose_major=major,
                diameter_px=diam, diameter_um=float(diameter_um), um=float(um), um_z=um_z, mode='3d' if three_d else '2d', cellprob=CELLPROB[side], n_cells=int(len(cells)), seconds=round(time.time() - t0, 1))
    log(f'{model_name} {side}: {len(cells)} cells, {meta["seconds"]} s'); return labels, cells, meta


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', choices=list(MODELS), required=True); ap.add_argument('--side', choices=['iv', 'ex'], required=True)
    ap.add_argument('--input', required=True); ap.add_argument('--um', type=float, required=True); ap.add_argument('--um_z', type=float); ap.add_argument('--diameter_um', type=float, required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--cpu', action='store_true'); a = ap.parse_args()
    image = np.load(a.input, mmap_mode='r')
    if a.side == 'ex' and image.ndim == 3: image = np.asarray(image).max(0)
    labels, cells, meta = run(a.model, a.side, np.asarray(image), a.um, a.diameter_um, gpu=not a.cpu, um_z=a.um_z)
    np.savez_compressed(a.out, labels=labels, cells=cells, meta=json.dumps(meta))


if __name__ == '__main__':
    main()
