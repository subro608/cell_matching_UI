"""Universal reader on real files of each dataset family (local; the data never leave scratch). Prints one line per file:
format, (Z, C, Y, X), dtype, pixel sizes + their source, channel names, positions, warnings, seconds.
usage: venvs/ivex_io_env/bin/python tests/test_readers.py [OUT_DIR]"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core'))
from ivexmatch import readers

R = Path('/scratch/sd5963/cell_registration_2026'); P = R / 'cellinvariance/data/new_data_pyramidal'; L = R / 'external_data/li2404_brainsmatics/extracted/data1/data1'
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else '/tmp/ivex_readers'); OUT.mkdir(parents=True, exist_ok=True)
CASES = [('Attila jy306 EX tile (ND2)', R / 'jy306_data/registration_video/row1/1.nd2'),
         ('JY369 IV raw tile (zip of Bruker per-plane OME-TIFF + XML)', P / 'IV/1_1.zip'),
         ('JY369 IV stitched (zip of tifffile per-plane OME-TIFF)', P / 'IV/stitched.zip'),
         ('JY369 IV one per-plane OME file on its own', next((P / 'IV/FOV_processed').glob('*.ome.tif'), None)),
         ('Attila jy306 IV (plain multipage TIFF)', R / 'jy306_data/registration_video/JY306_in_Vivo_stack_s80.tif'),
         ('Attila JY195 EX export (TIFF)', R / 'data_ex3d_JY195/JY195_ex_487_Region1.tif'),
         ('CellGPR EX fMOST (big TIFF)', L / 'fMOST.tif'),
         ('CellGPR IV 2P site (2D ImageJ TIFF)', sorted((L / '2p').glob('*.tif'))[0]),
         ('MARCO derived stack (NPZ)', next((R / 'MARCO/data/cellreg_stack').rglob('*_ivstack.npz'), None)),
         ('MARCO derived EX (PNG)', next((R / 'MARCO/data/cellreg_stack').rglob('*_ex.png'), None)),
         ('JY239 LUT pickle (must be refused)', R / 'cellinvariance/data/attila_lab/JY239/LUT_multimodal_Region1.pkl')]
for label, f in CASES:
    if f is None or not Path(f).exists(): print(f'SKIP {label}: not found'); continue
    t = time.time()
    try:
        arr, m = readers.read(f); prev = readers.previews(arr, OUT / Path(f).stem)
        print(f"OK   {label}: {m['format']} {tuple(arr.shape)} {m['dtype']} um_xy={m['um_xy']} um_z={m['um_z']} ({m['um_source']}) ch={m['channel_names']} "
              f"positions={len(m['positions'])} warn={m['warnings']} [{time.time() - t:.0f}s]", flush=True)
        del arr
    except readers.Unsupported as e:
        print(f'REFUSED {label}: {e} [{time.time() - t:.0f}s]', flush=True)
