# ivex-match (working name)

Web tool for in vivo (2-photon) ↔ ex vivo (confocal) cell matching. A user uploads an IV stack and an EX image or stack and gets:
- cell masks from three Cellpose models (fine-tuned cyto2, fine-tuned cyto3, CP-SAM), with an agreement view;
- PMAC's blind pose (mirror / rotation / scale with a shuffle-null p-value), and PMAC's matches at that pose;
- a verification page (keys A / R / U, crop cards, filters);
- slice-wise registration from the verified pairs (similarity / affine / thin-plate spline, leave-one-out, blend / swipe / checkerboard);
- exports: pairs CSV, per-plane `transforms.json`, verification JSON.

Plan: `/scratch/sd5963/.claude/plans/ok-the-main-idea-zany-quill.md`.

## Architecture ($0; NYU HPC makes outbound connections only)
```
browser ── Vercel (static Vite app + web/api/* Node functions) ── Vercel Blob (private store "ivex-match-data")
               │ presigned PUT / signed GET URLs: big files never pass through a function     ▲
               ▼                                                                            │ outbound HTTPS
         queue.json, sessions/<sid>/{request,state,verify}.json, raw/*.tif, web/*            │
NYU HPC: worker/poller.sbatch (1 CPU, cpu_short, resubmits itself) ─ sbatch ─▶ worker/analyze.sbatch (GPU) ─▶ ivex analyze / rematch
         scratch /scratch/sd5963/ivex-sessions/<sid>/ is the system of record; Blob holds uploads briefly and results for 7 days
```
Blob layout and the free-tier budget are documented at the top of `web/api/_lib.js`. In short, Hobby allows 2,000 uploads/lists,
10,000 cache-miss reads, 1 GB stored and 10 GB downloaded a month, and blocks the store for 30 days beyond that. So there are no
`list()` calls, reads go through the CDN cache, sessions are a dozen large files, and results expire after 7 days (`restore` re-uploads).

## Parts
| Part | Files | State (10-27) |
|---|---|---|
| PMAC model (vendored) + inference | `core/ivexmatch/pmac/` | parity PASS: GFP+DAPI 206/324; JY195 B1 293/439, bit-identical |
| Blind pose (PMAC appearance descriptors) | `core/ivexmatch/pose.py` | JY195 gate 6/8 (both failures flagged p ≥ 0.05); synthetic pair (mirror, 40°, 0.95) recovered exactly |
| Segmentation (Cellpose 3 / 4) | `core/ivexmatch/segment.py` | IV 3D with fixed limits (the fine-tunes' setting); parity vs the reference masks: see below |
| Analyze / rematch / export CLI | `core/ivexmatch/cli.py`, `bundle.py`, `webexport.py` | end-to-end on GPU: JY369 crop 229 s; synthetic pair through the cloud |
| Registration | `core/ivexmatch/register.py`, `web/src/lib/fit.js` | parity PASS (2e-13); browser TPS = scipy to 1.5e-11 µm |
| Worker | `worker/poller.py`, `poller.sbatch`, `analyze.sbatch`, `run_job.py`, `blob.mjs` | running; claim → GPU → upload → done verified |
| API | `web/api/*.js` | tested locally (`web/dev_api_server.mjs`) against the real store; not yet deployed |
| Web app | `web/src/` (React 19, Viv 0.23, deck.gl 9.4) | upload, status, viewer; e2e tests pass |

## Run
```bash
# analysis by hand (inside the container)
SIF=/scratch/sd5963/containers/w2v2_cuda_cu128.sif
singularity exec --nv $SIF bash -c "cd /scratch/sd5963/ivex-match && PYTHONPATH=core /scratch/sd5963/venvs/marco_env/bin/python -m ivexmatch.cli \
  analyze --iv iv.tif --iv-um 0.684 --iv-z-um 3 --ex ex.tif --ex-um 0.324 --ex-dapi-channel 1 --out S/analysis --web S/web"
# a session directory as the worker runs it (S/job.json + S/raw/{iv,ex}.tif)
sbatch worker/analyze.sbatch /scratch/sd5963/ivex-sessions/<sid>

# the worker: start once; it resubmits itself every ~5 h 40 min. Stop: touch /scratch/sd5963/ivex-sessions/_worker/STOP and scancel the ivex_poller jobs
sbatch worker/poller.sbatch

# viewer on a local bundle folder
cd web && npm run build && python serve.py --data /scratch/sd5963/ivex-match-data/demo_jy369   # http://127.0.0.1:8765/?data=/data/
# full app locally (API against the real Blob store)
BLOB_READ_WRITE_TOKEN=$(cat /scratch/sd5963/.secrets/vercel_blob_ivex_rw) INVITE_CODES=local-dev-test node web/dev_api_server.mjs 8790
```
Node is `/scratch/sd5963/conda_envs/neuro_env_311/bin/node`; npm uses `web/.npmrc` (`bin-links=false`, no symlinks).

## Tests
- PMAC parity: `sbatch tests/parity/run_parity.sbatch` (JY195 B1 293/439, jy306va 206/324).
- Blind pose: `sbatch tests/pose/run_pose_gate.sbatch` (criteria pre-declared in `tests/pose/pose_gate_jy195.py`).
- Registration: `python tests/test_register.py`.
- Browser (host `python3` 3.13 with `PYTHONUSERBASE=/scratch/sd5963/python_user`, `LD_LIBRARY_PATH=/scratch/sd5963/conda_envs/headless_libs/lib`):
  - `tests/web/e2e_demo.py URL OUT`: local bundle click-through;
  - `tests/web/e2e_upload.py BASE IV EX INVITE OUT`: upload flow (pick → previews → presigned upload → submit);
  - `tests/web/e2e_session.py BASE SID OUT`: session viewer (pose card, crops, 12 accepts, three fits, overlays, agreement, exports, server save, reload).
- Test data: `tools/make_synthetic_pair.py` (no real data; safe for the cloud); `tools/make_test_crop_jy369.py` (unpublished, local only).

## Deploy (not done yet: needs the user's OK)
1. Create the Vercel project from `web/` (Vite; `api/` becomes functions) and set env vars `BLOB_READ_WRITE_TOKEN` (the store's token) and `INVITE_CODES`.
2. Preview first; production only with the user's OK (`/scratch/sd5963/.claude/skills/vercel-deploy`).
3. The CVPR 2027 double-blind question (the name PMAC on a public page) is open.
