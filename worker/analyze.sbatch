#!/bin/bash
#SBATCH --job-name=ivex_job
#SBATCH --account=torch_pr_62_tandon_priority
#SBATCH --partition=a100_tandon,l40s_public,h100_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:50:00
#SBATCH --output=/scratch/sd5963/ivex-match/logs/jobs/%x_%j.out
# One ivex session job (analyze or rematch) on a GPU: sbatch analyze.sbatch SESSION_DIR. Under 2 h so it avoids the idle-GPU kill.
# Three partitions: Slurm starts it on whichever frees first (the tandon A100 group cap is often full; PMAC needs < 7 GiB).
set -uo pipefail
SESSION=$1
SIF=/scratch/sd5963/containers/w2v2_cuda_cu128.sif
PY=/scratch/sd5963/venvs/marco_env/bin/python
nvidia-smi -L
singularity exec --nv $SIF bash -c "source /scratch/sd5963/cell_registration_2026/microscopy_ssl/experiments/nb_zeroshot_2026-10-08/env.sh && cd /scratch/sd5963/ivex-match && $PY -u worker/run_job.py $SESSION"
status=$?
echo "IVEX_JOB_EXIT=$status"
exit $status
