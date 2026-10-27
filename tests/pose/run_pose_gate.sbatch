#!/bin/bash
#SBATCH --job-name=ivex_pose_gate
#SBATCH --account=torch_pr_60_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/sd5963/ivex-match/logs/%x_%j.out
# PMAC blind-pose gate (JY195 part); queued afterok on the PMAC parity job.
set -uo pipefail
REPO=/scratch/sd5963/ivex-match
SIF=/scratch/sd5963/containers/w2v2_cuda_cu128.sif
PY=/scratch/sd5963/venvs/marco_env/bin/python
nvidia-smi -L
singularity exec --nv $SIF bash -c "source /scratch/sd5963/cell_registration_2026/microscopy_ssl/experiments/nb_zeroshot_2026-10-08/env.sh && cd $REPO && $PY -u tests/pose/pose_gate_jy195.py"
