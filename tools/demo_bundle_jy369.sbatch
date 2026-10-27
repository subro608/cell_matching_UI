#!/bin/bash
#SBATCH --job-name=ivex_demo_bundle
#SBATCH --account=torch_pr_62_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/sd5963/ivex-match/logs/%x_%j.out
# CPU only: builds the local JY369 demo web bundle (unpublished data, stays on scratch).
set -euo pipefail
singularity exec /scratch/sd5963/containers/w2v2_cuda_cu128.sif /scratch/sd5963/venvs/marco_env/bin/python -u /scratch/sd5963/ivex-match/tools/demo_bundle_jy369.py
