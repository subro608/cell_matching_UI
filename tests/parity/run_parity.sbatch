#!/bin/bash
#SBATCH --job-name=ivex_parity
#SBATCH --account=torch_pr_62_tandon_priority
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/sd5963/ivex-match/logs/%x_%j.out
# PMAC parity gates for the ivex core package (A100, same GPU family as the reference runs).
set -uo pipefail
REPO=/scratch/sd5963/ivex-match
SIF=/scratch/sd5963/containers/w2v2_cuda_cu128.sif
PY=/scratch/sd5963/venvs/marco_env/bin/python
nvidia-smi -L
run() { singularity exec --nv $SIF bash -c "source /scratch/sd5963/cell_registration_2026/microscopy_ssl/experiments/nb_zeroshot_2026-10-08/env.sh && cd $REPO && $PY -u $1"; }
TESTS=${@:-tests/parity/pmac_b1_jy195.py tests/parity/pmac_jy306va_gfpdapi.py}   # sbatch run_parity.sbatch [test.py ...]
status=0
for t in $TESTS; do run $t; s=$?; echo "PARITY $t exit=$s"; status=$(( status || s )); done
exit $status
