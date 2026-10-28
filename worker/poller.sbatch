#!/bin/bash
#SBATCH --job-name=ivex_poller
#SBATCH --account=torch_pr_60_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=05:55:00
#SBATCH --output=/scratch/sd5963/ivex-match/logs/poller/%x_%j.out
# Self-resubmitting HPC pull worker (scrontab is disabled here). Start once: sbatch worker/poller.sbatch. Stop: touch STOP_FILE, or
# scancel every ivex_poller job (the pending successor too). Exactly one poller runs: a second one exits at once.
set -uo pipefail
HERE=/scratch/sd5963/ivex-match/worker
STOP_FILE=/scratch/sd5963/ivex-sessions/_worker/STOP
mine=$SLURM_JOB_ID
others=$(squeue -h -u $USER -n ivex_poller -t RUNNING -o %i | grep -vx "$mine" || true)
if [ -n "$others" ]; then echo "another poller is running ($others); exiting"; exit 0; fi
[ -e $STOP_FILE ] && { echo "STOP file present; not starting"; exit 0; }
# successor first, so a crash below still hands over
sbatch --parsable --dependency=afterany:$mine $HERE/poller.sbatch > /dev/null && echo "successor submitted"
/scratch/sd5963/conda_envs/neuro_env_311/bin/python -u $HERE/poller.py --max-seconds $((5 * 3600 + 40 * 60))
