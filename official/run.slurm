#!/bin/bash
#SBATCH --job-name=icml26-exp
#SBATCH --partition=seas_compute,tambe,serial_requeue
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --array=0-8
#SBATCH --cpus-per-task=64

ALL_GAMMA=(0.5 0.7 0.9)
ALL_N=(5 10 15)

NUM_N=${#ALL_N[@]}
IDX=${SLURM_ARRAY_TASK_ID}
i_n=$((IDX % NUM_N))
i_gamma=$((IDX / NUM_N))
GAMMA=${ALL_GAMMA[$i_gamma]}
N=${ALL_N[$i_n]}

mkdir -p logs results
source .venv/bin/activate
python3 run_experiments.py 30 200 "${GAMMA}" "${N}"
