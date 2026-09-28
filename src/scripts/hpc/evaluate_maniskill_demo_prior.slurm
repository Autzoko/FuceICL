#!/bin/bash
#SBATCH --job-name=maniskill-demo-prior
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.err

source /etc/profile
set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly DATA_ROOT="${2:?ManiSkill chunk root is required}"
readonly OUTPUT_PATH="${3:?output report path is required}"
readonly ENV_NAME="FuseICL-maniskill"

module purge
module load miniconda/3-4.11.0

mkdir -p /scratch/ll5582/data/ManiSkill3/logs
cd "${PROJECT_ROOT}"

conda run --no-capture-output -n "${ENV_NAME}" \
    python -m dev.simulator.evaluate_maniskill_demo_prior \
    --project-root "${PROJECT_ROOT}" \
    --data-root "${DATA_ROOT}" \
    --output "${OUTPUT_PATH}" \
    --k 4 \
    --bootstrap-resamples 5000
