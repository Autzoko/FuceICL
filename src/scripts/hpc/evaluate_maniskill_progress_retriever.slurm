#!/bin/bash
#SBATCH --job-name=maniskill-progress
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.err

source /etc/profile
set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly DATA_ROOT="${2:?ManiSkill chunk root is required}"
readonly OUTPUT_PATH="${3:?output report path is required}"
readonly CONFIG_PATH="${4:-dev/simulator/config/maniskill_progress_retriever_v1.json}"
readonly ENV_NAME="FuseICL-maniskill"

module purge
module load miniconda/3-4.11.0

mkdir -p /scratch/ll5582/data/ManiSkill3/logs
cd "${PROJECT_ROOT}"

conda run --no-capture-output -n "${ENV_NAME}" \
    python -m dev.simulator.evaluate_maniskill_progress_retriever \
    --project-root "${PROJECT_ROOT}" \
    --data-root "${DATA_ROOT}" \
    --config "${CONFIG_PATH}" \
    --output "${OUTPUT_PATH}"
