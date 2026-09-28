#!/bin/bash
#SBATCH --job-name=maniskill-ljat
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly DATA_ROOT="${2:?ManiSkill chunk root is required}"
readonly OUTPUT_ROOT="${3:?output root is required}"
readonly CONFIG_PATH="${4:-dev/simulator/config/maniskill_local_transport_v1.json}"
readonly ENV_NAME="FuseICL-maniskill"

module purge
module load miniconda/3-4.11.0

mkdir -p /scratch/ll5582/data/ManiSkill3/logs
cd "${PROJECT_ROOT}"

conda run --no-capture-output -n "${ENV_NAME}" \
    python -m dev.simulator.train_maniskill_local_transport \
    --project-root "${PROJECT_ROOT}" \
    --data-root "${DATA_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --config "${CONFIG_PATH}" \
    --device cuda
