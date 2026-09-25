#!/bin/bash
#SBATCH --job-name=rlbench-pointnet-preprocess
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/ll5582/data/RLBench/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/RLBench/logs/%x-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly OUTPUT_ROOT="${2:?output root is required}"
readonly CONFIG_PATH="${3:-dev/pointnet/config/rlbench_pilot.json}"
readonly DATA_ROOT="/scratch/ll5582/data/RLBench/raw/data"
readonly ENV_NAME="FuseICL-pointnet"

module purge
module load miniconda/3-4.11.0

mkdir -p /scratch/ll5582/data/RLBench/logs
cd "${PROJECT_ROOT}"

conda run --no-capture-output -n "${ENV_NAME}" \
    python -m dev.pointnet.preprocess_rlbench \
    --data-root "${DATA_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --config "${CONFIG_PATH}"

conda run --no-capture-output -n "${ENV_NAME}" \
    python -m dev.pointnet.inspect_dataset "${OUTPUT_ROOT}"

date -u +%Y-%m-%dT%H:%M:%SZ > "${OUTPUT_ROOT}/PREPROCESS_COMPLETE"
