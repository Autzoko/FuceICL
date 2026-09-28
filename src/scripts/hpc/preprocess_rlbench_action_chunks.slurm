#!/bin/bash
#SBATCH --job-name=rlbench-action-chunks
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ll5582/data/RLBench/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/RLBench/logs/%x-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly PARENT_ROOT="${2:?parent PointNet dataset root is required}"
readonly OUTPUT_ROOT="${3:?output root is required}"
readonly CONFIG_PATH="${4:-dev/predictor/config/action_chunk_pilot.json}"
readonly RAW_DATA_ROOT="/scratch/ll5582/data/RLBench/raw/data"
readonly ENV_NAME="FuseICL-pointnet"

module purge
module load miniconda/3-4.11.0

mkdir -p /scratch/ll5582/data/RLBench/logs
cd "${PROJECT_ROOT}"

conda run --no-capture-output -n "${ENV_NAME}" \
    python -m dev.predictor.preprocess_action_chunks \
    --raw-data-root "${RAW_DATA_ROOT}" \
    --parent-root "${PARENT_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --config "${CONFIG_PATH}"

date -u +%Y-%m-%dT%H:%M:%SZ > "${OUTPUT_ROOT}/PREPROCESS_COMPLETE"
