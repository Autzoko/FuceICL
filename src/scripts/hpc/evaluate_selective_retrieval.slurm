#!/bin/bash
#SBATCH --job-name=retriever-reliability
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/ll5582/data/RLBench/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/RLBench/logs/%x-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly DATA_ROOT="${2:?PointNet dataset root is required}"
readonly CHECKPOINT="${3:?Retriever checkpoint is required}"
readonly TEXT_SCORES="${4:?text score artifact is required}"
readonly OUTPUT_PATH="${5:?output report path is required}"
readonly ENV_NAME="FuseICL-pointnet"

module purge
module load miniconda/3-4.11.0

mkdir -p /scratch/ll5582/data/RLBench/logs
cd "${PROJECT_ROOT}"

conda run --no-capture-output -n "${ENV_NAME}" \
    python -m dev.pointnet.evaluate_selective_retrieval \
    --project-root "${PROJECT_ROOT}" \
    --data-root "${DATA_ROOT}" \
    --checkpoint "${CHECKPOINT}" \
    --text-scores "${TEXT_SCORES}" \
    --output "${OUTPUT_PATH}" \
    --device cpu
