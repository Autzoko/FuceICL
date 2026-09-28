#!/bin/bash
#SBATCH --job-name=retriever-e2e
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ll5582/data/RLBench/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/RLBench/logs/%x-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly DATA_ROOT="${2:?processed data root is required}"
readonly CHECKPOINT="${3:?checkpoint path is required}"
readonly TEXT_SCORES="${4:?text score artifact is required}"
readonly OUTPUT="${5:?report path is required}"
readonly ENV_NAME="FuseICL-pointnet"

module purge
module load miniconda/3-4.11.0

mkdir -p "$(dirname "${OUTPUT}")" /scratch/ll5582/data/RLBench/logs
cd "${PROJECT_ROOT}"

conda run --no-capture-output -n "${ENV_NAME}" \
    python -m dev.pointnet.evaluate_end_to_end_retriever \
    --data-root "${DATA_ROOT}" \
    --checkpoint "${CHECKPOINT}" \
    --text-scores "${TEXT_SCORES}" \
    --output "${OUTPUT}" \
    --device cuda
