#!/bin/bash
#SBATCH --job-name=action-chunk-predictor
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/ll5582/data/RLBench/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/RLBench/logs/%x-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly CONTEXT_ROOT="${2:?PointNet context root is required}"
readonly ACTION_ROOT="${3:?action chunk root is required}"
readonly RETRIEVER_CHECKPOINT="${4:?retriever checkpoint is required}"
readonly TEXT_SCORES="${5:?text score artifact is required}"
readonly OUTPUT_ROOT="${6:?output root is required}"
readonly CONFIG_PATH="${7:-dev/predictor/config/action_chunk_train_pilot.json}"
readonly ENV_NAME="FuseICL-pointnet"

module purge
module load miniconda/3-4.11.0

mkdir -p /scratch/ll5582/data/RLBench/logs
cd "${PROJECT_ROOT}"

conda run --no-capture-output -n "${ENV_NAME}" \
    python -m dev.predictor.train_action_chunks \
    --project-root "${PROJECT_ROOT}" \
    --context-root "${CONTEXT_ROOT}" \
    --action-root "${ACTION_ROOT}" \
    --retriever-checkpoint "${RETRIEVER_CHECKPOINT}" \
    --text-scores "${TEXT_SCORES}" \
    --output-root "${OUTPUT_ROOT}" \
    --config "${CONFIG_PATH}" \
    --device cuda
