#!/bin/bash
#SBATCH --job-name=demo-utility-selector
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ll5582/data/RLBench/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/RLBench/logs/%x-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly CONTEXT_ROOT="${2:?PointNet context root is required}"
readonly ACTION_ROOT="${3:?action chunk root is required}"
readonly RETRIEVER_CHECKPOINT="${4:?retriever checkpoint is required}"
readonly TRAIN_TEXT_SCORES="${5:?train text score artifact is required}"
readonly VAL_TEXT_SCORES="${6:?validation text score artifact is required}"
readonly TRANSPORT_CHECKPOINT="${7:?transport checkpoint is required}"
readonly OUTPUT_ROOT="${8:?output root is required}"
readonly CONFIG_PATH="${9:-dev/predictor/config/jacobian_transport_pilot.json}"
readonly ENV_NAME="FuseICL-pointnet"

module purge
module load miniconda/3-4.11.0

mkdir -p /scratch/ll5582/data/RLBench/logs
cd "${PROJECT_ROOT}"

conda run --no-capture-output -n "${ENV_NAME}" \
    python -m dev.predictor.train_demo_utility_selector \
    --project-root "${PROJECT_ROOT}" \
    --context-root "${CONTEXT_ROOT}" \
    --action-root "${ACTION_ROOT}" \
    --retriever-checkpoint "${RETRIEVER_CHECKPOINT}" \
    --train-text-scores "${TRAIN_TEXT_SCORES}" \
    --val-text-scores "${VAL_TEXT_SCORES}" \
    --transport-checkpoint "${TRANSPORT_CHECKPOINT}" \
    --output-root "${OUTPUT_ROOT}" \
    --config "${CONFIG_PATH}" \
    --device cpu
