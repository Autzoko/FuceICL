#!/bin/bash
#SBATCH --job-name=demo-action-consensus
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/ll5582/data/RLBench/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/RLBench/logs/%x-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly CONTEXT_ROOT="${2:?PointNet context root is required}"
readonly ACTION_ROOT="${3:?action chunk root is required}"
readonly RETRIEVER_CHECKPOINT="${4:?retriever checkpoint is required}"
readonly TEXT_SCORES="${5:?text score artifact is required}"
readonly TRANSPORT_CHECKPOINT="${6:?transport checkpoint is required}"
readonly OUTPUT_PATH="${7:?output report path is required}"
readonly CONFIG_PATH="${8:-dev/predictor/config/jacobian_transport_pilot.json}"
readonly ENV_NAME="FuseICL-pointnet"

module purge
module load miniconda/3-4.11.0

mkdir -p /scratch/ll5582/data/RLBench/logs
cd "${PROJECT_ROOT}"

conda run --no-capture-output -n "${ENV_NAME}" \
    python -m dev.predictor.evaluate_demo_action_consensus \
    --project-root "${PROJECT_ROOT}" \
    --context-root "${CONTEXT_ROOT}" \
    --action-root "${ACTION_ROOT}" \
    --retriever-checkpoint "${RETRIEVER_CHECKPOINT}" \
    --text-scores "${TEXT_SCORES}" \
    --transport-checkpoint "${TRANSPORT_CHECKPOINT}" \
    --output "${OUTPUT_PATH}" \
    --config "${CONFIG_PATH}" \
    --device cpu
