#!/bin/bash
#SBATCH --job-name=rlbench-bcsg
#SBATCH --partition=visual
#SBATCH --gres=gpu:p4000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/ll5582/data/RLBench/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/RLBench/logs/%x-%j.err

source /etc/profile
set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly CONTEXT_ROOT="${2:?RLBench context root is required}"
readonly ACTION_ROOT="${3:?RLBench action root is required}"
readonly RETRIEVER_CHECKPOINT="${4:?PointNet checkpoint is required}"
readonly TEXT_SCORES="${5:?validation text scores are required}"
readonly PARENT_FOLD_ROOT="${6:?QA-LRDT v2 fold root is required}"
readonly OUTPUT_ROOT="${7:?new BCSG output root is required}"
readonly CONFIG_PATH="${8:?config path is required}"
readonly ENV_NAME="FuseICL-pointnet"

if [[ -e "${OUTPUT_ROOT}" ]]; then
    echo "输出目录已存在，拒绝覆盖：${OUTPUT_ROOT}" >&2
    exit 2
fi

module purge
module load miniconda/3-4.11.0
cd "${PROJECT_ROOT}"

conda run --no-capture-output -n "${ENV_NAME}" \
    python -m dev.predictor.train_benefit_calibrated_shrinkage \
    --project-root "${PROJECT_ROOT}" \
    --context-root "${CONTEXT_ROOT}" \
    --action-root "${ACTION_ROOT}" \
    --retriever-checkpoint "${RETRIEVER_CHECKPOINT}" \
    --text-scores "${TEXT_SCORES}" \
    --parent-fold-root "${PARENT_FOLD_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --config "${CONFIG_PATH}" \
    --device cuda
