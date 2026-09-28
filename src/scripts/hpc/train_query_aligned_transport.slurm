#!/bin/bash
#SBATCH --job-name=rlbench-qa-lrdt
#SBATCH --partition=visual
#SBATCH --gres=gpu:p4000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ll5582/data/RLBench/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/RLBench/logs/%x-%j.err

source /etc/profile
set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly CONTEXT_ROOT="${2:?RLBench context root is required}"
readonly ACTION_ROOT="${3:?RLBench action root is required}"
readonly RETRIEVER_CHECKPOINT="${4:?PointNet checkpoint is required}"
readonly TEXT_SCORES="${5:?validation text scores are required}"
readonly OUTPUT_ROOT="${6:?new output root is required}"
readonly CONFIG_PATH="${7:?config path is required}"
readonly FOLD_NAME="${8:?fold name is required}"
shift 8
readonly ENV_NAME="FuseICL-pointnet"

if [[ "$#" -lt 1 ]]; then
    echo "至少需要一个 held-out task" >&2
    exit 2
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
    echo "输出目录已存在，拒绝覆盖：${OUTPUT_ROOT}" >&2
    exit 2
fi

module purge
module load miniconda/3-4.11.0

mkdir -p /scratch/ll5582/data/RLBench/logs
cd "${PROJECT_ROOT}"

held_out_arguments=()
for task in "$@"; do
    if [[ ! "${task}" =~ ^[a-z0-9_]+$ ]]; then
        echo "held-out task 含非法字符：${task}" >&2
        exit 2
    fi
    held_out_arguments+=(--held-out-task "${task}")
done

conda run --no-capture-output -n "${ENV_NAME}" \
    python -m dev.predictor.train_query_aligned_transport \
    --project-root "${PROJECT_ROOT}" \
    --context-root "${CONTEXT_ROOT}" \
    --action-root "${ACTION_ROOT}" \
    --retriever-checkpoint "${RETRIEVER_CHECKPOINT}" \
    --text-scores "${TEXT_SCORES}" \
    --output-root "${OUTPUT_ROOT}" \
    --config "${CONFIG_PATH}" \
    --fold-name "${FOLD_NAME}" \
    --device cuda \
    "${held_out_arguments[@]}"
