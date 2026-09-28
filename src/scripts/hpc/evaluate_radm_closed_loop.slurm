#!/bin/bash
#SBATCH --job-name=maniskill-radm-cl
#SBATCH --partition=visual
#SBATCH --gres=gpu:p4000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.err

source /etc/profile
set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly TASK_ID="${2:?ManiSkill task id is required}"
readonly DATA_ROOT="${3:?ManiSkill chunk root is required}"
readonly SOURCE_METADATA="${4:?pointcloud replay metadata is required}"
readonly SOURCE_CHECKPOINT="${5:?QA-LRDT/BCSG checkpoint is required}"
readonly RADM_CHECKPOINT="${6:?RADM checkpoint is required}"
readonly OUTPUT_PATH="${7:?new output report path is required}"
readonly CONFIG_PATH="${8:?closed-loop config is required}"
readonly ENV_NAME="FuseICL-maniskill"
readonly CACHE_ROOT="/scratch/ll5582/.cache/fuceicl-maniskill"

if [[ -e "${OUTPUT_PATH}" ]]; then
    echo "输出已存在，拒绝覆盖：${OUTPUT_PATH}" >&2
    exit 2
fi

module purge
module load miniconda/3-4.11.0
mkdir -p \
    /scratch/ll5582/data/ManiSkill3/logs \
    "$(dirname "${OUTPUT_PATH}")" \
    "${CACHE_ROOT}/matplotlib" \
    "${CACHE_ROOT}/xdg" \
    "${CACHE_ROOT}/torch" \
    "${CACHE_ROOT}/huggingface"

export MPLCONFIGDIR="${CACHE_ROOT}/matplotlib"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TORCH_HOME="${CACHE_ROOT}/torch"
export HF_HOME="${CACHE_ROOT}/huggingface"

cd "${PROJECT_ROOT}"
conda run --no-capture-output -n "${ENV_NAME}" \
    python -m dev.simulator.evaluate_query_aligned_closed_loop \
    --project-root "${PROJECT_ROOT}" \
    --task-id "${TASK_ID}" \
    --data-root "${DATA_ROOT}" \
    --source-metadata "${SOURCE_METADATA}" \
    --checkpoint "${SOURCE_CHECKPOINT}" \
    --radm-checkpoint "${RADM_CHECKPOINT}" \
    --config "${CONFIG_PATH}" \
    --output "${OUTPUT_PATH}" \
    --device cuda
