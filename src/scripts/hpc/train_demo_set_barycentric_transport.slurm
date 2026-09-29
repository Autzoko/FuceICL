#!/bin/bash
#SBATCH --job-name=maniskill-dsbt
#SBATCH --partition=visual
#SBATCH --gres=gpu:p4000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.err

source /etc/profile
set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly SOURCE_CHECKPOINT="${2:?source checkpoint is required}"
readonly RADM_CHECKPOINT="${3:?RADM checkpoint is required}"
readonly CONFIG_PATH="${4:?DSBT config is required}"
readonly OUTPUT_ROOT="${5:?new output root is required}"
shift 5
readonly ENV_NAME="FuseICL-maniskill"
readonly CACHE_ROOT="/scratch/ll5582/.cache/fuceicl-maniskill"

if [[ "$#" -ne 3 ]]; then
    echo "DSBT 固定要求三个 ManiSkill data roots" >&2
    exit 2
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
    echo "输出目录已存在，拒绝覆盖：${OUTPUT_ROOT}" >&2
    exit 2
fi

module purge
module load miniconda/3-4.11.0
mkdir -p \
    /scratch/ll5582/data/ManiSkill3/logs \
    "$(dirname "${OUTPUT_ROOT}")" \
    "${CACHE_ROOT}/matplotlib" \
    "${CACHE_ROOT}/xdg" \
    "${CACHE_ROOT}/torch" \
    "${CACHE_ROOT}/huggingface"

export MPLCONFIGDIR="${CACHE_ROOT}/matplotlib"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TORCH_HOME="${CACHE_ROOT}/torch"
export HF_HOME="${CACHE_ROOT}/huggingface"

cd "${PROJECT_ROOT}"
data_arguments=()
for root in "$@"; do
    data_arguments+=(--data-root "${root}")
done

conda run --no-capture-output -n "${ENV_NAME}" \
    python -m dev.simulator.train_demo_set_barycentric_transport \
    --project-root "${PROJECT_ROOT}" \
    --source-checkpoint "${SOURCE_CHECKPOINT}" \
    --radm-checkpoint "${RADM_CHECKPOINT}" \
    --config "${CONFIG_PATH}" \
    --output-root "${OUTPUT_ROOT}" \
    --device cuda \
    "${data_arguments[@]}"
