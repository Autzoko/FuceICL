#!/bin/bash
#SBATCH --job-name=maniskill-bidir-pair
#SBATCH --partition=visual
#SBATCH --gres=gpu:p4000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.err

source /etc/profile
set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly CONFIG_PATH="${2:?pair pilot config is required}"
readonly OUTPUT_PATH="${3:?new output report is required}"
readonly ENV_NAME="FuseICL-maniskill"
readonly CACHE_ROOT="/scratch/ll5582/.cache/fuceicl-maniskill"

if [[ -e "${OUTPUT_PATH}" ]]; then
    echo "输出文件已存在，拒绝覆盖：${OUTPUT_PATH}" >&2
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
    python -m dev.simulator.audit_bidirectional_slide_pairs \
    --project-root "${PROJECT_ROOT}" \
    --config "${CONFIG_PATH}" \
    --output "${OUTPUT_PATH}"
