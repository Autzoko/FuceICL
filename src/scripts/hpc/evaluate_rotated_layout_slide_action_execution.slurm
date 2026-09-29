#!/bin/bash
#SBATCH --job-name=layout-h6-execution
#SBATCH --partition=visual
#SBATCH --gres=gpu:p4000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly CONFIG_PATH="${2:?execution config is required}"
readonly BANK_ROOT="${3:?training Demo bank root is required}"
readonly QUERY_ROOT="${4:?new predictor query root is required}"
readonly TRAINING_ROOT="${5:?frozen policy training root is required}"
readonly REPLAY_ROOT="${6:?new replay root is required}"
readonly OUTPUT_PATH="${7:?new report path is required}"
readonly PYTHON_BIN="/scratch/ll5582/conda/envs/FuseICL-maniskill/bin/python"
readonly CACHE_ROOT="/scratch/ll5582/.cache/fuceicl-maniskill"
export LD_LIBRARY_PATH="/scratch/ll5582/conda/envs/FuseICL-maniskill/lib:${LD_LIBRARY_PATH:-}"

if [[ -e "${OUTPUT_PATH}" ]]; then
    echo "输出已存在，拒绝覆盖：${OUTPUT_PATH}" >&2
    exit 2
fi
test -x "${PYTHON_BIN}"
test -s "${BANK_ROOT}/branch_samples.npz"
test -s "${QUERY_ROOT}/branch_samples.npz"
test -s "${TRAINING_ROOT}/checkpoint.pt"
test -s "${REPLAY_ROOT}/replay_audit_report.json"
mkdir -p \
    /scratch/ll5582/data/ManiSkill3/logs \
    "$(dirname "${OUTPUT_PATH}")" \
    "${CACHE_ROOT}/matplotlib" \
    "${CACHE_ROOT}/xdg"
export MPLCONFIGDIR="${CACHE_ROOT}/matplotlib"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" \
    -m dev.simulator.evaluate_rotated_layout_slide_action_execution \
    --project-root "${PROJECT_ROOT}" \
    --config "${CONFIG_PATH}" \
    --bank-root "${BANK_ROOT}" \
    --query-root "${QUERY_ROOT}" \
    --training-root "${TRAINING_ROOT}" \
    --replay-root "${REPLAY_ROOT}" \
    --output "${OUTPUT_PATH}"
