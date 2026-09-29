#!/bin/bash
#SBATCH --job-name=demo-relative-preterminal
#SBATCH --partition=visual
#SBATCH --gres=gpu:p4000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly CONFIG_PATH="${2:?preterminal execution config is required}"
readonly BANK_ROOT="${3:?belief Demo bank is required}"
readonly QUERY_ROOT="${4:?fresh preterminal queries are required}"
readonly TRAINING_ROOT="${5:?training experiment root is required}"
readonly CONFIRMATION_PATH="${6:?internal-test confirmation is required}"
readonly REPLAY_ROOT="${7:?fresh replay root is required}"
readonly OUTPUT_PATH="${8:?new execution report path is required}"
readonly PYTHON_BIN="/scratch/ll5582/conda/envs/FuseICL-maniskill/bin/python"
readonly CACHE_ROOT="/scratch/ll5582/.cache/fuceicl-maniskill"
export LD_LIBRARY_PATH="/scratch/ll5582/conda/envs/FuseICL-maniskill/lib:\
${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${CACHE_ROOT}/matplotlib"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"

if [[ -e "${OUTPUT_PATH}" ]]; then
    echo "输出已存在，拒绝覆盖：${OUTPUT_PATH}" >&2
    exit 2
fi
test -x "${PYTHON_BIN}"
test -s "${QUERY_ROOT}/progress_chunks.npz"
test -s "${REPLAY_ROOT}/replay_audit_report.json"
test -s "${CONFIRMATION_PATH}"
mkdir -p \
    /scratch/ll5582/data/ManiSkill3/logs \
    "$(dirname "${OUTPUT_PATH}")" \
    "${CACHE_ROOT}/matplotlib" \
    "${CACHE_ROOT}/xdg"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m dev.simulator.evaluate_demo_relative_preterminal_execution \
    --project-root "${PROJECT_ROOT}" \
    --config "${CONFIG_PATH}" \
    --bank-root "${BANK_ROOT}" \
    --query-root "${QUERY_ROOT}" \
    --training-root "${TRAINING_ROOT}" \
    --confirmation "${CONFIRMATION_PATH}" \
    --replay-root "${REPLAY_ROOT}" \
    --output "${OUTPUT_PATH}"
