#!/bin/bash
#SBATCH --job-name=belief-schedule-audit
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly CONFIG_PATH="${2:?belief schedule config is required}"
readonly REPLAY_ROOT="${3:?dual-view replay root is required}"
readonly OUTPUT_PATH="${4:?new audit report path is required}"
readonly PYTHON_BIN="/scratch/ll5582/conda/envs/FuseICL-maniskill/bin/python"
export LD_LIBRARY_PATH="/scratch/ll5582/conda/envs/FuseICL-maniskill/lib:${LD_LIBRARY_PATH:-}"

if [[ -e "${OUTPUT_PATH}" ]]; then
    echo "输出已存在，拒绝覆盖：${OUTPUT_PATH}" >&2
    exit 2
fi
test -x "${PYTHON_BIN}"
test -s "${REPLAY_ROOT}/replay_audit_report.json"
mkdir -p /scratch/ll5582/data/ManiSkill3/logs "$(dirname "${OUTPUT_PATH}")"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m dev.simulator.audit_belief_schedule \
    --project-root "${PROJECT_ROOT}" \
    --config "${CONFIG_PATH}" \
    --replay-root "${REPLAY_ROOT}" \
    --output "${OUTPUT_PATH}"
