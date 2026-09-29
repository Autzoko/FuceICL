#!/bin/bash
#SBATCH --job-name=maniskill-bidir-audit
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly SOURCE_ROOT="${2:?paired source dataset root is required}"
readonly REPLAY_ROOT="${3:?paired replay dataset root is required}"
readonly CONFIG_PATH="${4:?replay audit config is required}"
readonly OUTPUT_PATH="${5:?new audit report path is required}"
readonly PYTHON_BIN="/scratch/ll5582/conda/envs/FuseICL-maniskill/bin/python"

if [[ -e "${OUTPUT_PATH}" ]]; then
    echo "输出已存在，拒绝覆盖：${OUTPUT_PATH}" >&2
    exit 2
fi
test -x "${PYTHON_BIN}"
mkdir -p /scratch/ll5582/data/ManiSkill3/logs

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m dev.simulator.audit_bidirectional_slide_replay \
    --project-root "${PROJECT_ROOT}" \
    --config "${CONFIG_PATH}" \
    --source-root "${SOURCE_ROOT}" \
    --replay-root "${REPLAY_ROOT}" \
    --output "${OUTPUT_PATH}"
