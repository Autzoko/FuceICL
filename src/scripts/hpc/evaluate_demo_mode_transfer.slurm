#!/bin/bash
#SBATCH --job-name=demo-mode-oracle
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly REPLAY_ROOT="${2:?replay root is required}"
readonly CONFIG_PATH="${3:?Demo mode transfer config is required}"
readonly OUTPUT_PATH="${4:?new oracle report path is required}"
readonly PYTHON_BIN="/scratch/ll5582/conda/envs/FuseICL-maniskill/bin/python"
export LD_LIBRARY_PATH="/scratch/ll5582/conda/envs/FuseICL-maniskill/lib:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="/scratch/ll5582/.cache"
export MPLCONFIGDIR="/scratch/ll5582/.cache/matplotlib"

if [[ -e "${OUTPUT_PATH}" ]]; then
    echo "输出文件已存在，拒绝覆盖：${OUTPUT_PATH}" >&2
    exit 2
fi
test -x "${PYTHON_BIN}"
test -s "${REPLAY_ROOT}/replay_audit_report.json"
mkdir -p /scratch/ll5582/data/ManiSkill3/logs "$(dirname "${OUTPUT_PATH}")"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" \
    -m dev.simulator.evaluate_demo_mode_transfer \
    --project-root "${PROJECT_ROOT}" \
    --config "${CONFIG_PATH}" \
    --replay-root "${REPLAY_ROOT}" \
    --output "${OUTPUT_PATH}"
