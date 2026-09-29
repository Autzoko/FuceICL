#!/bin/bash
#SBATCH --job-name=layout-anchor-data
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
readonly CONFIG_PATH="${3:?causal anchor data config is required}"
readonly OUTPUT_ROOT="${4:?new causal anchor data root is required}"
readonly PYTHON_BIN="/scratch/ll5582/conda/envs/FuseICL-maniskill/bin/python"
export LD_LIBRARY_PATH="/scratch/ll5582/conda/envs/FuseICL-maniskill/lib:${LD_LIBRARY_PATH:-}"

if [[ -e "${OUTPUT_ROOT}" ]]; then
    echo "输出目录已存在，拒绝覆盖：${OUTPUT_ROOT}" >&2
    exit 2
fi
test -x "${PYTHON_BIN}"
test -s "${REPLAY_ROOT}/replay_audit_report.json"
mkdir -p /scratch/ll5582/data/ManiSkill3/logs "$(dirname "${OUTPUT_ROOT}")"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" \
    -m dev.simulator.preprocess_rotated_layout_slide_causal_anchor_chunks \
    --project-root "${PROJECT_ROOT}" \
    --config "${CONFIG_PATH}" \
    --replay-root "${REPLAY_ROOT}" \
    --output-root "${OUTPUT_ROOT}"
