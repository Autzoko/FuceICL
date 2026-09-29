#!/bin/bash
#SBATCH --job-name=demo-relative-test
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly CONFIG_PATH="${2:?confirmation config is required}"
readonly DATA_ROOT="${3:?belief chunk data root is required}"
readonly TRAINING_ROOT="${4:?training experiment root is required}"
readonly OUTPUT_PATH="${5:?new confirmation report path is required}"
readonly PYTHON_BIN="/scratch/ll5582/conda/envs/FuseICL-maniskill/bin/python"
export LD_LIBRARY_PATH="/scratch/ll5582/conda/envs/FuseICL-maniskill/lib:${LD_LIBRARY_PATH:-}"

if [[ -e "${OUTPUT_PATH}" ]]; then
    echo "输出已存在，拒绝覆盖：${OUTPUT_PATH}" >&2
    exit 2
fi
test -x "${PYTHON_BIN}"
test -s "${DATA_ROOT}/progress_chunks.npz"
test -s "${TRAINING_ROOT}/report.json"
test -s "${TRAINING_ROOT}/checkpoint.pt"
mkdir -p /scratch/ll5582/data/ManiSkill3/logs "$(dirname "${OUTPUT_PATH}")"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m dev.simulator.evaluate_demo_relative_action_regret_confirmation \
    --project-root "${PROJECT_ROOT}" \
    --config "${CONFIG_PATH}" \
    --data-root "${DATA_ROOT}" \
    --training-root "${TRAINING_ROOT}" \
    --output "${OUTPUT_PATH}"
