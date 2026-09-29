#!/bin/bash
#SBATCH --job-name=layout-policy-confirm
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
readonly BANK_ROOT="${3:?training Demo bank root is required}"
readonly CONFIRMATION_ROOT="${4:?confirmation predictor data root is required}"
readonly TRAINING_ROOT="${5:?frozen policy training root is required}"
readonly OUTPUT_PATH="${6:?new confirmation report path is required}"
readonly PYTHON_BIN="/scratch/ll5582/conda/envs/FuseICL-maniskill/bin/python"

if [[ -e "${OUTPUT_PATH}" ]]; then
    echo "输出已存在，拒绝覆盖：${OUTPUT_PATH}" >&2
    exit 2
fi
test -x "${PYTHON_BIN}"
test -s "${BANK_ROOT}/branch_samples.npz"
test -s "${CONFIRMATION_ROOT}/branch_samples.npz"
test -s "${TRAINING_ROOT}/checkpoint.pt"
test -s "${TRAINING_ROOT}/report.json"
mkdir -p /scratch/ll5582/data/ManiSkill3/logs "$(dirname "${OUTPUT_PATH}")"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" \
    -m dev.simulator.evaluate_rotated_layout_slide_demo_policy_confirmation \
    --project-root "${PROJECT_ROOT}" \
    --config "${CONFIG_PATH}" \
    --bank-root "${BANK_ROOT}" \
    --confirmation-root "${CONFIRMATION_ROOT}" \
    --training-root "${TRAINING_ROOT}" \
    --output "${OUTPUT_PATH}"
