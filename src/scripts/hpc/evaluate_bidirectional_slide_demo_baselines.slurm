#!/bin/bash
#SBATCH --job-name=maniskill-bidir-demo-base
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly CONFIG_PATH="${2:?Demo baseline config is required}"
readonly DATA_ROOT="${3:?paired predictor data root is required}"
readonly OUTPUT_PATH="${4:?new baseline report path is required}"
readonly PYTHON_BIN="/scratch/ll5582/conda/envs/FuseICL-maniskill/bin/python"

if [[ -e "${OUTPUT_PATH}" ]]; then
    echo "输出已存在，拒绝覆盖：${OUTPUT_PATH}" >&2
    exit 2
fi
test -x "${PYTHON_BIN}"
test -s "${DATA_ROOT}/branch_samples.npz"
mkdir -p /scratch/ll5582/data/ManiSkill3/logs

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m dev.simulator.evaluate_bidirectional_slide_demo_baselines \
    --project-root "${PROJECT_ROOT}" \
    --config "${CONFIG_PATH}" \
    --data-root "${DATA_ROOT}" \
    --output "${OUTPUT_PATH}"
