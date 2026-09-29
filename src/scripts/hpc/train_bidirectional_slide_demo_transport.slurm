#!/bin/bash
#SBATCH --job-name=maniskill-bidir-transport
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly CONFIG_PATH="${2:?Demo transport config is required}"
readonly DATA_ROOT="${3:?paired predictor data root is required}"
readonly OUTPUT_ROOT="${4:?new transport output root is required}"
readonly BASELINE_REPORT="${DATA_ROOT}/demo_baselines_v1_0a37540.json"
readonly PYTHON_BIN="/scratch/ll5582/conda/envs/FuseICL-maniskill/bin/python"

if [[ -e "${OUTPUT_ROOT}" ]]; then
    echo "输出目录已存在，拒绝覆盖：${OUTPUT_ROOT}" >&2
    exit 2
fi
test -x "${PYTHON_BIN}"
test -s "${DATA_ROOT}/branch_samples.npz"
test -s "${BASELINE_REPORT}"
mkdir -p /scratch/ll5582/data/ManiSkill3/logs

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m dev.simulator.train_bidirectional_slide_demo_transport \
    --project-root "${PROJECT_ROOT}" \
    --config "${CONFIG_PATH}" \
    --data-root "${DATA_ROOT}" \
    --baseline-report "${BASELINE_REPORT}" \
    --output-root "${OUTPUT_ROOT}"
