#!/bin/bash
#SBATCH --job-name=layout-demo-policy
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly CONFIG_PATH="${2:?Demo policy config is required}"
readonly DATA_ROOT="${3:?predictor data root is required}"
readonly OUTPUT_ROOT="${4:?new policy output root is required}"
readonly PYTHON_BIN="/scratch/ll5582/conda/envs/FuseICL-maniskill/bin/python"
export LD_LIBRARY_PATH="/scratch/ll5582/conda/envs/FuseICL-maniskill/lib:${LD_LIBRARY_PATH:-}"

if [[ -e "${OUTPUT_ROOT}" ]]; then
    echo "输出目录已存在，拒绝覆盖：${OUTPUT_ROOT}" >&2
    exit 2
fi
test -x "${PYTHON_BIN}"
test -s "${DATA_ROOT}/branch_samples.npz"
test -s "${DATA_ROOT}/preprocess_report.json"
mkdir -p /scratch/ll5582/data/ManiSkill3/logs "$(dirname "${OUTPUT_ROOT}")"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m dev.simulator.train_rotated_layout_slide_demo_policy \
    --project-root "${PROJECT_ROOT}" \
    --config "${CONFIG_PATH}" \
    --data-root "${DATA_ROOT}" \
    --output-root "${OUTPUT_ROOT}"
