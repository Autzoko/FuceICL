#!/bin/bash
#SBATCH --job-name=layout-transport-eval
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.err

source /etc/profile
module purge
module load miniconda/3-4.11.0
set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly DATA_ROOT="${2:?predictor data root is required}"
readonly CONFIG_PATH="${3:?transport baseline config is required}"
readonly OUTPUT_PATH="${4:?new report path is required}"
readonly PYTHON_BIN="/scratch/ll5582/conda/envs/FuseICL-maniskill/bin/python"

if [[ -e "${OUTPUT_PATH}" ]]; then
    echo "输出已存在，拒绝覆盖：${OUTPUT_PATH}" >&2
    exit 2
fi
test -s "${DATA_ROOT}/branch_samples.npz"
test -s "${DATA_ROOT}/preprocess_report.json"
test -x "${PYTHON_BIN}"
mkdir -p /scratch/ll5582/data/ManiSkill3/logs "$(dirname "${OUTPUT_PATH}")"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m dev.simulator.evaluate_rotated_layout_slide_transport \
    --project-root "${PROJECT_ROOT}" \
    --config "${CONFIG_PATH}" \
    --data-root "${DATA_ROOT}" \
    --output "${OUTPUT_PATH}"
