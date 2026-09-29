#!/bin/bash
#SBATCH --job-name=layout-predictor-data
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.err

source /etc/profile
module purge
module load miniconda/3-4.11.0
set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly REPLAY_ROOT="${2:?replay root is required}"
readonly CONFIG_PATH="${3:?predictor data config is required}"
readonly OUTPUT_ROOT="${4:?new predictor data root is required}"
readonly PYTHON_BIN="/scratch/ll5582/conda/envs/FuseICL-maniskill/bin/python"
readonly AUDIT_PATH="${REPLAY_ROOT}/replay_audit_report.json"

if [[ -e "${OUTPUT_ROOT}" ]]; then
    echo "输出目录已存在，拒绝覆盖：${OUTPUT_ROOT}" >&2
    exit 2
fi
test -s "${AUDIT_PATH}"
test -x "${PYTHON_BIN}"
mkdir -p /scratch/ll5582/data/ManiSkill3/logs "$(dirname "${OUTPUT_ROOT}")"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m dev.simulator.preprocess_rotated_layout_slide_predictor \
    --project-root "${PROJECT_ROOT}" \
    --config "${CONFIG_PATH}" \
    --replay-root "${REPLAY_ROOT}" \
    --audit "${AUDIT_PATH}" \
    --output-root "${OUTPUT_ROOT}"
