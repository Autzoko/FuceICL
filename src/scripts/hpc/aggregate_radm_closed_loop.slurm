#!/bin/bash
#SBATCH --job-name=maniskill-radm-cl-agg
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.err

source /etc/profile
set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly PICK_REPORT="${2:?Pick report is required}"
readonly PUSH_REPORT="${3:?Push report is required}"
readonly PULL_REPORT="${4:?Pull report is required}"
readonly OUTPUT_PATH="${5:?new aggregate output is required}"
readonly ENV_NAME="FuseICL-maniskill"

if [[ -e "${OUTPUT_PATH}" ]]; then
    echo "输出已存在，拒绝覆盖：${OUTPUT_PATH}" >&2
    exit 2
fi

module purge
module load miniconda/3-4.11.0
mkdir -p /scratch/ll5582/data/ManiSkill3/logs "$(dirname "${OUTPUT_PATH}")"
cd "${PROJECT_ROOT}"

conda run --no-capture-output -n "${ENV_NAME}" \
    python -m dev.simulator.aggregate_radm_closed_loop \
    --project-root "${PROJECT_ROOT}" \
    --report "${PICK_REPORT}" \
    --report "${PUSH_REPORT}" \
    --report "${PULL_REPORT}" \
    --output "${OUTPUT_PATH}"
