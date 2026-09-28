#!/bin/bash
#SBATCH --job-name=maniskill-replay-audit
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.err

# Environment Modules 需要在严格 shell 选项前初始化。
source /etc/profile
set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly H5_PATH="${2:?replayed HDF5 path is required}"
readonly METADATA_PATH="${3:?replayed metadata path is required}"
readonly OUTPUT_PATH="${4:?audit output path is required}"
readonly ENV_NAME="FuseICL-maniskill"

if [[ -e "${OUTPUT_PATH}" ]]; then
    echo "输出已存在，拒绝覆盖：${OUTPUT_PATH}" >&2
    exit 2
fi

module purge
module load miniconda/3-4.11.0

mkdir -p /scratch/ll5582/data/ManiSkill3/logs
cd "${PROJECT_ROOT}"

conda run --no-capture-output -n "${ENV_NAME}" \
    python -m dev.simulator.audit_replayed_trajectory \
    --h5 "${H5_PATH}" \
    --json "${METADATA_PATH}" \
    --output "${OUTPUT_PATH}"
