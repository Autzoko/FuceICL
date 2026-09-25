#!/bin/bash
#SBATCH --job-name=rlbench-mask-diagnose
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ll5582/data/RLBench/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/RLBench/logs/%x-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly REPORT_PATH="${2:?report path is required}"
readonly EPISODE_LIMIT="${3:-1}"
readonly DATA_ROOT="/scratch/ll5582/data/RLBench/raw/data"
readonly ENV_NAME="FuseICL-pointnet"

module purge
module load miniconda/3-4.11.0

mkdir -p "$(dirname "${REPORT_PATH}")" /scratch/ll5582/data/RLBench/logs
cd "${PROJECT_ROOT}"

conda run --no-capture-output -n "${ENV_NAME}" \
    python -m dev.pointnet.diagnose_robot_filter \
    --data-root "${DATA_ROOT}" \
    --config dev/pointnet/config/rlbench_smoke.json \
    --split train \
    --episode-limit "${EPISODE_LIMIT}" \
    > "${REPORT_PATH}"
