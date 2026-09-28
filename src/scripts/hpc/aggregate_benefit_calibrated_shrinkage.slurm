#!/bin/bash
#SBATCH --job-name=rlbench-bcsg-aggregate
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/ll5582/data/RLBench/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/RLBench/logs/%x-%j.err

source /etc/profile
set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly FOLD_A="${2:?fold A root is required}"
readonly FOLD_B="${3:?fold B root is required}"
readonly FOLD_C="${4:?fold C root is required}"
readonly OUTPUT_PATH="${5:?new aggregate report path is required}"
readonly ENV_NAME="FuseICL-pointnet"

if [[ -e "${OUTPUT_PATH}" ]]; then
    echo "输出已存在，拒绝覆盖：${OUTPUT_PATH}" >&2
    exit 2
fi

module purge
module load miniconda/3-4.11.0
cd "${PROJECT_ROOT}"

conda run --no-capture-output -n "${ENV_NAME}" \
    python -m dev.predictor.aggregate_benefit_calibrated_shrinkage \
    --project-root "${PROJECT_ROOT}" \
    --fold-root "${FOLD_A}" \
    --fold-root "${FOLD_B}" \
    --fold-root "${FOLD_C}" \
    --output "${OUTPUT_PATH}" \
    --expected-task-count 18 \
    --seed 20260928 \
    --bootstrap-resamples 10000
