#!/bin/bash
#SBATCH --job-name=pointnet-compare
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ll5582/data/RLBench/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/RLBench/logs/%x-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly DATA_ROOT="${2:?processed data root is required}"
readonly CHECKPOINT="${3:?checkpoint path is required}"
readonly OUTPUT="${4:?report path is required}"
readonly ENV_NAME="FuseICL-pointnet"

module purge
module load miniconda/3-4.11.0

mkdir -p "$(dirname "${OUTPUT}")" /scratch/ll5582/data/RLBench/logs
cd "${PROJECT_ROOT}"

nvidia-smi
conda run --no-capture-output -n "${ENV_NAME}" \
    python -m dev.pointnet.compare_retrievers \
    --data-root "${DATA_ROOT}" \
    --checkpoint "${CHECKPOINT}" \
    --output "${OUTPUT}" \
    --device cuda \
    --batch-size 64 \
    --num-workers 4
