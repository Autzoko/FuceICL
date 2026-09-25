#!/bin/bash
#SBATCH --job-name=tiny-pointnet-retriever
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/ll5582/data/RLBench/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/RLBench/logs/%x-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly DATA_ROOT="${2:?processed data root is required}"
readonly OUTPUT_ROOT="${3:?training output root is required}"
readonly CONFIG_PATH="${4:-dev/pointnet/config/train_pilot.json}"
readonly ENV_NAME="FuseICL-pointnet"

module purge
module load miniconda/3-4.11.0

mkdir -p /scratch/ll5582/data/RLBench/logs
cd "${PROJECT_ROOT}"

nvidia-smi
conda run --no-capture-output -n "${ENV_NAME}" \
    python -m dev.pointnet.train \
    --project-root "${PROJECT_ROOT}" \
    --data-root "${DATA_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --config "${CONFIG_PATH}" \
    --device cuda
