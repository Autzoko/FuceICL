#!/bin/bash
#SBATCH --job-name=maniskill-chunks
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.err

source /etc/profile
set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly H5_PATH="${2:?replayed HDF5 path is required}"
readonly METADATA_PATH="${3:?replayed metadata path is required}"
readonly OUTPUT_ROOT="${4:?output root is required}"
readonly CONFIG_PATH="${5:-dev/simulator/config/pick_cube_chunks_v1.json}"
readonly ENV_NAME="FuseICL-maniskill"

module purge
module load miniconda/3-4.11.0

mkdir -p /scratch/ll5582/data/ManiSkill3/logs
cd "${PROJECT_ROOT}"

conda run --no-capture-output -n "${ENV_NAME}" \
    python -m dev.simulator.preprocess_maniskill_chunks \
    --h5 "${H5_PATH}" \
    --metadata "${METADATA_PATH}" \
    --output-root "${OUTPUT_ROOT}" \
    --config "${CONFIG_PATH}"
