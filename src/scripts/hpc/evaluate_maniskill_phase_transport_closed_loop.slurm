#!/bin/bash
#SBATCH --job-name=maniskill-phase-transport-cl
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.err

source /etc/profile
set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly DATA_ROOT="${2:?ManiSkill chunk root is required}"
readonly SOURCE_METADATA="${3:?pointcloud replay metadata is required}"
readonly OUTPUT_PATH="${4:?output report path is required}"
readonly CONFIG_PATH="${5:-dev/simulator/config/pick_cube_phase_transport_closed_loop_v1.json}"
readonly ENV_NAME="FuseICL-maniskill"
readonly CACHE_ROOT="/scratch/ll5582/.cache/fuceicl-maniskill"

module purge
module load miniconda/3-4.11.0

mkdir -p \
    /scratch/ll5582/data/ManiSkill3/logs \
    "${CACHE_ROOT}/matplotlib" \
    "${CACHE_ROOT}/xdg" \
    "${CACHE_ROOT}/torch" \
    "${CACHE_ROOT}/huggingface"

export MPLCONFIGDIR="${CACHE_ROOT}/matplotlib"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TORCH_HOME="${CACHE_ROOT}/torch"
export HF_HOME="${CACHE_ROOT}/huggingface"

cd "${PROJECT_ROOT}"

conda run --no-capture-output -n "${ENV_NAME}" \
    python -m dev.simulator.evaluate_maniskill_closed_loop \
    --project-root "${PROJECT_ROOT}" \
    --data-root "${DATA_ROOT}" \
    --source-metadata "${SOURCE_METADATA}" \
    --config "${CONFIG_PATH}" \
    --output "${OUTPUT_PATH}" \
    --device cuda
