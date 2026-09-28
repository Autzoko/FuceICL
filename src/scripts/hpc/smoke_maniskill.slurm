#!/bin/bash
#SBATCH --job-name=maniskill-smoke
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly OUTPUT_PATH="${2:?output report path is required}"
readonly ENV_ID="${3:-PickCube-v1}"
readonly OBS_MODE="${4:-state}"
readonly CONTROL_MODE="${5:-pd_ee_delta_pose}"
readonly SIM_BACKEND="${6:-physx_cpu}"
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
    python -m dev.simulator.smoke_maniskill \
    --project-root "${PROJECT_ROOT}" \
    --output "${OUTPUT_PATH}" \
    --env-id "${ENV_ID}" \
    --obs-mode "${OBS_MODE}" \
    --control-mode "${CONTROL_MODE}" \
    --sim-backend "${SIM_BACKEND}" \
    --seed 0 \
    --num-steps 2
