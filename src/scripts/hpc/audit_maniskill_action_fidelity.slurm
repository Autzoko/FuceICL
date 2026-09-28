#!/bin/bash
#SBATCH --job-name=maniskill-action-fidelity
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.err

# 非 login Slurm shell 不保证预加载 Environment Modules；profile 必须在 nounset 前读取。
source /etc/profile
set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly H5_PATH="${2:?pointcloud replay HDF5 is required}"
readonly METADATA_PATH="${3:?pointcloud replay metadata is required}"
readonly OUTPUT_PATH="${4:?output report path is required}"
readonly CONFIG_PATH="${5:-dev/simulator/config/pick_cube_action_fidelity_v1.json}"
readonly ENV_NAME="FuseICL-maniskill"
readonly CACHE_ROOT="/scratch/ll5582/.cache/fuceicl-maniskill"

module purge
module load miniconda/3-4.11.0

mkdir -p \
    /scratch/ll5582/data/ManiSkill3/logs \
    "${CACHE_ROOT}/matplotlib" \
    "${CACHE_ROOT}/xdg" \
    "${CACHE_ROOT}/torch"

export MPLCONFIGDIR="${CACHE_ROOT}/matplotlib"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TORCH_HOME="${CACHE_ROOT}/torch"

cd "${PROJECT_ROOT}"

conda run --no-capture-output -n "${ENV_NAME}" \
    python -m dev.simulator.audit_maniskill_action_fidelity \
    --project-root "${PROJECT_ROOT}" \
    --h5 "${H5_PATH}" \
    --metadata "${METADATA_PATH}" \
    --config "${CONFIG_PATH}" \
    --output "${OUTPUT_PATH}"
