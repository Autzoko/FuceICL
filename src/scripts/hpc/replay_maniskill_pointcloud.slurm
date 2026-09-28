#!/bin/bash
#SBATCH --job-name=maniskill-replay-pc
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly TRAJECTORY_PATH="${2:?source trajectory.h5 is required}"
readonly REPLAY_COUNT="${3:-32}"
readonly ENV_NAME="FuseICL-maniskill"
readonly CACHE_ROOT="/scratch/ll5582/.cache/fuceicl-maniskill"
readonly SOURCE_JSON="${TRAJECTORY_PATH%.h5}.json"
readonly OUTPUT_DIRECTORY="${TRAJECTORY_PATH%/*}"
readonly OUTPUT_H5="${OUTPUT_DIRECTORY}/trajectory.pointcloud.pd_ee_delta_pose.physx_cpu.h5"
readonly OUTPUT_JSON="${OUTPUT_DIRECTORY}/trajectory.pointcloud.pd_ee_delta_pose.physx_cpu.json"

if [[ -e "${OUTPUT_H5}" || -e "${OUTPUT_JSON}" ]]; then
    echo "输出已存在，拒绝覆盖：${OUTPUT_H5}" >&2
    exit 2
fi
if [[ ! -s "${TRAJECTORY_PATH}" || ! -s "${SOURCE_JSON}" ]]; then
    echo "source HDF5/JSON 不完整：${TRAJECTORY_PATH}" >&2
    exit 2
fi
if [[ ! "${REPLAY_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "REPLAY_COUNT 必须为正整数" >&2
    exit 2
fi

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
    python -m mani_skill.trajectory.replay_trajectory \
    --traj-path "${TRAJECTORY_PATH}" \
    --sim-backend physx_cpu \
    --obs-mode pointcloud \
    --target-control-mode pd_ee_delta_pose \
    --use-env-states \
    --save-traj \
    --record-rewards \
    --no-save-video \
    --no-allow-failure \
    --count "${REPLAY_COUNT}" \
    --num-envs 1

test -s "${OUTPUT_H5}"
test -s "${OUTPUT_JSON}"
sha256sum "${OUTPUT_H5}" "${OUTPUT_JSON}"
