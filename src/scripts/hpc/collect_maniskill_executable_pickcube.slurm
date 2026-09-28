#!/bin/bash
#SBATCH --job-name=maniskill-executable-demos
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.err

# Environment Modules 需要在严格 shell 选项前初始化。
source /etc/profile
set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly OUTPUT_ROOT="${2:?new output root is required}"
readonly NUM_TRAJECTORIES="${3:-32}"
readonly ENV_NAME="FuseICL-maniskill"
readonly CACHE_ROOT="/scratch/ll5582/.cache/fuceicl-maniskill"
readonly SOURCE_DIR="${OUTPUT_ROOT}/PickCube-v1/motionplanning"
readonly SOURCE_H5="${SOURCE_DIR}/trajectory.h5"
readonly SOURCE_JSON="${SOURCE_DIR}/trajectory.json"
readonly TARGET_H5="${SOURCE_DIR}/trajectory.pointcloud.pd_ee_delta_pose.physx_cpu.h5"
readonly TARGET_JSON="${SOURCE_DIR}/trajectory.pointcloud.pd_ee_delta_pose.physx_cpu.json"
readonly AUDIT_JSON="${OUTPUT_ROOT}/pointcloud_replay_audit.json"

if [[ -e "${OUTPUT_ROOT}" ]]; then
    echo "输出目录已存在，拒绝覆盖：${OUTPUT_ROOT}" >&2
    exit 2
fi
if [[ ! "${NUM_TRAJECTORIES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_TRAJECTORIES 必须为正整数" >&2
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

# 官方 Panda motion-planning expert 只在数据采集时读取仿真真值。
conda run --no-capture-output -n "${ENV_NAME}" \
    python -m mani_skill.examples.motionplanning.panda.run \
    --env-id PickCube-v1 \
    --obs-mode none \
    --num-traj "${NUM_TRAJECTORIES}" \
    --only-count-success \
    --sim-backend physx_cpu \
    --traj-name trajectory \
    --record-dir "${OUTPUT_ROOT}"

test -s "${SOURCE_H5}"
test -s "${SOURCE_JSON}"

# 不使用 env-state forcing；只有 action replay 真正成功的转换轨迹才保存。
conda run --no-capture-output -n "${ENV_NAME}" \
    python -m mani_skill.trajectory.replay_trajectory \
    --traj-path "${SOURCE_H5}" \
    --sim-backend physx_cpu \
    --obs-mode pointcloud \
    --target-control-mode pd_ee_delta_pose \
    --save-traj \
    --record-rewards \
    --no-save-video \
    --no-allow-failure \
    --max-retry 2 \
    --count "${NUM_TRAJECTORIES}" \
    --num-envs 1

test -s "${TARGET_H5}"
test -s "${TARGET_JSON}"

conda run --no-capture-output -n "${ENV_NAME}" \
    python -m dev.simulator.audit_replayed_trajectory \
    --h5 "${TARGET_H5}" \
    --json "${TARGET_JSON}" \
    --output "${AUDIT_JSON}"

conda run --no-capture-output -n "${ENV_NAME}" python -c \
    'import json,sys; p=json.load(open(sys.argv[1])); n=int(sys.argv[2]); assert p["num_episodes"] == n; assert p["metadata_success_rate"] == 1.0; assert p["terminal_success_rate"] == 1.0' \
    "${AUDIT_JSON}" "${NUM_TRAJECTORIES}"

sha256sum \
    "${SOURCE_H5}" \
    "${SOURCE_JSON}" \
    "${TARGET_H5}" \
    "${TARGET_JSON}" \
    "${AUDIT_JSON}"
