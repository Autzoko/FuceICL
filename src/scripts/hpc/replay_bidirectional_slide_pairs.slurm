#!/bin/bash
#SBATCH --job-name=maniskill-bidir-replay
#SBATCH --partition=visual
#SBATCH --gres=gpu:p4000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly SOURCE_ROOT="${2:?paired source dataset root is required}"
readonly OUTPUT_ROOT="${3:?new replay dataset root is required}"
readonly CONFIG_PATH="${4:?replay audit config is required}"
readonly PYTHON_BIN="/scratch/ll5582/conda/envs/FuseICL-maniskill/bin/python"
readonly CACHE_ROOT="/scratch/ll5582/.cache/fuceicl-maniskill"
readonly REPLAY_COUNT="48"

if [[ -e "${OUTPUT_ROOT}" ]]; then
    echo "输出目录已存在，拒绝覆盖：${OUTPUT_ROOT}" >&2
    exit 2
fi
for operation in left right; do
    test -s "${SOURCE_ROOT}/${operation}/trajectory.h5"
    test -s "${SOURCE_ROOT}/${operation}/trajectory.json"
done

test -x "${PYTHON_BIN}"
mkdir -p \
    /scratch/ll5582/data/ManiSkill3/logs \
    "${OUTPUT_ROOT}/left" \
    "${OUTPUT_ROOT}/right" \
    "${CACHE_ROOT}/matplotlib" \
    "${CACHE_ROOT}/xdg" \
    "${CACHE_ROOT}/torch" \
    "${CACHE_ROOT}/huggingface"

export MPLCONFIGDIR="${CACHE_ROOT}/matplotlib"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TORCH_HOME="${CACHE_ROOT}/torch"
export HF_HOME="${CACHE_ROOT}/huggingface"

cd "${PROJECT_ROOT}"
for operation in left right; do
    cp "${SOURCE_ROOT}/${operation}/trajectory.h5" \
        "${OUTPUT_ROOT}/${operation}/trajectory.h5"
    cp "${SOURCE_ROOT}/${operation}/trajectory.json" \
        "${OUTPUT_ROOT}/${operation}/trajectory.json"
    "${PYTHON_BIN}" -m dev.simulator.replay_bidirectional_slide \
        --traj-path "${OUTPUT_ROOT}/${operation}/trajectory.h5" \
        --sim-backend physx_cpu \
        --obs-mode pointcloud \
        --target-control-mode pd_ee_delta_pose \
        --save-traj \
        --record-rewards \
        --no-save-video \
        --no-allow-failure \
        --max-retry 0 \
        --count "${REPLAY_COUNT}" \
        --num-envs 1
done

"${PYTHON_BIN}" -m dev.simulator.audit_bidirectional_slide_replay \
    --project-root "${PROJECT_ROOT}" \
    --config "${CONFIG_PATH}" \
    --source-root "${SOURCE_ROOT}" \
    --replay-root "${OUTPUT_ROOT}" \
    --output "${OUTPUT_ROOT}/replay_audit_report.json"

sha256sum \
    "${OUTPUT_ROOT}/left/trajectory.pointcloud.pd_ee_delta_pose.physx_cpu.h5" \
    "${OUTPUT_ROOT}/right/trajectory.pointcloud.pd_ee_delta_pose.physx_cpu.h5" \
    "${OUTPUT_ROOT}/replay_audit_report.json"
