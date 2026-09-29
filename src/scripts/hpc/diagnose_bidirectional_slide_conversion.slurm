#!/bin/bash
#SBATCH --job-name=maniskill-bidir-diag
#SBATCH --partition=visual
#SBATCH --gres=gpu:p4000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly SOURCE_H5="${2:?right source HDF5 is required}"
readonly OUTPUT_ROOT="${3:?new diagnostic output root is required}"
readonly PYTHON_BIN="/scratch/ll5582/conda/envs/FuseICL-maniskill/bin/python"
readonly SOURCE_JSON="${SOURCE_H5%.h5}.json"
readonly REPLAY_H5="${OUTPUT_ROOT}/trajectory.pointcloud.pd_ee_delta_pose.physx_cpu.h5"
readonly REPLAY_JSON="${OUTPUT_ROOT}/trajectory.pointcloud.pd_ee_delta_pose.physx_cpu.json"

if [[ -e "${OUTPUT_ROOT}" ]]; then
    echo "输出目录已存在，拒绝覆盖：${OUTPUT_ROOT}" >&2
    exit 2
fi
test -x "${PYTHON_BIN}"
test -s "${SOURCE_H5}"
test -s "${SOURCE_JSON}"
mkdir -p \
    /scratch/ll5582/data/ManiSkill3/logs \
    /scratch/ll5582/.cache/fuceicl-maniskill/matplotlib \
    /scratch/ll5582/.cache/fuceicl-maniskill/xdg
export MPLCONFIGDIR="/scratch/ll5582/.cache/fuceicl-maniskill/matplotlib"
export XDG_CACHE_HOME="/scratch/ll5582/.cache/fuceicl-maniskill/xdg"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m dev.simulator.extract_maniskill_episodes \
    --source-h5 "${SOURCE_H5}" \
    --source-json "${SOURCE_JSON}" \
    --episode-id 41 \
    --episode-id 42 \
    --output-root "${OUTPUT_ROOT}"

"${PYTHON_BIN}" -m dev.simulator.replay_bidirectional_slide \
    --traj-path "${OUTPUT_ROOT}/trajectory.h5" \
    --sim-backend physx_cpu \
    --obs-mode pointcloud \
    --target-control-mode pd_ee_delta_pose \
    --save-traj \
    --record-rewards \
    --no-save-video \
    --allow-failure \
    --verbose \
    --max-retry 0 \
    --count 2 \
    --num-envs 1

"${PYTHON_BIN}" -m dev.simulator.diagnose_bidirectional_slide_conversion \
    --source-h5 "${OUTPUT_ROOT}/trajectory.h5" \
    --source-json "${OUTPUT_ROOT}/trajectory.json" \
    --replay-h5 "${REPLAY_H5}" \
    --replay-json "${REPLAY_JSON}" \
    --output "${OUTPUT_ROOT}/diagnosis_report.json"
