#!/bin/bash
#SBATCH --job-name=layout-belief-confirm
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
readonly SOURCE_ROOT="${2:?confirmation source root is required}"
readonly OUTPUT_ROOT="${3:?new confirmation output root is required}"
readonly REPLAY_CONFIG="${4:?replay audit config is required}"
readonly OCCLUSION_CONFIG="${5:?occlusion audit config is required}"
readonly BELIEF_CONFIG="${6:?belief confirmation config is required}"
readonly PYTHON_BIN="/scratch/ll5582/conda/envs/FuseICL-maniskill/bin/python"
readonly CACHE_ROOT="/scratch/ll5582/.cache/fuceicl-maniskill"
export LD_LIBRARY_PATH="/scratch/ll5582/conda/envs/FuseICL-maniskill/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${CACHE_ROOT}/matplotlib"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"

if [[ -e "${OUTPUT_ROOT}" ]]; then
    echo "输出目录已存在，拒绝覆盖：${OUTPUT_ROOT}" >&2
    exit 2
fi
test -x "${PYTHON_BIN}"
test -s "${SOURCE_ROOT}/collection_report.json"
mkdir -p \
    /scratch/ll5582/data/ManiSkill3/logs \
    "${CACHE_ROOT}/matplotlib" \
    "${CACHE_ROOT}/xdg"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" \
    -m dev.simulator.prepare_rotated_layout_multiview_replay \
    --project-root "${PROJECT_ROOT}" \
    --source-root "${SOURCE_ROOT}" \
    --output-root "${OUTPUT_ROOT}"

for operation in toward away; do
    "${PYTHON_BIN}" -m dev.simulator.replay_rotated_layout_slide \
        --traj-path "${OUTPUT_ROOT}/${operation}/trajectory.h5" \
        --sim-backend physx_cpu \
        --obs-mode pointcloud \
        --target-control-mode pd_ee_delta_pose \
        --save-traj \
        --record-rewards \
        --no-save-video \
        --no-allow-failure \
        --max-retry 0 \
        --count 24 \
        --num-envs 1
done

"${PYTHON_BIN}" -m dev.simulator.audit_rotated_layout_slide_replay \
    --project-root "${PROJECT_ROOT}" \
    --config "${REPLAY_CONFIG}" \
    --source-root "${SOURCE_ROOT}" \
    --replay-root "${OUTPUT_ROOT}" \
    --output "${OUTPUT_ROOT}/replay_audit_report.json"

"${PYTHON_BIN}" -m dev.simulator.analyze_rotated_layout_slide_occlusion \
    --project-root "${PROJECT_ROOT}" \
    --config "${OCCLUSION_CONFIG}" \
    --replay-root "${OUTPUT_ROOT}" \
    --output "${OUTPUT_ROOT}/occlusion_audit_report.json"

"${PYTHON_BIN}" -m dev.simulator.evaluate_visibility_conditioned_belief \
    --project-root "${PROJECT_ROOT}" \
    --config "${BELIEF_CONFIG}" \
    --source-report "${SOURCE_ROOT}/collection_report.json" \
    --replay-root "${OUTPUT_ROOT}" \
    --audit "${OUTPUT_ROOT}/occlusion_audit_report.json" \
    --output "${OUTPUT_ROOT}/belief_confirmation_report.json"

sha256sum \
    "${OUTPUT_ROOT}/replay_audit_report.json" \
    "${OUTPUT_ROOT}/occlusion_audit_report.json" \
    "${OUTPUT_ROOT}/belief_confirmation_report.json"
