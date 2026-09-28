#!/bin/bash
#SBATCH --job-name=maniskill-locality
#SBATCH --partition=visual
#SBATCH --gres=gpu:p4000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.err

source /etc/profile
set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly OUTPUT_PATH="${2:?new report path is required}"
readonly CONFIG_PATH="${3:?config path is required}"
readonly PICK_DATA="${4:?PickCube data root is required}"
readonly PUSH_DATA="${5:?PushCube data root is required}"
readonly PULL_DATA="${6:?PullCube data root is required}"
readonly PICK_FOLD="${7:?held-out Pick fold root is required}"
readonly PUSH_FOLD="${8:?held-out Push fold root is required}"
readonly PULL_FOLD="${9:?held-out Pull fold root is required}"
readonly ENV_NAME="FuseICL-maniskill"

if [[ -e "${OUTPUT_PATH}" ]]; then
    echo "输出已存在，拒绝覆盖：${OUTPUT_PATH}" >&2
    exit 2
fi

module purge
module load miniconda/3-4.11.0

mkdir -p /scratch/ll5582/data/ManiSkill3/logs
cd "${PROJECT_ROOT}"

conda run --no-capture-output -n "${ENV_NAME}" \
    python -m dev.simulator.evaluate_maniskill_locality_gate \
    --project-root "${PROJECT_ROOT}" \
    --config "${CONFIG_PATH}" \
    --output "${OUTPUT_PATH}" \
    --data-root "${PICK_DATA}" \
    --data-root "${PUSH_DATA}" \
    --data-root "${PULL_DATA}" \
    --fold-root "${PICK_FOLD}" \
    --fold-root "${PUSH_FOLD}" \
    --fold-root "${PULL_FOLD}"
