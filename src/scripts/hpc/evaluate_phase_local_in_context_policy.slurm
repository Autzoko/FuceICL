#!/bin/bash
#SBATCH --job-name=maniskill-phase-local-icl
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
shift 3
readonly ENV_NAME="FuseICL-maniskill"

if [[ "$#" -lt 1 ]]; then
    echo "至少需要一个 ManiSkill task data root" >&2
    exit 2
fi
if [[ -e "${OUTPUT_PATH}" ]]; then
    echo "输出已存在，拒绝覆盖：${OUTPUT_PATH}" >&2
    exit 2
fi

module purge
module load miniconda/3-4.11.0

mkdir -p /scratch/ll5582/data/ManiSkill3/logs
cd "${PROJECT_ROOT}"

data_arguments=()
for data_root in "$@"; do
    data_arguments+=(--data-root "${data_root}")
done

conda run --no-capture-output -n "${ENV_NAME}" \
    python -m dev.simulator.evaluate_phase_local_in_context_policy \
    --project-root "${PROJECT_ROOT}" \
    --output "${OUTPUT_PATH}" \
    --config "${CONFIG_PATH}" \
    "${data_arguments[@]}"
