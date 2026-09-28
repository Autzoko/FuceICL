#!/bin/bash
#SBATCH --job-name=rccs-feasibility
#SBATCH --partition=visual
#SBATCH --gres=gpu:p4000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.err

source /etc/profile
set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly CHECKPOINT="${2:?frozen QA-LRDT/BCSG checkpoint is required}"
readonly OUTPUT_PATH="${3:?new report path is required}"
shift 3
readonly ENV_NAME="FuseICL-maniskill"

if [[ "$#" -ne 3 ]]; then
    echo "RCCS audit 固定要求三个 ManiSkill task data roots" >&2
    exit 2
fi
if [[ -e "${OUTPUT_PATH}" ]]; then
    echo "输出已存在，拒绝覆盖：${OUTPUT_PATH}" >&2
    exit 2
fi

module purge
module load miniconda/3-4.11.0
mkdir -p /scratch/ll5582/data/ManiSkill3/logs "$(dirname "${OUTPUT_PATH}")"
cd "${PROJECT_ROOT}"

data_arguments=()
for data_root in "$@"; do
    data_arguments+=(--data-root "${data_root}")
done

conda run --no-capture-output -n "${ENV_NAME}" \
    python -m dev.simulator.analyze_retrieval_cell_certificate \
    --project-root "${PROJECT_ROOT}" \
    --checkpoint "${CHECKPOINT}" \
    --output "${OUTPUT_PATH}" \
    --device cuda \
    "${data_arguments[@]}"
