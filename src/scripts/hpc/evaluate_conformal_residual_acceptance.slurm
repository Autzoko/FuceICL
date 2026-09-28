#!/bin/bash
#SBATCH --job-name=maniskill-codra
#SBATCH --partition=visual
#SBATCH --gres=gpu:p4000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.err

source /etc/profile
set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly CHECKPOINT="${2:?frozen QA-LRDT/BCSG checkpoint is required}"
readonly CONFIG_PATH="${3:?CoDRA config is required}"
readonly OUTPUT_ROOT="${4:?new output root is required}"
shift 4
readonly ENV_NAME="FuseICL-maniskill"

if [[ "$#" -ne 6 ]]; then
    echo "CoDRA 固定要求三个 bank roots 和三个 evaluation roots" >&2
    exit 2
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
    echo "输出已存在，拒绝覆盖：${OUTPUT_ROOT}" >&2
    exit 2
fi

module purge
module load miniconda/3-4.11.0
mkdir -p /scratch/ll5582/data/ManiSkill3/logs "$(dirname "${OUTPUT_ROOT}")"
cd "${PROJECT_ROOT}"

bank_arguments=()
evaluation_arguments=()
for root in "${@:1:3}"; do
    bank_arguments+=(--bank-root "${root}")
done
for root in "${@:4:3}"; do
    evaluation_arguments+=(--evaluation-root "${root}")
done

conda run --no-capture-output -n "${ENV_NAME}" \
    python -m dev.simulator.evaluate_conformal_residual_acceptance \
    --project-root "${PROJECT_ROOT}" \
    --checkpoint "${CHECKPOINT}" \
    --config "${CONFIG_PATH}" \
    --output-root "${OUTPUT_ROOT}" \
    --device cuda \
    "${bank_arguments[@]}" \
    "${evaluation_arguments[@]}"
