#!/bin/bash
#SBATCH --job-name=maniskill-lr-dat
#SBATCH --partition=visual
#SBATCH --gres=gpu:p4000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.err

# Environment Modules 需要在严格 shell 选项前初始化。
source /etc/profile
set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly OUTPUT_ROOT="${2:?new output root is required}"
readonly CONFIG_PATH="${3:?config path is required}"
shift 3
readonly ENV_NAME="FuseICL-maniskill"

if [[ "$#" -lt 2 ]]; then
    echo "至少需要两个 ManiSkill task data roots" >&2
    exit 2
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
    echo "输出目录已存在，拒绝覆盖：${OUTPUT_ROOT}" >&2
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
    python -m dev.simulator.train_maniskill_low_rank_transport \
    --project-root "${PROJECT_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --config "${CONFIG_PATH}" \
    --device cuda \
    "${data_arguments[@]}"
