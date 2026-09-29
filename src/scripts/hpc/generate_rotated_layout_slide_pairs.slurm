#!/bin/bash
#SBATCH --job-name=maniskill-layout-data
#SBATCH --partition=visual
#SBATCH --gres=gpu:p4000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly CONFIG_PATH="${2:?collection config is required}"
readonly OUTPUT_ROOT="${3:?new paired dataset root is required}"
readonly PYTHON_BIN="/scratch/ll5582/conda/envs/FuseICL-maniskill/bin/python"
readonly CACHE_ROOT="/scratch/ll5582/.cache/fuceicl-maniskill"
export LD_LIBRARY_PATH="/scratch/ll5582/conda/envs/FuseICL-maniskill/lib:${LD_LIBRARY_PATH:-}"

if [[ -e "${OUTPUT_ROOT}" ]]; then
    echo "输出目录已存在，拒绝覆盖：${OUTPUT_ROOT}" >&2
    exit 2
fi
test -x "${PYTHON_BIN}"
mkdir -p \
    /scratch/ll5582/data/ManiSkill3/logs \
    "$(dirname "${OUTPUT_ROOT}")" \
    "${CACHE_ROOT}/matplotlib" \
    "${CACHE_ROOT}/xdg"
export MPLCONFIGDIR="${CACHE_ROOT}/matplotlib"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m dev.simulator.generate_rotated_layout_slide_pairs \
    --project-root "${PROJECT_ROOT}" \
    --config "${CONFIG_PATH}" \
    --output-root "${OUTPUT_ROOT}"
