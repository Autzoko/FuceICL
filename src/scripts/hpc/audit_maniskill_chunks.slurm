#!/bin/bash
#SBATCH --job-name=maniskill-chunk-audit
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/ManiSkill3/logs/%x-%j.err

source /etc/profile
set -euo pipefail

readonly PROJECT_ROOT="${1:?project root is required}"
readonly CHUNK_ROOT="${2:?chunk root is required}"
readonly REPORT_PATH="${3:?audit report path is required}"
readonly ENV_NAME="FuseICL-maniskill"

module purge
module load miniconda/3-4.11.0

mkdir -p /scratch/ll5582/data/ManiSkill3/logs
cd "${PROJECT_ROOT}"

conda run --no-capture-output -n "${ENV_NAME}" \
    python -m dev.simulator.audit_maniskill_chunks \
    --root "${CHUNK_ROOT}" \
    --output "${REPORT_PATH}"
