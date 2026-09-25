#!/bin/bash
#SBATCH --job-name=rlbench-retrieval-adapter
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/ll5582/data/RLBench/logs/%x-%j.out
#SBATCH --error=/scratch/ll5582/data/RLBench/logs/%x-%j.err

set -euo pipefail

module purge
module load python/3.9.0

readonly PROJECT_ROOT="${1:?project root is required}"
readonly DATA_ROOT="/scratch/ll5582/data/RLBench/raw/data"
readonly OUTPUT_ROOT="/scratch/ll5582/data/RLBench/processed/retrieval_eval_v1"

mkdir -p "${OUTPUT_ROOT}" /scratch/ll5582/data/RLBench/logs
cd "${PROJECT_ROOT}"

python "${PROJECT_ROOT}/dev/retriever/rlbench_adapter.py" \
    --root "${DATA_ROOT}" \
    --output "${OUTPUT_ROOT}/chunks.jsonl" \
    --candidate-episodes 8 \
    --query-episodes 4 \
    --tasks \
        close_jar \
        insert_onto_square_peg \
        place_shape_in_shape_sorter \
        put_money_in_safe \
        slide_block_to_color_target \
        stack_blocks

sha256sum "${OUTPUT_ROOT}/chunks.jsonl" > "${OUTPUT_ROOT}/chunks.jsonl.sha256"
date -u +%Y-%m-%dT%H:%M:%SZ > "${OUTPUT_ROOT}/EXTRACTION_COMPLETE"
