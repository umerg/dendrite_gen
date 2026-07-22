#!/bin/bash

set -euo pipefail

PERMANENT_STORAGE_DIR="${PERMANENT_STORAGE_DIR:-/itet-stor/speltonen/net_scratch}"
PROJECT_ROOT="${PROJECT_ROOT:-${PERMANENT_STORAGE_DIR}/generating-trees}"
REPOSITORY_DIR="${REPOSITORY_DIR:-${PROJECT_ROOT}/dendrite_gen}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/data/neurons_conditional}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/metric_study/matrices}"
CONDA_ROOT="${CONDA_ROOT:-${PERMANENT_STORAGE_DIR}/conda}"
CONDA_ENVIRONMENT="${CONDA_ENVIRONMENT:-trees}"
SCRIPT_DIR="${REPOSITORY_DIR}/visualization/metric_study/slurm"
ARRAY_SCRIPT="${SCRIPT_DIR}/elastic_srvft_array.sbatch"
MERGE_SCRIPT="${SCRIPT_DIR}/elastic_srvft_merge.sbatch"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/log/metric_study}"

SPLIT="${SPLIT:-test}"
MAX_TREES_PER_CLASS="${MAX_TREES_PER_CLASS:-20}"
SEED="${SEED:-0}"
PAIRS_PER_SHARD="${PAIRS_PER_SHARD:-8}"
SO2_GRID_SIZE="${SO2_GRID_SIZE:-36}"
REFINEMENT_TOLERANCE="${REFINEMENT_TOLERANCE:-1e-3}"
MAX_CONCURRENT="${MAX_CONCURRENT:-100}"
RUN_NAME="${RUN_NAME:-elastic_full_depth_${SPLIT}_cap${MAX_TREES_PER_CLASS}_grid${SO2_GRID_SIZE}_seed${SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}/elastic}"
ELASTIC_CHECKOUT="${ELASTIC_CHECKOUT:-}"

for NAME in MAX_TREES_PER_CLASS PAIRS_PER_SHARD SO2_GRID_SIZE MAX_CONCURRENT; do
  VALUE="${!NAME}"
  if [[ ! "${VALUE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${NAME} must be a positive integer, got: ${VALUE}" >&2
    exit 2
  fi
done
if [[ "${SO2_GRID_SIZE}" -lt 3 ]]; then
  echo "SO2_GRID_SIZE must be at least 3, got: ${SO2_GRID_SIZE}" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"
source "${CONDA_ROOT}/bin/activate" "${CONDA_ENVIRONMENT}"
cd "${REPOSITORY_DIR}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

if [[ ! -f "${OUTPUT_DIR}/run.json" ]]; then
  PREPARE_COMMAND=(
    python -u -m visualization.metric_study.run_elastic_distance_matrix
    prepare
    --dataset-root "${DATASET_ROOT}"
    --split "${SPLIT}"
    --max-trees-per-class "${MAX_TREES_PER_CLASS}"
    --seed "${SEED}"
    --pairs-per-shard "${PAIRS_PER_SHARD}"
    --so2-grid-size "${SO2_GRID_SIZE}"
    --refinement-tolerance "${REFINEMENT_TOLERANCE}"
    --output-dir "${OUTPUT_DIR}"
  )
  if [[ -n "${ELASTIC_CHECKOUT}" ]]; then
    PREPARE_COMMAND+=(--elastic-checkout "${ELASTIC_CHECKOUT}")
  fi
  "${PREPARE_COMMAND[@]}"
else
  echo "Reusing prepared Elastic run: ${OUTPUT_DIR}"
fi

TASK_COUNT_PATH="${OUTPUT_DIR}/task_count.txt"
if [[ ! -f "${TASK_COUNT_PATH}" ]]; then
  echo "Prepared run is missing ${TASK_COUNT_PATH}" >&2
  exit 2
fi
IFS= read -r TASK_COUNT < "${TASK_COUNT_PATH}"
if [[ ! "${TASK_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid task count in ${TASK_COUNT_PATH}: ${TASK_COUNT}" >&2
  exit 2
fi
ARRAY_END=$((TASK_COUNT - 1))

COMMON_EXPORT="ALL,PERMANENT_STORAGE_DIR=${PERMANENT_STORAGE_DIR},PROJECT_ROOT=${PROJECT_ROOT},REPOSITORY_DIR=${REPOSITORY_DIR},CONDA_ROOT=${CONDA_ROOT},CONDA_ENVIRONMENT=${CONDA_ENVIRONMENT},OUTPUT_DIR=${OUTPUT_DIR}"

ARRAY_SUBMISSION="$(
  sbatch \
    --parsable \
    --array="0-${ARRAY_END}%${MAX_CONCURRENT}" \
    --output="${LOG_DIR}/%x-%A_%a.out" \
    --error="${LOG_DIR}/%x-%A_%a.err" \
    --export="${COMMON_EXPORT}" \
    "${ARRAY_SCRIPT}"
)"
ARRAY_JOB_ID="${ARRAY_SUBMISSION%%;*}"

MERGE_SUBMISSION="$(
  sbatch \
    --parsable \
    --dependency="afterany:${ARRAY_JOB_ID}" \
    --output="${LOG_DIR}/%x-%j.out" \
    --error="${LOG_DIR}/%x-%j.err" \
    --export="${COMMON_EXPORT}" \
    "${MERGE_SCRIPT}"
)"

echo "Output: ${OUTPUT_DIR}"
echo "Array job: ${ARRAY_SUBMISSION} (${TASK_COUNT} shards, max ${MAX_CONCURRENT} active)"
echo "Merge job: ${MERGE_SUBMISSION} (afterany:${ARRAY_JOB_ID})"
