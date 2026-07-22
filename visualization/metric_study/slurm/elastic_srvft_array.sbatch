#!/bin/bash

#SBATCH --job-name=elastic-srvft
#SBATCH --mail-type=NONE
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --output=elastic-srvft-%A_%a.out
#SBATCH --error=elastic-srvft-%A_%a.err

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR to the prepared Elastic matrix directory}"
SHARD_COUNT="${SHARD_COUNT:?Set SHARD_COUNT to the number of prepared shards}"

PERMANENT_STORAGE_DIR="${PERMANENT_STORAGE_DIR:-/itet-stor/speltonen/net_scratch}"
PROJECT_ROOT="${PROJECT_ROOT:-${PERMANENT_STORAGE_DIR}/generating-trees}"
REPOSITORY_DIR="${REPOSITORY_DIR:-${PROJECT_ROOT}/dendrite_gen}"
CONDA_ROOT="${CONDA_ROOT:-${PERMANENT_STORAGE_DIR}/conda}"
CONDA_ENVIRONMENT="${CONDA_ENVIRONMENT:-trees}"

source "${CONDA_ROOT}/bin/activate" "${CONDA_ENVIRONMENT}"
cd "${REPOSITORY_DIR}"

# Each child process owns one CPU and one independently writable shard.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:?This job must run as a Slurm array}"
CPUS_PER_JOB="${SLURM_CPUS_PER_TASK:?Slurm must set SLURM_CPUS_PER_TASK}"
FIRST_SHARD=$((ARRAY_TASK_ID * CPUS_PER_JOB))

echo "Elastic shard group: ${ARRAY_TASK_ID}"
echo "Output: ${OUTPUT_DIR}"

PIDS=()
SHARD_IDS=()
for ((OFFSET = 0; OFFSET < CPUS_PER_JOB; OFFSET++)); do
  SHARD_ID=$((FIRST_SHARD + OFFSET))
  if ((SHARD_ID >= SHARD_COUNT)); then
    break
  fi
  srun --exclusive --ntasks=1 --cpus-per-task=1 \
    python -u -m visualization.metric_study.run_elastic_distance_matrix \
    compute-shard --output-dir "${OUTPUT_DIR}" --shard-id "${SHARD_ID}" &
  PIDS+=("$!")
  SHARD_IDS+=("${SHARD_ID}")
done

FAILED=0
for INDEX in "${!PIDS[@]}"; do
  if ! wait "${PIDS[${INDEX}]}"; then
    echo "Elastic shard ${SHARD_IDS[${INDEX}]} failed" >&2
    FAILED=1
  fi
done
exit "${FAILED}"
