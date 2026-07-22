#!/bin/bash

#SBATCH --job-name=elastic-merge
#SBATCH --mail-type=NONE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --output=elastic-merge-%j.out
#SBATCH --error=elastic-merge-%j.err

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR to the prepared Elastic matrix directory}"

PERMANENT_STORAGE_DIR="${PERMANENT_STORAGE_DIR:-/itet-stor/speltonen/net_scratch}"
PROJECT_ROOT="${PROJECT_ROOT:-${PERMANENT_STORAGE_DIR}/generating-trees}"
REPOSITORY_DIR="${REPOSITORY_DIR:-${PROJECT_ROOT}/dendrite_gen}"
CONDA_ROOT="${CONDA_ROOT:-${PERMANENT_STORAGE_DIR}/conda}"
CONDA_ENVIRONMENT="${CONDA_ENVIRONMENT:-trees}"

source "${CONDA_ROOT}/bin/activate" "${CONDA_ENVIRONMENT}"
cd "${REPOSITORY_DIR}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

echo "Merging Elastic shards in: ${OUTPUT_DIR}"
srun python -u -m visualization.metric_study.run_elastic_distance_matrix \
  merge --output-dir "${OUTPUT_DIR}"
