#!/bin/bash

#SBATCH --job-name=tree-metrics
#SBATCH --mail-type=NONE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=tree-metrics-%j.out
#SBATCH --error=tree-metrics-%j.err

set -euo pipefail

METRIC_FAMILY="${METRIC_FAMILY:?Set METRIC_FAMILY to chamfer, barcodes, distributions, or fgw}"

case "${METRIC_FAMILY}" in
  chamfer)
    METRICS=(chamfer)
    ;;
  barcodes)
    METRICS=(persistence)
    ;;
  distributions)
    METRICS=(distributions)
    ;;
  fgw)
    METRICS=(fgw)
    ;;
  *)
    echo "Unknown METRIC_FAMILY: ${METRIC_FAMILY}" >&2
    exit 2
    ;;
esac

PERMANENT_STORAGE_DIR="${PERMANENT_STORAGE_DIR:-/itet-stor/speltonen/net_scratch}"
PROJECT_ROOT="${PROJECT_ROOT:-${PERMANENT_STORAGE_DIR}/generating-trees}"
REPOSITORY_DIR="${REPOSITORY_DIR:-${PROJECT_ROOT}/dendrite_gen}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/data/neurons_conditional/neurons_conditional}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/metric_study/matrices}"
CONDA_ROOT="${CONDA_ROOT:-${PERMANENT_STORAGE_DIR}/conda}"
CONDA_ENVIRONMENT="${CONDA_ENVIRONMENT:-trees}"

SPLIT="${SPLIT:-test}"
PER_CLASS="${PER_CLASS:-50}"
SEED="${SEED:-0}"
SO2_GRID_SIZE="${SO2_GRID_SIZE:-72}"
SO2_REFINEMENT_TOLERANCE="${SO2_REFINEMENT_TOLERANCE:-1e-8}"
FGW_MAX_NODES="${FGW_MAX_NODES:-1000}"
RUN_NAME="${RUN_NAME:-balanced_${SPLIT}_${PER_CLASS}_seed${SEED}}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}/${METRIC_FAMILY}"

source "${CONDA_ROOT}/bin/activate" "${CONDA_ENVIRONMENT}"
cd "${REPOSITORY_DIR}"

# Pair-level parallelism is handled by Python worker processes.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

COMMAND=(
  python -u -m visualization.metric_study.run_distance_matrices
  --dataset-root "${DATASET_ROOT}"
  --splits "${SPLIT}"
  --selection balanced
  --per-class "${PER_CLASS}"
  --seed "${SEED}"
  --metrics "${METRICS[@]}"
  --so2-grid-size "${SO2_GRID_SIZE}"
  --so2-refinement-tolerance "${SO2_REFINEMENT_TOLERANCE}"
  --fgw-max-nodes "${FGW_MAX_NODES}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -f "${OUTPUT_DIR}/run.json" ]]; then
  COMMAND+=(--resume)
fi

echo "Metric family: ${METRIC_FAMILY}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-1}"
echo "Output: ${OUTPUT_DIR}"

srun "${COMMAND[@]}"
