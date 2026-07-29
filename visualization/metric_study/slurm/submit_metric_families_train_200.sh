#!/bin/bash

set -euo pipefail

# Large train-split study: all implemented metric families except Elastic SRVFT.
SCRIPT_DIR="/itet-stor/speltonen/net_scratch/generating-trees/dendrite_gen/visualization/metric_study/slurm"
SBATCH_SCRIPT="${SCRIPT_DIR}/metric_family.sbatch"

PERMANENT_STORAGE_DIR="${PERMANENT_STORAGE_DIR:-/itet-stor/speltonen/net_scratch}"
PROJECT_ROOT="${PROJECT_ROOT:-${PERMANENT_STORAGE_DIR}/generating-trees}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/log/metric_study}"
mkdir -p "${LOG_DIR}"

SPLIT="${SPLIT:-train}"
PER_CLASS="${PER_CLASS:-200}"
SEED="${SEED:-0}"
RUN_NAME="${RUN_NAME:-balanced_${SPLIT}_${PER_CLASS}_seed${SEED}}"

if [[ "$#" -eq 0 ]]; then
  FAMILIES=(chamfer barcodes distributions morphometrics fgw)
else
  FAMILIES=("$@")
fi

for FAMILY in "${FAMILIES[@]}"; do
  ARRAY_TASKS=""
  MAX_NEW_PAIRS=""

  case "${FAMILY}" in
    chamfer)
      CPUS="${CHAMFER_CPUS:-8}"
      JOB_TIME="${CHAMFER_TIME_LIMIT:-${TIME_LIMIT:-08:00:00}}"
      ARRAY_TASKS="${CHAMFER_CHUNKS:-25}"
      MAX_NEW_PAIRS="${CHAMFER_MAX_NEW_PAIRS:-40000}"
      ;;
    barcodes)
      CPUS="${BARCODE_CPUS:-4}"
      JOB_TIME="${BARCODE_TIME_LIMIT:-${TIME_LIMIT:-04:00:00}}"
      ;;
    distributions)
      CPUS="${DISTRIBUTION_CPUS:-4}"
      JOB_TIME="${DISTRIBUTION_TIME_LIMIT:-${TIME_LIMIT:-06:00:00}}"
      ;;
    morphometrics)
      CPUS="${MORPHOMETRIC_CPUS:-1}"
      JOB_TIME="${MORPHOMETRIC_TIME_LIMIT:-${TIME_LIMIT:-02:00:00}}"
      ;;
    fgw)
      CPUS="${FGW_CPUS:-8}"
      JOB_TIME="${FGW_TIME_LIMIT:-${TIME_LIMIT:-08:00:00}}"
      ARRAY_TASKS="${FGW_CHUNKS:-25}"
      MAX_NEW_PAIRS="${FGW_MAX_NEW_PAIRS:-40000}"
      ;;
    *)
      echo "Unknown metric family: ${FAMILY}" >&2
      exit 2
      ;;
  esac

  EXPORTS="ALL,METRIC_FAMILY=${FAMILY},SPLIT=${SPLIT},PER_CLASS=${PER_CLASS},SEED=${SEED},RUN_NAME=${RUN_NAME}"

  if [[ -n "${ARRAY_TASKS}" ]]; then
    if [[ ! "${ARRAY_TASKS}" =~ ^[1-9][0-9]*$ ]]; then
      echo "${FAMILY} chunk count must be a positive integer: ${ARRAY_TASKS}" >&2
      exit 2
    fi
    if [[ ! "${MAX_NEW_PAIRS}" =~ ^[1-9][0-9]*$ ]]; then
      echo "${FAMILY} pair budget must be a positive integer: ${MAX_NEW_PAIRS}" >&2
      exit 2
    fi

    sbatch \
      --job-name="tree-train200-${FAMILY}" \
      --array="0-$((ARRAY_TASKS - 1))%1" \
      --cpus-per-task="${CPUS}" \
      --time="${JOB_TIME}" \
      --output="${LOG_DIR}/%x-%A_%a.out" \
      --error="${LOG_DIR}/%x-%A_%a.err" \
      --export="${EXPORTS},MAX_NEW_PAIRS=${MAX_NEW_PAIRS}" \
      "${SBATCH_SCRIPT}"
  else
    sbatch \
      --job-name="tree-train200-${FAMILY}" \
      --cpus-per-task="${CPUS}" \
      --time="${JOB_TIME}" \
      --output="${LOG_DIR}/%x-%j.out" \
      --error="${LOG_DIR}/%x-%j.err" \
      --export="${EXPORTS}" \
      "${SBATCH_SCRIPT}"
  fi
done
