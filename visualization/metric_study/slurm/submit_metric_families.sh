#!/bin/bash

set -euo pipefail

SCRIPT_DIR="/itet-stor/speltonen/net_scratch/generating-trees/dendrite_gen/visualization/metric_study/slurm"
SBATCH_SCRIPT="${SCRIPT_DIR}/metric_family.sbatch"

PERMANENT_STORAGE_DIR="${PERMANENT_STORAGE_DIR:-/itet-stor/speltonen/net_scratch}"
PROJECT_ROOT="${PROJECT_ROOT:-${PERMANENT_STORAGE_DIR}/generating-trees}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/log/metric_study}"
mkdir -p "${LOG_DIR}"

if [[ "$#" -eq 0 ]]; then
  FAMILIES=(chamfer barcodes distributions morphometrics fgw)
else
  FAMILIES=("$@")
fi

for FAMILY in "${FAMILIES[@]}"; do
  case "${FAMILY}" in
    chamfer)
      CPUS="${CHAMFER_CPUS:-16}"
      ;;
    barcodes)
      CPUS="${BARCODE_CPUS:-8}"
      ;;
    distributions)
      CPUS="${DISTRIBUTION_CPUS:-8}"
      ;;
    morphometrics)
      CPUS="${MORPHOMETRIC_CPUS:-4}"
      ;;
    fgw)
      CPUS="${FGW_CPUS:-16}"
      ;;
    *)
      echo "Unknown metric family: ${FAMILY}" >&2
      exit 2
      ;;
  esac

  sbatch \
    --job-name="tree-${FAMILY}" \
    --cpus-per-task="${CPUS}" \
    --output="${LOG_DIR}/%x-%j.out" \
    --error="${LOG_DIR}/%x-%j.err" \
    --export="ALL,METRIC_FAMILY=${FAMILY}" \
    "${SBATCH_SCRIPT}"
done
