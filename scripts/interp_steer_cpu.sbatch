#!/bin/bash
#SBATCH --mail-type=NONE # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/itet-stor/guptau/net_scratch/cluster/jobs/%j.out # where to store the output (%j is the JOBID), subdirectory "jobs" must exist
#SBATCH --error=/itet-stor/guptau/net_scratch/cluster/jobs/%j.err # where to store error messages
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --exclude=tikgpu10 # Exclude problematic nodes
#SBATCH --nodelist=tikgpu09 # Specify that it should run on this particular node
#CommentSBATCH --account=tik-internal
#CommentSBATCH --constraint='titan_rtx|tesla_v100|titan_xp|a100_80gb'

# CPU side of the generation-time interventions (methodology §11b).
#
# TASK=calib    GT tall-arm rates (must reproduce ~80.3 % one tall / 19.7 % >= 2 tall)
# TASK=clf      fit the morphology cell-class classifier (train) and score it on test
# TASK=dirs     steering directions from a teacher-forced store (KIND=class|reach)
# TASK=analyze  paired analysis of one rollout experiment directory (EXP_DIR)
#
# Usage:
#   sbatch --export=ALL,TASK=calib scripts/interp_steer_cpu.sbatch
#   sbatch --export=ALL,TASK=clf,OUT=/itet-stor/guptau/net_scratch/interp/class_clf scripts/interp_steer_cpu.sbatch
#   sbatch --export=ALL,TASK=dirs,KIND=class,STORE=$U,OUT=$U/results_v3 scripts/interp_steer_cpu.sbatch
#   sbatch --export=ALL,TASK=analyze,CONFIG=parity_neurons_uncond,EXP_DIR=.../rollouts_val/e0,CLF=.../class_clf scripts/interp_steer_cpu.sbatch

TASK=${TASK:?set TASK=calib|clf|dirs|analyze}
CONFIG=${CONFIG:-parity_neurons_uncond}
SPLIT=${SPLIT:-val}
KIND=${KIND:-class}
SITES=${SITES:-}
STORE=${STORE:-}
EXP_DIR=${EXP_DIR:-}
CLF=${CLF:-}
CLASS_REF=${CLASS_REF:-}
OUT=${OUT:-}
DATA_DIR=${DATA_DIR:-/scratch/guptau/neurons_conditional_full}
c() { echo "${1//+/,}"; }

ETH_USERNAME=guptau
PROJECT_NAME=dendrite_gen
DIRECTORY=/itet-stor/${ETH_USERNAME}/net_scratch/${PROJECT_NAME}
CONDA_ENVIRONMENT=GNN
mkdir -p ${DIRECTORY}/jobs

# Exit on errors
set -o errexit

# Set a directory for temporary files unique to the job with automatic removal at job termination
TMPDIR=$(mktemp -d)
if [[ ! -d ${TMPDIR} ]]; then
echo 'Failed to create temp directory' >&2
exit 1
fi
trap "exit 1" HUP INT TERM
trap 'rm -rf "${TMPDIR}"' EXIT
export TMPDIR

# Change the current directory to the location where you want to store temporary files, exit if changing didn't succeed.
# Adapt this to your personal preference
cd "${TMPDIR}" || exit 1

# Send some noteworthy information to the output log

echo "Running on node: $(hostname)"
echo "In directory: $(pwd)"
echo "Starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"


[[ -f /itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda ]] && eval "$(/itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda shell.bash hook)"
conda activate ${CONDA_ENVIRONMENT}
echo "Conda activated"
cd ${DIRECTORY}

case "${TASK}" in
  calib)   python -m interp.metrics --config "${CONFIG}" --splits val,test --data-dir "${DATA_DIR}" ;;
  clf)     python -m interp.class_clf --config "${CONFIG}" --data-dir "${DATA_DIR}" --out "${OUT:?set OUT}" ;;
  dirs)    python -m interp.analyses.steer_dirs --store "${STORE:?set STORE}" --split "${SPLIT}" --kind "${KIND}" \
               ${SITES:+--sites "$(c ${SITES})"} --out "${OUT:-${STORE}/results_v3}" ;;
  analyze) python -m interp.analyses.s_rollouts --exp-dir "${EXP_DIR:?set EXP_DIR}" --config "${CONFIG}" \
               --split "${SPLIT}" --data-dir "${DATA_DIR}" ${CLF:+--clf "${CLF}"} ${CLASS_REF:+--class-ref "${CLASS_REF}"} \
               ${OUT:+--out "${OUT}"} ;;
  *) echo "unknown TASK ${TASK}" >&2; exit 1 ;;
esac

echo "Finished at: $(date)"
exit 0
