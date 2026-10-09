#!/bin/bash
#SBATCH --mail-type=NONE # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/itet-stor/guptau/net_scratch/cluster/jobs/%j.out # where to store the output (%j is the JOBID), subdirectory "jobs" must exist
#SBATCH --error=/itet-stor/guptau/net_scratch/cluster/jobs/%j.err # where to store error messages
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --exclude=tikgpu10 # Exclude problematic nodes
#CommentSBATCH --account=tik-internal
#CommentSBATCH --constraint='titan_rtx|tesla_v100|titan_xp|a100_80gb'

# Probe analyses (CPU) on an extraction store from interp_extract.sbatch.
#
# Usage:
#   sbatch --export=ALL,STORE=/itet-stor/.../interp/parity_neurons_tmd/step_218500/ema scripts/interp_analyze.sbatch
#   P6 ablation (part c) — the ablated store must exist for the SAME split and t as the analysis
#   (interp_extract.sbatch ablation example extracts SPLITS=test, MAXN=500, T=0):
#   sbatch --export=ALL,STORE=...,SPLIT=test,ANALYSES=p6,ABLATED="mean=/itet-stor/.../ema_ablate-mean" scripts/interp_analyze.sbatch
#
# All analyses default to flow time t=0: the frontier inputs (positions, expansion signal) are pure
# noise there, so any information a probe finds was computed from the tree, not read off the
# flowed input. Set T=0.6 / T=0.9 for a t-sweep (the extraction stores t=0,0.6,0.9 by default).
# Results land in ${OUT} (default: <STORE>/results). Summarise with
#   python -m interp.analyses.report <OUT> --control <random-init OUT>

: "${STORE:?set STORE=/path/to/extraction/store}"
SPLIT=${SPLIT:-val}
ANALYSES=${ANALYSES:-p1,p2,p3,p5,p6}
T=${T:-0}
ABLATED=${ABLATED:-}                     # name=path[,name=path...] (P6 part c)
B=${B:-500}
JOBS=${JOBS:-8}                          # parallel worker processes (layers run in parallel); each gets
                                         # CPUs/JOBS BLAS threads. Peak memory ~2-3 GB per worker on full val.
OUT=${OUT:-${STORE}/results}

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

mkdir -p "${OUT}"
echo "store=${STORE} split=${SPLIT} analyses=${ANALYSES} out=${OUT}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export INTERP_JOBS=${JOBS}
# Resuming: every finished (target, layer) unit is checkpointed in ${OUT}/<analysis>.partial.jsonl;
# resubmitting the same command skips finished units (a change of settings starts that analysis fresh).

ABL_ARGS=()
IFS=',' read -ra ABL <<< "${ABLATED}"
for a in "${ABL[@]}"; do [[ -n "$a" ]] && ABL_ARGS+=(--ablated "$a"); done

# One failing analysis must not stop the others (or the report): run each with errexit off.
FAILED=()
IFS=',' read -ra AN <<< "${ANALYSES}"
for an in "${AN[@]}"; do
    echo "=== ${an} $(date)"
    set +o errexit
    case "${an}" in
        p1) python -m interp.analyses.p1_local     --store "${STORE}" --t "${T}" --split "${SPLIT}" --out "${OUT}" --B "${B}" ;;
        p2) python -m interp.analyses.p2_lookahead --store "${STORE}" --t "${T}" --split "${SPLIT}" --out "${OUT}" --B "${B}" ;;
        p3) python -m interp.analyses.p3_latent    --store "${STORE}" --t "${T}" --split "${SPLIT}" --out "${OUT}" --B "${B}" --tmd-input ;;
        p5) python -m interp.analyses.p5_stop      --store "${STORE}" --t "${T}" --split "${SPLIT}" --out "${OUT}" --B "${B}" ;;
        p6) python -m interp.analyses.p6_apical    --store "${STORE}" --t "${T}" --split "${SPLIT}" --out "${OUT}" --B "${B}" "${ABL_ARGS[@]}" ;;
        *) echo "unknown analysis ${an}" >&2; false ;;
    esac
    rc=$?
    set -o errexit
    [[ ${rc} -ne 0 ]] && { echo "!!! ${an} failed (exit ${rc})" >&2; FAILED+=("${an}"); }
done
python -m interp.analyses.report "${OUT}" > "${OUT}/report.txt" || echo "report failed" >&2
echo "Finished at: $(date)"
if [[ ${#FAILED[@]} -gt 0 ]]; then echo "FAILED analyses: ${FAILED[*]}" >&2; exit 1; fi
exit 0
