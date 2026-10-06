#!/bin/bash
#SBATCH --mail-type=NONE # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/itet-stor/guptau/net_scratch/cluster/jobs/%j.out # where to store the output (%j is the JOBID), subdirectory "jobs" must exist
#SBATCH --error=/itet-stor/guptau/net_scratch/cluster/jobs/%j.err # where to store error messages
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=tikgpu10 # Exclude problematic nodes
#SBATCH --nodelist=tikgpu07 # Specify that it should run on this particular node
#CommentSBATCH --account=tik-internal
#CommentSBATCH --constraint='titan_rtx|tesla_v100|titan_xp|a100_80gb'

# Teacher-forced activation extraction for the interp probes (interp/extract.py).
#
# Writes ${OUT}/t<t>/<split>/{acts,graph_acts,index.parquet,...}; see interp/store.py.
#
# Usage:
#   sbatch --export=ALL,CKPT=/path/to/checkpoints/step_218500.pt,CONFIG=parity_neurons_uncond scripts/interp_extract.sbatch
#   sbatch --export=ALL,CONFIG=parity_neurons_uncond,WHICH=random:0,T=0,DTYPE=float32 scripts/interp_extract.sbatch   # random-init control
#   sbatch --export=ALL,CKPT=...,CONFIG=...,SPLITS=val,MAXN=20,OUT=/tmp/x scripts/interp_extract.sbatch    # smoke run
#   sbatch --export=ALL,CKPT=...,CONFIG=...,SPLITS=test,MAXN=500,T=0,ABLATE=mean scripts/interp_extract.sbatch  # P6 ablation

# ---- experiment parameters (override with sbatch --export=ALL,VAR=value) ----
CONFIG=${CONFIG:-parity_neurons_uncond}  # hydra config the checkpoint was trained with
WHICH=${WHICH:-ema}                      # ema | raw | random:<seed>
CKPT=${CKPT:-}                           # required unless WHICH=random:*
SPLITS=${SPLITS:-val,test}
T=${T:-0}                                # all analyses use t=0; add 0.6,0.9 only for a t-sweep
SEED=${SEED:-0}
ABLATE=${ABLATE:-none}                   # none | zero | mean | graphmean (P6 global-attention ablation)
DTYPE=${DTYPE:-float16}                  # use float32 for WHICH=random:* (deep layers overflow fp16)
MAXN=${MAXN:-}                           # per-split neuron cap (debug)
DATA_DIR=${DATA_DIR:-/scratch/guptau/neurons_conditional_full}
TAG=${WHICH%%:*}; [[ "${WHICH}" == random:* ]] && TAG="random${WHICH#random:}"
[[ "${ABLATE}" != none ]] && TAG="${TAG}_ablate-${ABLATE}"
CKPT_TAG=nockpt; [[ -n "${CKPT}" ]] && CKPT_TAG=$(basename "${CKPT%.pt}")
OUT=${OUT:-/itet-stor/guptau/net_scratch/interp/${CONFIG}/${CKPT_TAG}/${TAG}}


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

if [[ "${WHICH}" != random* ]]; then
    [[ -n "${CKPT}" && -f "${CKPT}" ]] || { echo "CKPT not found: '${CKPT}'" >&2; exit 1; }
fi
mkdir -p "${OUT}"
echo "config=${CONFIG} which=${WHICH} ckpt=${CKPT} splits=${SPLITS} t=${T} out=${OUT}"

python -m interp.extract --config "${CONFIG}" --which "${WHICH}" ${CKPT:+--ckpt "${CKPT}"} \
    --splits "${SPLITS}" --t "${T}" --seed "${SEED}" --data-dir "${DATA_DIR}" --out "${OUT}" --dtype "${DTYPE}" --ablate "${ABLATE}" \
    ${MAXN:+--max-neurons "${MAXN}"}

du -sh "${OUT}"
echo "Finished at: $(date)"
exit 0
