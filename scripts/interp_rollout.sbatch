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

# Generation-time interventions (methodology §11b): rollouts for E0 / E1 S-class / E3 local growth
# via interp/rollout.py. One job = one experiment, all its conditions × seeds; finished
# <condition>_seed<s>.pkl files are skipped, so a killed job is resumed by resubmitting.
#
# Usage (OUT defaults to /itet-stor/.../interp/<CONFIG>/<ckpt>/rollouts):
#   E0 (both arms):   sbatch --export=ALL,CKPT=$CK,CONFIG=parity_neurons_uncond,EXP=e0 scripts/interp_rollout.sbatch
#   E1 layer sweep:   sbatch --export=ALL,CKPT=$CK,CONFIG=parity_neurons_uncond,EXP=class,DIRS=$D/directions_class_val_t0.npz,SITES=L3+L5+L7+L9,ALPHAS=1 scripts/interp_rollout.sbatch
#   E1 yardstick:     sbatch --export=ALL,CKPT=$CKT,CONFIG=parity_neurons_tmd,EXP=intact,TMD_CLASS=2 scripts/interp_rollout.sbatch
#   E3 site sweep:    sbatch --export=ALL,CKPT=$CK,CONFIG=...,EXP=growth,DIRS=$D/directions_growth_val_t0.npz,SITES=L7+L8+L9,ALPHAS=2,DIR_NAMES=grow scripts/interp_rollout.sbatch
#   E3 dose+controls: sbatch --export=ALL,CKPT=$CK,CONFIG=...,EXP=growth,DIRS=...,SITES=L8,ALPHAS=-4+-2+-1+1+2+4,KINDS=v+random+onmanifold,MASKS=target_lineage+target scripts/interp_rollout.sbatch
#   Phase B (val, neurons not in Phase A, fresh seeds): add SUBSET=rest,SEEDS=2+3+4
# Lists use '+' (sbatch --export splits on commas); they are converted to commas below.

CONFIG=${CONFIG:-parity_neurons_uncond}
WHICH=${WHICH:-ema}
CKPT=${CKPT:-}
EXP=${EXP:-e0}                         # intact | e0 | class | growth
SPLIT=${SPLIT:-val}
MAXN=${MAXN:-256}                      # Phase A subset size
SUBSET=${SUBSET:-phaseA}               # phaseA | rest (Phase B: val neurons not in Phase A) | all
SEEDS=${SEEDS:-0+1}
DIRS=${DIRS:-}
SITES=${SITES:-L5}
ALPHAS=${ALPHAS:-1}
KINDS=${KINDS:-v}
DIR_NAMES=${DIR_NAMES:-}
MASKS=${MASKS:-target_lineage}         # growth: target_lineage | target
TMD_CLASS=${TMD_CLASS:-}
BATCH=${BATCH:-64}
DATA_DIR=${DATA_DIR:-/scratch/guptau/neurons_conditional_full}
CKPT_TAG=nockpt; [[ -n "${CKPT}" ]] && CKPT_TAG=$(basename "${CKPT%.pt}")
OUT=${OUT:-/itet-stor/guptau/net_scratch/interp/${CONFIG}/${CKPT_TAG}/rollouts_${SPLIT}}
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

[[ -n "${CKPT}" && -f "${CKPT}" ]] || { echo "CKPT not found: '${CKPT}'" >&2; exit 1; }
mkdir -p "${OUT}"
echo "config=${CONFIG} exp=${EXP} split=${SPLIT} maxn=${MAXN} subset=${SUBSET} seeds=${SEEDS} out=${OUT}"

python -m interp.rollout --config "${CONFIG}" --which "${WHICH}" --ckpt "${CKPT}" --exp "${EXP}" \
    --split "${SPLIT}" --data-dir "${DATA_DIR}" --max-neurons "${MAXN}" --subset "${SUBSET}" --seeds "$(c ${SEEDS})" \
    --batch-items "${BATCH}" --sites "$(c ${SITES})" --alphas "$(c ${ALPHAS})" --kinds "$(c ${KINDS})" \
    --masks "$(c ${MASKS})" ${DIRS:+--dirs "${DIRS}"} ${DIR_NAMES:+--dir-names "$(c ${DIR_NAMES})"} \
    ${TMD_CLASS:+--tmd-class "${TMD_CLASS}"} --out "${OUT}"

echo "Finished at: $(date)"
exit 0
