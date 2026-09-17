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

# k-seed TMD-conditional generation on the TEST set.
#
# Runs the ordinary test pass (main.py testing=True training.resume=<ckpt>) once per seed,
# inside ONE process so the 22k-neuron train dataset is only built once. Each seed writes
#   ${OUT}/test/step_<N>_seed_<s>.pkl   ->  {"ema_0.999": {"pred_graphs": [...], "dist": ..., "tmd_cond": ...}}
# i.e. the usual validation-pickle shape, readable by the visualization runners with
# --ema-key ema_0.999 (GT = the test SWC dir, paired by index).
#
# Usage (CKPT is required; everything else has a default):
#   sbatch --export=ALL,CKPT=/path/to/outputs/<date>/<time>/checkpoints/step_218500.pt scripts/test_k_seeds.sbatch
#   sbatch --export=ALL,CKPT=...,K=1 scripts/test_k_seeds.sbatch                 # timing probe
#   sbatch --export=ALL,CKPT=...,CONFIG=parity_neurons_class,EXTRA="dataset.data_dir=/scratch/..." ...

ETH_USERNAME=guptau
PROJECT_NAME=dendrite_gen
DIRECTORY=/itet-stor/${ETH_USERNAME}/net_scratch/${PROJECT_NAME}
CONDA_ENVIRONMENT=GNN
mkdir -p ${DIRECTORY}/jobs

# ---- experiment parameters (override with sbatch --export=ALL,VAR=value) ----
: "${CKPT:?set CKPT=/path/to/step_N.pt (sbatch --export=ALL,CKPT=...)}"
CONFIG=${CONFIG:-parity_neurons_tmd}     # hydra config the checkpoint was trained with
K=${K:-5}                                # number of seeds (0..K-1)
EXTRA=${EXTRA:-}                         # extra hydra overrides, e.g. "dataset.data_dir=/scratch/x"
OUT=${OUT:-/itet-stor/${ETH_USERNAME}/net_scratch/k_seeds/${CONFIG}_$(basename "${CKPT%.pt}")}

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

[[ -f "${CKPT}" ]] || { echo "CKPT not found: ${CKPT}" >&2; exit 1; }
SEEDS=$(python -c "print('[' + ','.join(str(i) for i in range(${K})) + ']')")
mkdir -p "${OUT}"
echo "config=${CONFIG}  ckpt=${CKPT}  seeds=${SEEDS}  out=${OUT}  extra='${EXTRA}'"

# Execute your code
#   wandb.logging=False        : with training.resume the trainer would re-attach to the
#                                original run's wandb run_id; nothing here needs logging.
#   validation.tmd_cond_every=1: matched pairwise (gen[i] vs GT[i]) metrics on every pass
#                                regardless of the checkpoint step (they are step-gated).
#   hydra.run.dir=${OUT}       : pkls land in ${OUT}/test/ instead of outputs/<date>/<time>/.
python main.py -cn "${CONFIG}" \
    testing=True \
    training.resume="${CKPT}" \
    "+test_seeds=${SEEDS}" \
    wandb.logging=False \
    ++validation.tmd_cond_every=1 \
    hydra.run.dir="${OUT}" \
    ${EXTRA}

echo "Generated pickles:"
ls -la "${OUT}/test/"

# Send more noteworthy information to the output log
echo "Finished at: $(date)"

# End the script with exit code 0
exit 0
