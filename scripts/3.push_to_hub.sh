#!/bin/sh
#SBATCH --partition=standard-g
#SBATCH --account=project_000000000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=60G
#SBATCH --cpus-per-task=56
#SBATCH --time=02:00:00
#SBATCH --job-name=push_model_to_hub
#SBATCH --output=/scratch/project_000000000/cache/push_to_hub_%A_%a.out
#SBATCH --error=/scratch/project_000000000/cache/push_to_hub_%A_%a.ERROR.out

# Debug mode for verbose output
set -x

# Load modules
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits

# Define the container image and HF token
CONTAINER=/scratch/project_000000000/test-time-facts/lumi-pytorch-rocm-6.2.3-python-3.12-pytorch-v2.5.1.sif
export HF_TOKEN=""

# Directories to push
DIRS=("")

CHECKPOINT_DIRS=("")

for i in "${!DIRS[@]}"; do
    echo "[INFO] Now processing: ${DIRS[$i]}"
    srun --ntasks=1 --exclusive \
         singularity exec -B /scratch/project_000000000/cache \
         "$CONTAINER" \
         bash -c "\$WITH_CONDA; python3 push_to_hub.py \
           \"${DIRS[$i]}\" \
           \"${CHECKPOINT_DIRS[$i]}\" \
           --private"
done
