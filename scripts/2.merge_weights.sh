#!/bin/bash
#SBATCH --partition=standard-g
#SBATCH --account=project_000000000 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mem-per-gpu=60G 
#SBATCH --cpus-per-task=56
#SBATCH --time=03:00:00
#SBATCH --job-name=merge_all
#SBATCH --output=/scratch/project_000000000/cache/merge_weights-%A_%a.out
#SBATCH --error=/scratch/project_000000000/cache/merge_weights_%A_%a.ERROR.out

# Debug mode for verbose output
set -x

# Load modules
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits

# Container
CONTAINER=/scratch/project_000000000/test-time-facts/lumi-pytorch-rocm-6.2.3-python-3.12-pytorch-v2.5.1.sif
export HF_TOKEN=""

# Arrays of checkpoint dirs, output paths, and model dirs
CHECKPOINT_DIRS=("")

OUTPUT_DIRS=("")

MODEL=("")

MAX_SHARD_SIZE="10GB"

# Loop over the array indices
for i in "${!CHECKPOINT_DIRS[@]}"; do
    srun singularity exec -B /scratch/project_000000000/cache $CONTAINER \
         bash -c "\
           \$WITH_CONDA; \
           python3 merge_weights.py \
             --model '${MODEL[i]}' \
             --checkpoint_dir '${CHECKPOINT_DIRS[i]}' \
             --output_path '${OUTPUT_DIRS[i]}' \
             --max_shard_size '${MAX_SHARD_SIZE}' \
         " &
done

# Wait for all background jobs to finish
wait

echo "All merges are complete!"
