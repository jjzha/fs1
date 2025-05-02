#!/bin/bash -l
#SBATCH --job-name=inference_qwq
#SBATCH --output=/scratch/project_00000000/test-time-facts-cache/tokenize_%j.out
#SBATCH --error=/scratch/project_00000000/test-time-facts-cache/tokenize_%j.err
#SBATCH --partition=standard-g
#SBATCH --account=project_00000000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=1:00:00

# ------------------------------
# Setup environment
# ------------------------------
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits

CONTAINER=/scratch/project_00000000/test-time-facts/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif

SCRATCH=/scratch/project_00000000/test-time-facts-cache
export TORCH_HOME=$SCRATCH/.torch-cache
export HF_HOME=$SCRATCH/.hf-cache
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=""

mkdir -p $TORCH_HOME $HF_HOME

echo "Starting tokenization script..."
srun singularity exec -B /scratch/project_00000000/ \
    $CONTAINER \
    bash -c "\$WITH_CONDA; \
    export TORCH_HOME=$TORCH_HOME; \
    export HF_HOME=$HF_HOME; \
    python tokenize_think.py"

echo "Done."
