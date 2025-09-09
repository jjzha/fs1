#!/bin/bash -l
#SBATCH --job-name=evals
#SBATCH --output=/scratch/project_000000000/cache/eval_%j.out
#SBATCH --error=/scratch/project_000000000/cache/eval_%j.err
#SBATCH --partition=standard-g
#SBATCH --account=project_000000000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=1:00:00

# ------------------------------
# Setup environment
# ------------------------------
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits

CONTAINER=/scratch/project_000000000/test-time-facts/lumi-pytorch-rocm-6.2.3-python-3.12-pytorch-v2.5.1.sif
SCRATCH=/scratch/project_000000000/cache

export HF_TOKEN=""

# input paths is essentially the output of the inference step.
# benchmark.json is just the original benchmark file.
# output_json is the directory to save the results.

srun singularity exec -B /scratch/project_000000000/ \
    $CONTAINER \
    bash -c "\$WITH_CONDA; \
    python prepare_llm_as_a_judge.py \
        --input_paths $SCRATCH/data/outputs/* \
        --benchmark_json data/test/test.json \
        --output_json $SCRATCH/data/evals/ \
        "

echo "Done."
