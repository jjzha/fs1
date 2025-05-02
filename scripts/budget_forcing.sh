#SBATCH --job-name=budget_force
#SBATCH --output=/scratch/project_00000000/test-time-facts-cache/budget_force_%j.out
#SBATCH --error=/scratch/project_00000000/test-time-facts-cache/budget_force_%j.err
#SBATCH --partition=standard-g
#SBATCH --account=project_00000000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=2-00:00:00

# ------------------------------
# Setup environment
# ------------------------------
module use /appl/local/csc/modulefiles/
module load pytorch/2.5

SCRATCH=/scratch/project_00000000/test-time-facts-cache
export TORCH_HOME=$SCRATCH/.torch-cache
export HF_HOME=$SCRATCH/.hf-cache
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=""

BUDGET=8192
DATASET=""
MODEL=""
safe_model=$(echo "$MODEL" | tr '/' '_')

echo "Starting budget_forcing script..."
python budget_forcing.py \
        --model $MODEL \
        --dataset $DATASET \
        --split test \
        --column question \
        --tp-size 8 \
        --budgets $BUDGET \
        --source-filter "ComplexWebQuestions" \
        --max-num-seqs 32 \
        --output "data/budget_forcing_output/budget_force_"$safe_model"_8192.jsonl" \
        --resume-from-id "ComplexWebQuestions_952"

echo "Done."
