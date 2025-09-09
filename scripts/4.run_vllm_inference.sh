#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --output=/scratch/project_000000000/cache/logs/inference_vllm_%A_%a.out
#SBATCH --error=/scratch/project_000000000/cache/logs/inference_vllm_%A_%a.err
#SBATCH --account=project_000000000
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem-per-gpu=60G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-37

set -e

module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits

CONTAINER=/scratch/project_000000000/test-time-facts/lumi-pytorch-rocm-6.2.3-python-3.12-pytorch-v2.5.1.sif

SCRATCH=/scratch/project_000000000/cache
export TORCH_HOME=$SCRATCH/.torch-cache
export HF_HOME=$SCRATCH/.hf-cache
export XDG_CACHE_HOME="$SCRATCH/.cache"
export XDG_CONFIG_HOME="$SCRATCH/.config"
export XDG_DATA_HOME="$SCRATCH/.local/share"
export TRITON_CACHE_DIR="$SCRATCH/.triton/.triton-cache"
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN="" 

# --- Data and Output Directories ---
DATA_DIR="$SCRATCH/data/evals"
OUTPUT_DIR="$SCRATCH/data/evals"
PROMPT_FILE="$DATA_DIR/_part_1.jsonl"
LOG_DIR="$SCRATCH/logs"
mkdir -p $DATA_DIR $OUTPUT_DIR $LOG_DIR

# --- Model Definitions ---
MODELS=(
    "HuggingFaceTB/SmolLM2-360M-Instruct"
    "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    "Qwen/Qwen2.5-0.5B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-14B-Instruct"
    "Qwen/Qwen2.5-32B-Instruct"
    "Qwen/Qwen2.5-72B-Instruct"
    "Qwen/QwQ-32B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    "jjzha/Qwen2.5-0.5B-Instruct-fs1-2708"
    "jjzha/Qwen2.5-1.5B-Instruct-fs1-2708"
    "jjzha/Qwen2.5-3B-Instruct-fs1-2708"
    "jjzha/Qwen2.5-7B-Instruct-fs1-2708"
    "jjzha/Qwen2.5-14B-Instruct-fs1-2708"
    "jjzha/Qwen2.5-32B-Instruct-fs1-2708"
    "jjzha/Qwen2.5-0.5B-Instruct-rt-2708"
    "jjzha/Qwen2.5-1.5B-Instruct-rt-2708"
    "jjzha/Qwen2.5-3B-Instruct-rt-2708"
    "jjzha/Qwen2.5-7B-Instruct-rt-2708"
    "jjzha/Qwen2.5-14B-Instruct-rt-2708"
    "jjzha/Qwen2.5-32B-Instruct-rt-2708"
    "jjzha/SmolLM2-360M-Instruct-fs1-2708"
    "jjzha/SmolLM2-1.7B-Instruct-fs1-2708"
    "jjzha/SmolLM2-360M-Instruct-rt-2708"
    "jjzha/SmolLM2-1.7B-Instruct-rt-2708"
)

# --- 1. Prepare Prompts (runs only on the first array task) ---
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    echo "Preparing prompts..."
    srun singularity exec -B /scratch/project_000000000/ \
        $CONTAINER \
        bash -c "\$WITH_CONDA; \
        python3 prepare_vllm_batch.py \
            --dataset 'jjzha/fs1-benchmark-2708' \
            --split 'test' \
            --column 'question' \
            --output_file '$PROMPT_FILE'
        "
fi

# Wait for the prompt file to be created by the first task
while [ ! -f "$PROMPT_FILE" ]; do
  echo "Task ${SLURM_ARRAY_TASK_ID} waiting for prompt file..."
  sleep 10
done

# --- 2. Run Batched Inference ---
TASK_ID=$SLURM_ARRAY_TASK_ID
MODEL=${MODELS[$TASK_ID]}

if [ -z "$MODEL" ]; then
    echo "Error: Task ID $TASK_ID is out of bounds for the MODELS array."
    exit 1
fi
safe_model=$(echo "$MODEL" | tr '/' '_')

echo "----------------------------------------"
echo "Starting array task $TASK_ID"
echo "Model: $MODEL"
echo "----------------------------------------"

# --- Dynamic Parameter Configuration ---
TP_SIZE=4; PP_SIZE=1; TEMPERATURE=0.6; TOP_P=0.95;

case "$MODEL" in
    "Qwen/Qwen2.5-0.5B-Instruct")
        TP_SIZE=2; TEMPERATURE=0.7; TOP_P=0.8; TOP_K=20;;
    "HuggingFaceTB/SmolLM2-360M-Instruct")
        TP_SIZE=1; TEMPERATURE=0.7; TOP_P=0.8; TOP_K=20;;
    "HuggingFaceTB/SmolLM2-1.7B-Instruct")
        TP_SIZE=4; TEMPERATURE=0.7; TOP_P=0.8; TOP_K=20;;
    "jjzha/Qwen2.5-0.5B-Instruct-"*)
        TP_SIZE=2; TEMPERATURE=0.6; TOP_P=0.95; TOP_K=40;;
    "Qwen/Qwen2.5-"*)
        TP_SIZE=4; TEMPERATURE=0.7; TOP_P=0.8; TOP_K=20;;
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B")
        TP_SIZE=4; TEMPERATURE=0.6; TOP_P=0.95; TOP_K=40;;
    "Qwen/QwQ-32B")
        TP_SIZE=4; TEMPERATURE=0.6; TOP_P=0.95; TOP_K=40;;
    "jjzha/Qwen2.5-"*)
        TP_SIZE=4; TEMPERATURE=0.6; TOP_P=0.95; TOP_K=40;;
    "jjzha/SmolLM2-360M-Instruct"*)
        TP_SIZE=1; TEMPERATURE=0.6; TOP_P=0.95; TOP_K=40;;
    "jjzha/SmolLM2-1.7B-Instruct"*)
        TP_SIZE=4; TEMPERATURE=0.6; TOP_P=0.95; TOP_K=40;;
esac

echo "Running with configuration:"
echo "  Tensor Parallelism: $TP_SIZE"
echo "  Pipeline Parallelism: $PP_SIZE"
echo "  Temperature: $TEMPERATURE"
echo "  Top P: $TOP_P"

# --- EXECUTE VLLM ---
srun singularity exec -B /scratch/project_000000000/ \
    $CONTAINER \
    bash -c "\$WITH_CONDA; \
    python3 run_vllm_inference.py \
        --model_id '$MODEL' \
        --input_file '$PROMPT_FILE' \
        --output_dir '$OUTPUT_DIR/${safe_model}' \
        --tensor_parallel_size $TP_SIZE \
        --pipeline_parallel_size $PP_SIZE \
        --temperature $TEMPERATURE \
        --top_p $TOP_P \
        --max_model_len 8192 \
        --max_tokens 8000 \
    "

echo "✅ Task $TASK_ID for model $MODEL finished successfully."