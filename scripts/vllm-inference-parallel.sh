#!/bin/bash
#SBATCH --job-name=batch_inference
#SBATCH --output=/scratch/project_00000000/test-time-facts-cache/parallel/inference_vllm_%j.out
#SBATCH --error=/scratch/project_00000000/test-time-facts-cache/parallel/inference_vllm_%j.err
#SBATCH --account=project_00000000
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem-per-gpu=60G
#SBATCH --time=2-00:00:00

module use /appl/local/csc/modulefiles/
module load pytorch/2.5

# Where to store the huge models
# For example Deepseek-R1-Distill-Llama-70B requires 132GB
SCRATCH=/scratch/project_00000000/test-time-facts-cache
export TORCH_HOME=$SCRATCH/.torch-cache
export HF_HOME=$SCRATCH/.hf-cache
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=""

# Where to store the vLLM server log
VLLM_LOG=/scratch/project_00000000/test-time-facts-cache/parallel/${SLURM_JOB_ID}.log
mkdir -p $(dirname $VLLM_LOG)


MODEL=""
safe_model=$(echo "$MODEL" | tr '/' '_')

python -m vllm.entrypoints.openai.run_batch \
       --model=$MODEL \
       --input "" \
       --output "" \
       --tensor-parallel-size 4 \
       --pipeline-parallel-size 2 \
       --max-model-len 30000 \
       --max-num-seqs 128 \
       --enable-chunked-prefill


VLLM_PID=$!

echo "Starting vLLM process $VLLM_PID - logs go to $VLLM_LOG"

# Wait until vLLM is running properly
sleep 60
while ! curl localhost:8000 >/dev/null 2>&1
do
    # catch if vllm has crashed
    if [ -z "$(ps --pid $VLLM_PID --no-headers)" ]; then
        exit
    fi
    sleep 10
done

# To stop job after we have run what we want kill it
kill $VLLM_PID