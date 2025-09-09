#!/bin/bash -l
#SBATCH --job-name=post_training_array
#SBATCH --output=/scratch/project_000000000/cache/training_logs/fine-tune-multinode_4_8-%A_%a.out
#SBATCH --error=/scratch/project_000000000/cache/training_logs/fine-tune-multinode_4_8_%A_%a.ERROR.out
#SBATCH --partition=standard-g
#SBATCH --account=project_000000000
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=0
#SBATCH --time=1-00:00:00
#SBATCH --array=0-3

# Load required modules
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits

# Set path to your container
CONTAINER=/scratch/project_000000000/test-time-facts/lumi-pytorch-rocm-6.2.3-python-3.12-pytorch-v2.5.1.sif

# Setup caching directories
SCRATCH=/scratch/project_000000000/cache
export TORCH_HOME=$SCRATCH/.torch-cache
export HF_HOME=$SCRATCH/.hf-cache-multinode
mkdir -p $TORCH_HOME $HF_HOME
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=""

# Set network interfaces for RCCL/NCCL
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=PHB

# Define the models you want to train
MODELS=(
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-14B-Instruct"
    "Qwen/Qwen2.5-32B-Instruct"
)

DATASET=(
    "jjzha/rt-labeled"
)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
echo "Array Task ID: $SLURM_ARRAY_TASK_ID, Training model: $MODEL"

micro_batch_size=1
gradient_accumulation_steps=1

NUM_PROCESSES=$(expr $SLURM_NNODES \* $SLURM_GPUS_PER_NODE)
MAIN_PROCESS_IP=$(hostname -i)

LAUNCH_CMD="
accelerate launch \
    --config_file=configs/accelerate_hf_trainer_config_fsdp_4_8.yaml \
    --num_machines=$SLURM_NNODES \
    --num_processes=$NUM_PROCESSES \
    --machine_rank=\$SLURM_NODEID \
    --main_process_ip=$MAIN_PROCESS_IP \
  sft_fs1.py \
        --model_name_or_path $MODEL \
        --dataset_name $DATASET \
        --bf16 \
        --dataset_text_field 'text' \
        --max_seq_length 8192 \
        --learning_rate 1.0e-5 \
        --num_train_epochs 5 \
        --warmup_ratio 0.05 \
        --weight_decay 0.0001 \
        --per_device_train_batch_size=${micro_batch_size} \
        --per_device_eval_batch_size=${micro_batch_size} \
        --adam_beta1 0.9 \
        --adam_beta2 0.95 \
        --gradient_checkpointing \
        --logging_steps 1 \
        --eval_strategy "no" \
        --save_strategy "epoch" \
        --save_total_limit 1 \
        --output_dir /scratch/project_000000000/cache/$(basename $MODEL | tr / -) \
    "

srun singularity exec -B /scratch/project_000000000/ \
    $CONTAINER \
     bash -c "\$WITH_CONDA; \
            export TORCH_HOME=$TORCH_HOME; \
            export HF_HOME=$HF_HOME; \
            $LAUNCH_CMD"