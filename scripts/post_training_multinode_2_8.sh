#!/bin/bash -l
#SBATCH --job-name=post_training
#SBATCH --output=/scratch/project_00000000/test-time-facts-cache/training_logs/fine-tune-multinode_2_8-%A_%a.out
#SBATCH --error=/scratch/project_00000000/test-time-facts-cache/training_logs/fine-tune-multinode_2_8_%A_%a.ERROR.out
#SBATCH --partition=standard-g 
#SBATCH --account=project_00000000
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=0
#SBATCH --time=12:00:00

# Load required modules
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits

# Set path to your container
CONTAINER=/scratch/project_00000000/test-time-facts/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif

# Setup caching directories
SCRATCH=/scratch/project_00000000/test-time-facts-cache
export TORCH_HOME=$SCRATCH/.torch-cache
export HF_HOME=$SCRATCH/.hf-cache-multinode
mkdir -p $TORCH_HOME $HF_HOME
export TOKENIZERS_PARALLELISM=false

# Set network interfaces for RCCL/NCCL
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=PHB
export HF_TOKEN=""

# Define the models you want to train
MODELS=()
DATASET=() # make sure it's the tokenized version

micro_batch_size=1 # -> batch_size will be 16 if 16 gpus
gradient_accumulation_steps=1 # requires more GPU memory

NUM_PROCESSES=$(expr $SLURM_NNODES \* $SLURM_GPUS_PER_NODE)
MAIN_PROCESS_IP=$(hostname -i)

for MODEL in "${MODELS[@]}"
do
    echo "Training model: $MODEL"
    # echo "Removing old cache..."
    # rm -rf $HF_HOME/datasets

    LAUNCH_CMD="
    accelerate launch \
        --config_file=configs/accelerate_hf_trainer_config_fsdp_2_8.yaml \
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
            --logging_strategy "steps" \
            --logging_steps 1 \
            --eval_strategy "steps" \
            --eval_steps 50 \
            --save_strategy "steps" \
            --save_steps 100 \
            --save_total_limit 1 \
            --output_dir /scratch/project_00000000/test-time-facts-cache/tmp-3-7B/$(basename $MODEL | tr / -)-rt \
        "

    srun singularity exec -B /scratch/project_00000000/ \
        $CONTAINER \
         bash -c "\$WITH_CONDA; \
                export TORCH_HOME=$TORCH_HOME; \
                export HF_HOME=$HF_HOME; \
                $LAUNCH_CMD"

done
