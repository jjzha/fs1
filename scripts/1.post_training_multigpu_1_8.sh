#!/bin/sh
#SBATCH --partition=standard-g 
#SBATCH --account=project_000000000 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mem-per-gpu=60G 
#SBATCH --cpus-per-task=56
#SBATCH --time=4:00:00
#SBATCH --job-name=post_training
#SBATCH --output=/scratch/project_000000000/cache/training_logs/fine-tune-multigpu-%A_%a.out
#SBATCH --error=/scratch/project_000000000/cache/training_logs/fine-tune-multigpu_%A_%a.ERROR.out

# Set up the software environment
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits

CONTAINER=/scratch/project_000000000/test-time-facts/lumi-pytorch-rocm-6.2.3-python-3.12-pytorch-v2.5.1.sif

SCRATCH=/scratch/project_000000000/cache
export TORCH_HOME=$SCRATCH/.torch-cache
export HF_HOME=$SCRATCH/.hf-cache
export TOKENIZERS_PARALLELISM=false
mkdir -p $TORCH_HOME $HF_HOME
export HF_TOKEN=""

# List of models
MODELS=(
    "HuggingFaceTB/SmolLM2-360M-Instruct"
    "HuggingFaceTB/SmolLM2-1.7B-Instruct"
)

DATASET=(
    "jjzha/fs1-labeled"
)

micro_batch_size=2 # -> batch_size will be 16 if 16 gpus
gradient_accumulation_steps=1 # requires more GPU memory

for MODEL in "${MODELS[@]}"
do
    echo "Training model: $MODEL"
    # echo "Removing old cache..."
    # rm -rf $HF_HOME/datasets

    srun singularity exec -B /scratch/project_000000000/ \
         $CONTAINER \
            bash -c "\$WITH_CONDA; \
                     export TORCH_HOME=$TORCH_HOME; \
                     export HF_HOME=$HF_HOME; \
                     accelerate launch \
                        --config_file=configs/accelerate_hf_trainer_config.yaml \
                        --num_machines=1 \
                        --num_processes=${SLURM_GPUS_PER_NODE} \
                        --machine_rank=0 \
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
done
