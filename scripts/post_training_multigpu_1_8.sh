#!/bin/sh
#SBATCH --partition=standard-g 
#SBATCH --account=project_00000000 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mem-per-gpu=60G 
#SBATCH --cpus-per-task=56
#SBATCH --time=4:00:00
#SBATCH --job-name=post_training_qwq
#SBATCH --output=/scratch/project_00000000/test-time-facts-cache/training_logs/fine-tune-multigpu-%A_%a.out
#SBATCH --error=/scratch/project_00000000/test-time-facts-cache/training_logs/fine-tune-multigpu_%A_%a.ERROR.out

# Set up the software environment
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits

CONTAINER=/scratch/project_00000000/test-time-facts/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif

SCRATCH=/scratch/project_00000000/test-time-facts-cache
export TORCH_HOME=$SCRATCH/.torch-cache
export HF_HOME=$SCRATCH/.hf-cache
export TOKENIZERS_PARALLELISM=false
mkdir -p $TORCH_HOME $HF_HOME
export HF_TOKEN=""

# List of models
MODELS=()
DATASET=() DATASET=() # make sure it's the tokenized version


micro_batch_size=2 # -> batch_size will be 16 if 16 gpus
gradient_accumulation_steps=1 # requires more GPU memory

for MODEL in "${MODELS[@]}"
do
    echo "Training model: $MODEL"
    # echo "Removing old cache..."
    # rm -rf $HF_HOME/datasets

    srun singularity exec -B /scratch/project_00000000/ \
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
                            --logging_strategy "steps" \
                            --logging_steps 1 \
                            --eval_strategy "steps" \
                            --eval_steps 50 \
                            --save_strategy "steps" \
                            --save_steps 100 \
                            --save_total_limit 2 \
                            --output_dir /scratch/project_00000000/test-time-facts-cache/tmp-0.5B-1.5B-qwen/$(basename $MODEL | tr / -)-rt \
                            "
done
