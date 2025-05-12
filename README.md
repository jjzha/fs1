# fs1

This codebase accompanies _Does Longer Reasoning Improve Factuality in
Large Language Models?_.


## Requirements

For the used libraries, see `environments.yml`, we ran everything on Python 3.12 and ROCm 6.2.1.
The most important libraries are as follows:

```
transformers>=4.49.0
vllm>=0.6.3
```

## Reasoning Traces Data

The reasoning traces from both `QwQ-32B` and `DeepSeek-R1-671B` can be found here:

*rt*: https://huggingface.co/datasets/jjzha/rt-tokenized \
*fs1*: https://huggingface.co/datasets/jjzha/fs1-tokenized

## Models

Find the collection of fine-tuned models here: https://huggingface.co/collections/jjzha/fs1-681db4aca59c8721a43353d1

## Model Predictions

Find the collection of model predictions, including baselines such as `o3-mini`, `r1-distill-llama-70b`, and `Qwen2.5-72B-Instruct` here: https://huggingface.co/datasets/AAU-NLP/fs1-predictions

## Experiments

### 0. Tokenizing the data

To tokenize the data, use the script `tokenize_think.py`, to run the script on a slurm cluster, an example is depicted in `scripts/tokenize.sh`

### 1. Training the model

For fine-tuning the model on _rt_ or _fs1_, check out `sft_fs1.py`.
Depending on the number of nodes/GPUs you use, we show a couple of examples in 

```scripts/post_training_multi{gpu,node}_{1,2,4}_8.sh```

We used HuggingFace's Accelerate combined with PyTorch FSDP:

```
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
            --logging_strategy "steps" \
            --logging_steps 1 \
            --eval_strategy "steps" \
            --eval_steps 550 \
            --save_strategy "steps" \
            --save_steps 550 \
            --save_total_limit 1 \
            --output_dir tmp/$(basename $MODEL | tr / -)-fs1 \
        "
```

### 2. Merging the weights

Because of PyTorch FSDP all the weights are sharded. To merge them back again. Use `merge_weights.py`. For an example, check `scripts/merge_weights.sh`.

```
python3 merge_weights.py \
             --model "$MODEL" \
             --checkpoint_dir '${CHECKPOINT_DIRS[i]}' \
             --output_path '${OUTPUT_DIRS[i]}' \
             --max_shard_size '${MAX_SHARD_SIZE}'
```

### 3. Budget Forcing

To do budget forcing, check out `budget_forcing.py` with an example `scripts/budget_forcing.sh`.

```
python budget_forcing.py \
        --model $MODEL \
        --dataset $DATASET \
        --split test \
        --column question \
        --tp-size 8 \
        --budgets $BUDGET \ # e.g., 8192
        --source-filter "ComplexWebQuestions" \
        --max-num-seqs 32 \
        --output "data/budget_forcing_output/budget_force_"$MODEL"_8192.jsonl" \
        # --resume-from-id "" # optional
```

### 4. Parallel Scaling

For parallel scaling, we leveraged `vllm`. Essentially, we expanded the _fs1_ benchmark to have 16 copies of the same question and ran the models over it.

```
python -m vllm.entrypoints.openai.run_batch \
       --model=$MODEL \
       --input "" \
       --output "" \
       --tensor-parallel-size 4 \
       --pipeline-parallel-size 2 \
       --max-model-len 30000 \
       --max-num-seqs 128 \
       --enable-chunked-prefill
```

### 5. Pushing to HF

In `push_to_hub.py` you can find code to push your model to the HF hub.

## Credits

If you have been using our artefacts in your study, please feel free to cite us:

```
coming soon
```
