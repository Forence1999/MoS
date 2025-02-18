#!/bin/bash

TIME=$(date "+%Y%m%d-%H%M%S")
echo -e "\nTime: $TIME\n"

struc_dp=false
lora_dropout=0
seed=0
ckpt_dir=$1
LoRA_reduced_rank=$2
GPU_ID=$3

output_dir=./output/guanaco_llama2_7b/${ckpt_dir}_eval_r_${LoRA_reduced_rank}

CUDA_VISIBLE_DEVICES=$GPU_ID \
    python eval_qlora.py \
    --struc_dropout ${struc_dp} \
    --LoRA_reduced_rank ${LoRA_reduced_rank} \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --use_auth_token yes \
    --dataset alpaca \
    --output_dir ${output_dir} \
    --checkpoint_dir ./output/guanaco_llama2_7b/${ckpt_dir} \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 500 \
    --save_total_limit 2 \
    --evaluation_strategy steps \
    --eval_dataset_size 1 \
    --max_eval_samples 1 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 32 \
    --dataloader_num_workers 3 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train false \
    --do_eval \
    --do_mmlu_eval \
    --mmlu_split test \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --source_max_len 384 \
    --target_max_len 128 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --max_steps 10000 \
    --eval_steps 1000 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout $lora_dropout \
    --weight_decay 0.0 \
    --seed $seed
