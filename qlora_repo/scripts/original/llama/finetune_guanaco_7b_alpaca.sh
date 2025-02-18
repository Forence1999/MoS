#!/bin/bash

TIME=$(date "+%Y%m%d-%H%M%S")
echo -e "\nTime: $TIME\n"

struc_dp=$1
lora_dropout=$2
seed=$3
GPU_ID=$4

if [ "$struc_dp" = true ]; then
    output_dir=./output/guanaco_llama_7b/alpaca_${TIME}_GPU_${GPU_ID}_dp_${lora_dropout}_sd_${seed}_struc_dp
else
    output_dir=./output/guanaco_llama_7b/alpaca_${TIME}_GPU_${GPU_ID}_dp_${lora_dropout}_sd_${seed}
fi

CUDA_VISIBLE_DEVICES=$GPU_ID \
    python qlora.py \
    --struc_dropout ${struc_dp} \
    --model_name_or_path huggyllama/llama-7b \
    --use_auth_token yes \
    --dataset alpaca \
    --output_dir ${output_dir} \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 500 \
    --save_total_limit 2 \
    --evaluation_strategy steps \
    --eval_dataset_size 1024 \
    --max_eval_samples 1024 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 32 \
    --dataloader_num_workers 3 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --do_mmlu_eval \
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
