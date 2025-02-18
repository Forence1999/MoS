#!/bin/bash

TIME=$(date "+%Y%m%d-%H%M%S")
echo -e "Time: $TIME"

lora_r=$1
reduce_lora_x=$2
seed=$3
GPU_ID=$4
learning_rate=$5

output_dir=./output/gsm_${TIME}_GPU_${GPU_ID}_r_${lora_r}_x_${reduce_lora_x}_lr_${learning_rate}_sd_${seed}

CUDA_VISIBLE_DEVICES=$GPU_ID \
    python qlora_repo/qlora_open-instruct_gsm8k.py \
    --seed $seed \
    --learning_rate ${learning_rate} \
    --lora_r ${lora_r} \
    --reduce_lora_A_x ${reduce_lora_x} \
    --reduce_lora_B_x ${reduce_lora_x} \
    --enable_lora_rotation true \
    --enable_lora_vec false \
    --enable_lora_bias false \
    --init2zero_via_vec false \
    --filter_with_source_max_len true \
    --dataset data/processed/cot/cot_data.jsonl \
    --output_dir ${output_dir} \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --use_auth_token yes \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --data_seed 42 \
    --eval_dataset_size 1024 \
    --max_eval_samples 1024 \
    --per_device_eval_batch_size 1 \
    --do_val_eval \
    --val_dataset gsm8k_cot_8shot \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 2 \
    --dataloader_num_workers 4 \
    --group_by_length \
    --max_new_tokens 32 \
    --remove_unused_columns False \
    --warmup_ratio 0.03 \
    --lr_scheduler_type linear \
    --gradient_checkpointing \
    --source_max_len 384 \
    --target_max_len 128 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --max_steps 10000 \
    --eval_steps 1000 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --weight_decay 0.0 \
    --lora_alpha 16 \
    --lora_modules all \
    --lora_dropout 0.1 \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4
