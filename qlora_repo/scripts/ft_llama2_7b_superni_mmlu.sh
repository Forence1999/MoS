#!/bin/bash

TIME=$(date "+%Y%m%d-%H%M%S")
echo -e "Time: $TIME"

lora_r=$1
reduce_lora_x=$2
seed=$3
GPU_ID=$4
learning_rate=$5

output_dir=./output/superni_${TIME}_GPU_${GPU_ID}_r_${lora_r}_x_${reduce_lora_x}_lr_${learning_rate}_sd_${seed}

export PYTHONPATH="${PYTHONPATH}:/workspace"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Train QLoRA
echo "------------------- Training QLoRA -------------------"
python qlora_repo/qlora_open-instruct.py \
    --seed $seed \
    --learning_rate ${learning_rate} \
    --lora_r ${lora_r} \
    --reduce_lora_A_x ${reduce_lora_x} \
    --reduce_lora_B_x ${reduce_lora_x} \
    --enable_lora_rotation true \
    --enable_lora_vec false \
    --enable_lora_bias false \
    --init2zero_via_vec false \
    --dataset data/processed/super_ni/super_ni_data.jsonl \
    --output_dir ${output_dir} \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --use_auth_token yes \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --max_eval_samples 1024 \
    --per_device_eval_batch_size 16 \
    --full_finetune false \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 2 \
    --dataloader_num_workers 16 \
    --group_by_length true \
    --max_new_tokens 256 \
    --remove_unused_columns true \
    --warmup_ratio 0.03 \
    --lr_scheduler_type linear \
    --gradient_checkpointing \
    --max_seq_length 512 \
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
    --bits 4 \
    --report_to tensorboard

# Merge QLoRA
echo "------------------- Merge QLoRA -------------------"
python open_instruct/merge_lora.py \
    --base_model_name_or_path meta-llama/Llama-2-7b-hf \
    --lora_model_name_or_path ${output_dir} \
    --output_dir ${output_dir}/lora_merged/ \
    --qlora \
    --save_tokenizer

# Evaluating Tulu 7B model using 0 shot and chat format
echo "------------------- Evaluating on MMLU -------------------"
python -m eval.mmlu.run_eval \
    --ntrain 0 \
    --data_dir data/eval/mmlu \
    --save_dir ${output_dir}/mmlu_results \
    --model_name_or_path ${output_dir}/lora_merged/ \
    --tokenizer_name_or_path ${output_dir}/lora_merged/ \
    --eval_batch_size 16 \
    --use_slow_tokenizer \
    --load_in_8bit \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
