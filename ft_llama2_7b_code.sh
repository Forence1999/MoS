#!/bin/bash

TIME=$(date "+%Y%m%d-%H%M%S")
echo -e "Time: $TIME"

lora_r=$1
seed=$2
GPU_ID=$3
learning_rate=$4
finetune_mode=$5
init_lora_A_vec_value=$6
init_lora_A_vec_std=$7
valid_param_private_r=$8
valid_param_lora_r=$9
num_chunk_per_vec=${10}

output_dir=./output/LLaMA2_7b_MoS_code_alpaca_${TIME}_GPU_${GPU_ID}_r_${lora_r}_lr_${learning_rate}_sd_${seed}_ft_mode_${finetune_mode}_init_lora_A_vec_value_${init_lora_A_vec_value}_init_lora_A_vec_std_${init_lora_A_vec_std}_valid_param_private_r_${valid_param_private_r}_valid_param_lora_r_${valid_param_lora_r}_num_chunk_${num_chunk_per_vec}

export PYTHONPATH="${PYTHONPATH}:/workspace"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Train QLoRA
echo "------------------- Training QLoRA -------------------"
python finetune_trainer.py \
    --seed $seed \
    --num_chunk_per_vec ${num_chunk_per_vec} \
    --ft_mode ${finetune_mode} \
    --init_lora_A_vec_value ${init_lora_A_vec_value} \
    --init_lora_A_vec_std ${init_lora_A_vec_std} \
    --valid_param_private_r ${valid_param_private_r} \
    --valid_param_lora_r ${valid_param_lora_r} \
    --learning_rate ${learning_rate} \
    --lora_r ${lora_r} \
    --use_lora True \
    --use_qlora True \
    --enable_lora_vec True \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --enable_lora_rotation True \
    --enable_lora_bias False \
    --init2zero_via_vec False \
    --lora_modules all \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --token ${HF_TOKEN} \
    --output_dir ${output_dir} \
    --overwrite_output_dir True \
    --use_flash_attn False \
    --gradient_checkpointing True \
    --torch_dtype bfloat16 \
    --bf16 True \
    --tf32 True \
    --do_train True \
    --train_file data/processed/code_alpaca/code_alpaca_data.jsonl \
    --use_fast_tokenizer False \
    --streaming False \
    --overwrite_cache False \
    --remove_unused_columns True \
    --preprocessing_num_workers 16 \
    --max_seq_length 512 \
    --group_by_length True \
    --optim paged_adamw_32bit \
    --warmup_ratio 0.03 \
    --lr_scheduler_type linear \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --max_steps 10000 \
    --weight_decay 0.0 \
    --max_grad_norm 0.3 \
    --do_eval True \
    --max_eval_samples 1024 \
    --evaluation_strategy steps \
    --prediction_loss_only False \
    --per_device_eval_batch_size 16 \
    --eval_steps 1000 \
    --report_to tensorboard \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 1
# --resume_from_checkpoint None \
# --max_train_samples None \
# --use_auth_token True \
# --adam_beta2 0.999  \
# --max_new_tokens 256  \

# Merge QLoRA
echo "------------------- Merge QLoRA -------------------"
python /workspace/merge_lora.py \
    --base_model_name_or_path meta-llama/Llama-2-7b-hf \
    --lora_model_name_or_path ${output_dir} \
    --output_dir ${output_dir}/lora_merged/ \
    --qlora \
    --save_tokenizer

# Evaluating Tulu 7B model using temperature 0.1 to get the pass@1 score
echo "------------------- Evaluating on Codex-Eval -------------------"
python -m eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl \
    --model_name_or_path ${output_dir}/lora_merged/ \
    --tokenizer_name_or_path ${output_dir}/lora_merged/ \
    --use_slow_tokenizer \
    --save_dir ${output_dir}/codex_eval_results/pass1 \
    --eval_pass_at_ks 1 5 10 20 \
    --temperature 0.1 \
    --unbiased_sampling_size_n 20 \
    --eval_batch_size 4 \
    --load_in_8bit \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

# Evaluating Tulu 7B model using temperature 0.8 to get the pass@10 score
echo "------------------- Evaluating on Codex-Eval -------------------"
python -m eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl \
    --model_name_or_path ${output_dir}/lora_merged/ \
    --tokenizer_name_or_path ${output_dir}/lora_merged/ \
    --use_slow_tokenizer \
    --save_dir ${output_dir}/codex_eval_results/pass10 \
    --eval_pass_at_ks 10 \
    --temperature 0.8 \
    --unbiased_sampling_size_n 20 \
    --eval_batch_size 4 \
    --load_in_8bit \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
