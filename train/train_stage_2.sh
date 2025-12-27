#!/usr/bin/env bash 

MAX_PIXELS=602112 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift rlhf \
  --rlhf_type grpo \
  --model your_model_path \
  --output_dir result \
  --train_type lora \
  --dataset ../data/stage2/img_brief_stage2_train.jsonl ../data/stage2/seq_brief_stage2_train.jsonl  \
  --external_plugins ./plugin.py \
  --reward_funcs external_weatherradariqa_v2 format jsonformat \
  --reward_weights 1 0.1 0.1 \
  --torch_dtype bfloat16 \
  --attn_impl flash_attn \
  --num_train_epochs 1 \
  --max_length 8192 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --eval_steps 10 \
  --save_steps 10 \
  --learning_rate 1e-6 \
  --save_total_limit 2 \
  --logging_steps 1 \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 4 \
  --max_completion_length 2048 \
  --num_generations 8 \
  --use_vllm true \
  --vllm_gpu_memory_utilization 0.7 \
  --vllm_max_model_len 8192 \
  --deepspeed zero3 \
  --temperature 1.1 \
  --top_p 1.0 \
  --top_k 80 \
  --log_completions true \
  --num_infer_workers 8 \
  --tensor_parallel_size 4 \
  --async_generate false \
  --offload_optimizer true \
  --offload_model true \
  --gc_collect_after_offload true \
  --move_model_batches 40 \
  --sleep_level 1 \
  --vllm_limit_mm_per_prompt '{"image": 2, "video": 2}' \
  --system ms-swift/examples/train/grpo/prompt.txt