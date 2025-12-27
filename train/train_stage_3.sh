#!/usr/bin/env bash 

VIDEO_MAX_PIXELS=602112 \
FPS_MAX_FRAMES=12 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift sft --model your_model_path \
        --train_type lora \
        --dataset  ../data/stage3/img_stage3_train.jsonl ../data/stage3/seq_stage3_train.jsonl  \
        --num_train_epochs 1 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --learning_rate 1e-4 \
        --lora_rank 4 \
        --lora_alpha 32 \
        --target_modules all-linear \
        --gradient_accumulation_steps 8 \
        --save_steps 8 \
        --freeze_vit False \
        --save_total_limit 2 \
        --logging_steps 5 \
        --model_author swift \
        --model_name swift-robots \
        --output_dir result \
        --deepspeed zero2 \

    
sleep 2
rm -f batchscript-*