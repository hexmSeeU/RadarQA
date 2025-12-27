#!/usr/bin/env bash 

VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
swift infer \
    --model /your_model_path \
    --temperature 0 \
    --result_path /your_result_path.jsonl \
    --val_dataset /your_test_data_path.jsonl \

sleep 2
rm -f batchscript-*