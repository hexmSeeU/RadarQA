#!/usr/bin/env bash 

NPROC_PER_NODE=1 \
swift export \
    --adapters your_checkpoint_path \
    --stream true \
    --merge_lora true \
    --temperature 0 \

sleep 2
rm -f batchscript-*