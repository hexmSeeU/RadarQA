#!/bin/bash

# 设置重试次数（可以根据需要调整）
max_retries=5
retry_count=0

while true; do
    echo "正在运行 gpt4_score.py (第 $((retry_count + 1)) 次尝试)"
    python gpt4_score.py --folder /your/path/to/model_result_folder --type image

    if [ $? -eq 0 ]; then
        echo "gpt4_score.py 成功运行，任务完成！"
        exit 0
    else
        retry_count=$((retry_count + 1))
        if [ $retry_count -ge $max_retries ]; then
            echo "达到最大重试次数，任务失败！"
            exit 1
        fi
        echo "gpt4_score.py 运行失败，将在 5 秒后重试..."
        sleep 5  
    fi
done