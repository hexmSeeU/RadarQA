#!/bin/bash

max_retries=5
retry_count=0

while true; do
    echo "正在运行 generate.py (第 $((retry_count + 1)) 次尝试)"
    python generate.py --model_name gpt-5 \

    if [ $? -eq 0 ]; then
        echo "generate.py 成功运行，任务完成！"
        exit 0
    else
        retry_count=$((retry_count + 1))
        if [ $retry_count -ge $max_retries ]; then
            echo "达到最大重试次数，任务失败！"
            exit 1
        fi
        echo "generate.py 运行失败，将在 5 秒后重试..."
        sleep 5     
    fi
done
