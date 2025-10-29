#!/bin/bash
set -e

# 设置默认值
MODEL_PATH=${MODEL_PATH:-"/home/caiyueliang/models/univla-iros-manipulation-challenge-baseline"}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8080}

# 打印配置信息（可选）
echo "Starting Python service with the following configuration:"
echo "Model Path: $MODEL_PATH"
echo "Host: $HOST"
echo "Port: $PORT"

# 启动 Python 服务（假设入口脚本是 app.py，你可以替换成自己的启动命令）
python3 scripts/infer_server_1.py \
    --host ${HOST} \
    --port ${PORT} \
    --pretrained_checkpoint ${MODEL_PATH} \
    --action_decoder_path ${MODEL_PATH}"/action_decoder.pt"