#!/usr/bin/env bash

if [[ $# -ne 1 ]]; then
    PORT=8080
else
    PORT=${1}
fi
cd /home/stephen/local/llama.cpp
./build/bin/llama-server -m qwen2.5-0.5b-instruct-q5_k_m.gguf \
    --host 127.0.0.1 \
    --port ${PORT} \
    --n-predict 128 \
    --threads 8 \
    --no-warmup \
    --no-perf \
    >> /home/stephen/src/pris/llamacpp.out \
    2>&1 &
    
