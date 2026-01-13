#!/usr/bin/env bash

# Kill any running llama/ollama server process

pids=$(pgrep -f 'ollama serve|llama-server|llama.cpp')

if [[ -n "$pids" ]]; then
    kill $pids
fi

