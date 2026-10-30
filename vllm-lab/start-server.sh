#!/usr/bin/env bash
set -euo pipefail

cd /home/vllm-lab/lab2
exec .venv/bin/vllm serve Qwen/Qwen2.5-1.5B-Instruct-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.80 \
  --max-num-seqs 1 \
  --enforce-eager
