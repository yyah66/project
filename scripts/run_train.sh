#!/bin/bash
python3 scripts/train_qwen2vl_lora.py \
  --model-path /root/autodl-tmp/models/qwen/Qwen2.5-VL-7B-Instruct \
  --train-jsonl data/training/chartqa/merged_train.jsonl \
  --val-jsonl data/training/chartqa/merged_val.jsonl \
  --test-jsonl data/training/chartqa/merged_test.jsonl \
  --image-root . \
  --output-dir outputs/lora/merged \
  --batch-size 4 \
  --gradient-accumulation 4 \
  --lora-rank 64 \
  --epochs 3 \
  --resume-from outputs/lora/merged/checkpoint-96
