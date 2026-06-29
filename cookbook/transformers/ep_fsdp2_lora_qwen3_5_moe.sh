#!/bin/sh
set -eu

# EP + FSDP2 + LoRA training for Qwen3.5-MoE.
# All training config passed as CLI flags. Override at invocation, e.g.:
#   sh ep_fsdp2_lora_qwen3_5_moe.sh --batch-size 8 --lr 5e-5

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  torchrun --nproc-per-node=8 \
  cookbook/transformers/ep_fsdp2_lora_qwen3_5_moe.py \
    --model-id ms://Qwen/Qwen3.5-30B-A3B \
    --dataset-id ms://swift/self-cognition \
    --template-cls Qwen3_5Template \
    --dp-size 4 \
    --ep-size 2 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --output-dir ./output_qwen3_5_moe \
    --enable-ep 1 \
    "$@"
