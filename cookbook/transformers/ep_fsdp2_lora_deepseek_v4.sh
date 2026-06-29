#!/bin/sh
set -eu

# EP + FSDP2 + LoRA training for DeepSeek-V4.
# ENABLE_EP=1 trains expert LoRA with target_parameters.
# ENABLE_EP=0 runs plain FSDP2 LoRA and does not train expert parameters.
# All training config passed as CLI flags. Override at invocation, e.g.:
#   sh ep_fsdp2_lora_deepseek_v4.sh --batch-size 8 --lr 5e-5

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  torchrun --nproc-per-node=8 \
  cookbook/transformers/ep_fsdp2_lora_deepseek_v4.py \
    --model-id ms://deepseek-ai/DeepSeek-V3-0324 \
    --dataset-id ms://swift/self-cognition \
    --dp-size 4 \
    --ep-size 2 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --output-dir ./output_dsv4 \
    --enable-ep 1 \
    "$@"
