# Copyright (c) ModelScope Contributors. All rights reserved.
"""Helpers for Megatron MoE router replay.

Provides utilities for recording and replaying MoE routing decisions during
GRPO training, ensuring the training forward pass uses the same expert
assignments as the rollout (R3) or old-policy forward (R2) pass.

Two modes:
- R2 (Record): Record routing during a forward-only RECORD pass, then replay
  during the training forward_backward pass. No vLLM changes required.
- R3 (Replay from rollout): vLLM (>= 0.14.0) records routing decisions during
  generation. Training directly replays these via ``routed_experts`` data.
"""

from __future__ import annotations

from typing import Optional

import torch

from twinkle import Platform
from twinkle.utils import get_logger
from twinkle.utils.torch_utils import split_cp_inputs

logger = get_logger()

try:
    from megatron.core import mpu
    from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region
    from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction
    from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
    from megatron.core.transformer.transformer_block import get_num_layers_to_build
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    ROUTER_REPLAY_AVAILABLE = True
except ImportError:
    mpu = None
    scatter_to_sequence_parallel_region = None
    RouterReplay = None
    RouterReplayAction = None
    MoEAlltoAllTokenDispatcher = None
    get_num_layers_to_build = None
    get_transformer_layer_offset = None
    ROUTER_REPLAY_AVAILABLE = False


def is_moe_layer(tf_config, layer_idx: int) -> bool:
    """Check whether a transformer layer at *layer_idx* is an MoE layer."""
    moe_layer_freq = getattr(tf_config, 'moe_layer_freq', None)
    if moe_layer_freq is None:
        return True
    if isinstance(moe_layer_freq, int):
        if moe_layer_freq <= 0:
            return False
        return layer_idx % moe_layer_freq == 0
    if isinstance(moe_layer_freq, list):
        return moe_layer_freq[layer_idx] == 1
    raise ValueError(f'Unsupported moe_layer_freq type: {type(moe_layer_freq)}')


def get_moe_num_layers_to_build(tf_config, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None) -> int:
    """Count MoE layers within the given VP stage or PP rank."""
    total_layers = get_num_layers_to_build(tf_config, vp_stage=vp_stage, pp_rank=pp_rank)
    layer_offset = get_transformer_layer_offset(tf_config, vp_stage=vp_stage)
    return sum(1 for idx in range(layer_offset, layer_offset + total_layers) if is_moe_layer(tf_config, idx))


def get_local_layer_range(tf_config, vp_rank: Optional[int] = None, only_moe_layer: bool = True) -> tuple[int, int]:
    """Return ``(offset, count)`` of local RouterReplay instances for *vp_rank*.

    *offset* is the start index into ``RouterReplay.global_router_replay_instances``;
    *count* is the number of MoE layers owned by this VP stage.
    """
    vp_size = getattr(tf_config, 'virtual_pipeline_model_parallel_size', None)
    if vp_size is not None:
        if vp_rank is None and mpu is not None:
            try:
                vp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
            except Exception:
                vp_rank = 0
        vp_rank = 0 if vp_rank is None else vp_rank
        offset = 0
        for prev_stage in range(vp_rank):
            if only_moe_layer:
                offset += get_moe_num_layers_to_build(tf_config, prev_stage)
            else:
                offset += get_num_layers_to_build(tf_config, vp_stage=prev_stage)
    else:
        offset = 0
    if only_moe_layer:
        count = get_moe_num_layers_to_build(tf_config, vp_rank)
    else:
        count = get_num_layers_to_build(tf_config, vp_stage=vp_rank)
    return offset, count


def get_local_topk_idx_for_current_rank(
    global_topk_idx,
    tf_config,
    packed_seq_params=None,
    vp_rank: Optional[int] = None,
):
    """Slice *global_topk_idx* ``[B, S, L_global, K]`` for the current CP/TP rank.

    vLLM produces *routed_experts* across all transformer layers. This function
    filters to MoE-only layers and slices along the sequence (CP) and expert (TP)
    dimensions so every rank receives only the data it needs.
    """
    if global_topk_idx is None:
        return None
    if not ROUTER_REPLAY_AVAILABLE:
        raise RuntimeError('Router replay is unavailable in the current megatron-core installation.')

    if global_topk_idx.dim() != 4:
        raise ValueError(
            f'Expected routed_experts with shape [B, S, L, K], got {tuple(global_topk_idx.shape)}')

    # Use vp_stage=0 as the base offset because vLLM reports routed_experts
    # across all transformer layers (including dense layers in hybrid MoE models).
    # We filter to MoE-only layers for the full model (all VP stages).
    layer_offset = get_transformer_layer_offset(tf_config, vp_stage=0)
    offset, count = get_local_layer_range(
        tf_config, getattr(tf_config, 'virtual_pipeline_model_parallel_size', None), only_moe_layer=False)
    num_layers = offset + count
    moe_layer_idx = torch.tensor(
        [idx for idx in range(layer_offset, layer_offset + num_layers) if is_moe_layer(tf_config, idx)],
        dtype=torch.long,
        device=global_topk_idx.device,
    )
    local_topk_idx = torch.index_select(global_topk_idx, dim=2, index=moe_layer_idx)

    # CP slicing along the sequence dimension (no-op when CP=1)
    local_topk_idx = split_cp_inputs(
        local_topk_idx, getattr(packed_seq_params, 'cu_seqlens_q', None), 1)

    # TP slicing along the expert dimension (no-op when TP=1 or SP disabled)
    local_topk_idx = scatter_to_sequence_parallel_region(
        local_topk_idx.transpose(0, 1)).transpose(0, 1)

    return local_topk_idx


def set_router_replay_data(layers_topk_idx, tf_config, vp_rank: Optional[int] = None) -> None:
    """Inject recorded routing indices into local RouterReplay instances.

    *layers_topk_idx* is expected to contain indices for ALL global MoE layers
    (output of :func:`get_local_topk_idx_for_current_rank`). The function slices
    the local VP stage portion and calls ``set_target_indices`` on each router.
    """
    if layers_topk_idx is None:
        return
    if not ROUTER_REPLAY_AVAILABLE:
        raise RuntimeError('Router replay is unavailable in the current megatron-core installation.')

    layers_topk_idx = layers_topk_idx.to(Platform.get_local_device())
    layers_topk_idx = layers_topk_idx.flatten(0, 1).transpose(0, 1)
    offset, count = get_local_layer_range(tf_config, vp_rank)
    router_instances = RouterReplay.global_router_replay_instances[offset:offset + count]
    for idx, router in enumerate(router_instances):
        router.set_target_indices(layers_topk_idx[idx + offset].to(torch.int64))


def get_router_replay_data(tf_config, vp_rank: Optional[int] = None):
    """Collect recorded routing top-k indices from local router instances (R2 mode).

    Returns a tensor of shape ``[1, seq_len, local_layer_num, topk]`` containing
    the recorded routing decisions, or *None* if no routers are available.
    """
    router_instances = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
    if not router_instances:
        return None
    layers_topk_idx = []
    for router in router_instances:
        layers_topk_idx.append(router.recorded_topk_idx.to(torch.uint8))
    layers_topk_idx = torch.stack(layers_topk_idx).transpose(0, 1).unsqueeze(0).to(Platform.get_local_device())
    return layers_topk_idx


def gather_router_replay_data(tf_config, vp_rank: Optional[int] = None):
    """Gather recorded routing data across CP, TP/SP, and PP dimensions (R2 mode).

    Collects local routing data from router instances then performs collective
    all-gather operations to reconstruct the full ``[1, total_seq, all_layers, topk]``
    tensor. Only the PP-last + CP-0 rank returns the result; other ranks return None.
    """
    local_data = get_router_replay_data(tf_config, vp_rank)
    if local_data is None:
        return None

    # CP all-gather along sequence dim (dim=1)
    cp_size = mpu.get_context_parallel_world_size() if mpu.is_initialized() else 1
    if cp_size > 1:
        cp_group = mpu.get_context_parallel_group()
        shapes = [torch.zeros(1, dtype=torch.long, device=local_data.device) for _ in range(cp_size)]
        local_seq = torch.tensor([local_data.shape[1]], dtype=torch.long, device=local_data.device)
        torch.distributed.all_gather(shapes, local_seq, group=cp_group)
        max_seq = max(s.item() for s in shapes)
        B, _, L, K = local_data.shape
        padded = torch.zeros(B, max_seq, L, K, dtype=local_data.dtype, device=local_data.device)
        padded[:, :local_data.shape[1], :, :] = local_data
        gathered = [torch.zeros_like(padded) for _ in range(cp_size)]
        torch.distributed.all_gather(gathered, padded, group=cp_group)
        local_data = torch.cat([g[:, :s.item(), :, :] for g, s in zip(gathered, shapes)], dim=1)

    # TP/SP all-gather along sequence dim (dim=1)
    tp_size = mpu.get_tensor_model_parallel_world_size() if mpu.is_initialized() else 1
    if tp_size > 1:
        try:
            from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
            local_data = gather_from_sequence_parallel_region(local_data.transpose(0, 1)).transpose(0, 1)
        except ImportError:
            pass

    # PP all-gather along layer dim (dim=2)
    pp_size = mpu.get_pipeline_model_parallel_world_size() if mpu.is_initialized() else 1
    if pp_size > 1:
        pp_group = mpu.get_pipeline_model_parallel_group()
        local_layers = torch.tensor([local_data.shape[2]], dtype=torch.long, device=local_data.device)
        all_layers = [torch.zeros(1, dtype=torch.long, device=local_data.device) for _ in range(pp_size)]
        torch.distributed.all_gather(all_layers, local_layers, group=pp_group)
        max_layers = max(l.item() for l in all_layers)
        B, S, _, K = local_data.shape
        padded = torch.zeros(B, S, max_layers, K, dtype=local_data.dtype, device=local_data.device)
        padded[:, :, :local_data.shape[2], :] = local_data
        gathered = [torch.zeros_like(padded) for _ in range(pp_size)]
        torch.distributed.all_gather(gathered, padded, group=pp_group)
        local_data = torch.cat([g[:, :, :l.item(), :] for g, l in zip(gathered, all_layers)], dim=2)

    # Only the PP-last + CP-0 rank returns the full tensor
    pp_last = (pp_size <= 1 or mpu.get_pipeline_model_parallel_rank() == pp_size - 1)
    cp0 = (cp_size <= 1 or mpu.get_context_parallel_rank() == 0)
    if pp_last and cp0:
        return local_data
    return None


def apply_router_replay_patch() -> None:
    """Monkey-patch ``MoEAlltoAllTokenDispatcher.preprocess`` for routing replay.

    When routing replay is active, the dispatcher must compute ``num_out_tokens``
    from the actual routing map instead of using capacity-based heuristics.
    """
    if not ROUTER_REPLAY_AVAILABLE:
        raise RuntimeError('Router replay requires megatron-core with router replay support.')

    if MoEAlltoAllTokenDispatcher is None or hasattr(MoEAlltoAllTokenDispatcher,
                                                     '_twinkle_router_replay_patched'):
        return

    original_preprocess = MoEAlltoAllTokenDispatcher.preprocess

    def patched_preprocess(self, routing_map):
        result = original_preprocess(self, routing_map)
        if (getattr(self.config, 'moe_enable_routing_replay', False) and not self.drop_and_pad
                and self.config.moe_expert_capacity_factor is None
                and not (getattr(self.config, 'moe_router_padding_for_quantization', None)
                         or getattr(self.config, 'moe_router_padding_for_fp8', None))):
            self.num_out_tokens = int(routing_map.sum().item())
        return result

    MoEAlltoAllTokenDispatcher.preprocess = patched_preprocess
    MoEAlltoAllTokenDispatcher._twinkle_router_replay_patched = True
    logger.info('Applied Twinkle router replay patch to MoEAlltoAllTokenDispatcher.')


class RouterReplayHelper:
    """Helper for querying router replay state and locating local instances."""

    @staticmethod
    def get_micro_batch_router_list(tf_config, vp_rank: Optional[int] = None):
        offset, count = get_local_layer_range(tf_config, vp_rank)
        return RouterReplay.global_router_replay_instances[offset:offset + count]

    @staticmethod
    def set_micro_batch_action(tf_config, action, vp_rank: Optional[int] = None) -> None:
        for router in RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank):
            router.set_router_replay_action(action)

    @staticmethod
    def is_r2_record_action(tf_config, vp_rank: Optional[int] = None) -> bool:
        router_instances = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        return bool(router_instances
                    and router_instances[0].router_replay_action == RouterReplayAction.RECORD)

    @staticmethod
    def is_replay_forward_action(tf_config, vp_rank: Optional[int] = None) -> bool:
        router_instances = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        return bool(router_instances
                    and router_instances[0].router_replay_action == RouterReplayAction.REPLAY_FORWARD)

    @staticmethod
    def is_replay_backward_action(tf_config, vp_rank: Optional[int] = None) -> bool:
        router_instances = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        return bool(router_instances
                    and router_instances[0].router_replay_action == RouterReplayAction.REPLAY_BACKWARD)
