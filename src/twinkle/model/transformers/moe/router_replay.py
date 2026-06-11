# Copyright (c) ModelScope Contributors. All rights reserved.
"""FSDP routing replay utilities for HF Transformers MoE models.

Provides RouterReplayAction, per-MoE-block replay state, and functions for
recording / replaying expert routing decisions during GRPO training.

Naming conventions follow the Megatron ``router_replay.py`` module
so the two backends expose a consistent API surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from twinkle.utils import get_logger

logger = get_logger()


class RouterReplayAction(Enum):
    """Mirrors ``megatron.core.transformer.moe.router_replay.RouterReplayAction``."""

    DISABLED = 0
    RECORD = 1          # R2: record selected_experts during forward
    REPLAY_FORWARD = 2  # R2 / R3: use pre-recorded selected_experts


@dataclass
class _RouterReplayState:
    """Per-MoE-block replay state, stored in the global ``_registry``."""

    action: RouterReplayAction = RouterReplayAction.DISABLED
    recorded_indices: Optional[torch.Tensor] = None  # [num_tokens, topk]
    target_indices: Optional[torch.Tensor] = None    # [num_tokens, topk]


# ---------------------------------------------------------------------------
# Global registry: {block_name: _RouterReplayState}
# ---------------------------------------------------------------------------
_registry: Dict[str, _RouterReplayState] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_moe_blocks(model: nn.Module) -> None:
    """Walk *model*, find every MoE block and register it in ``_registry``.

    Safe to call multiple times — already-registered blocks are skipped.
    """
    from .expert_parallel import find_moe_blocks_with_names

    for name, _ in find_moe_blocks_with_names(model):
        if name not in _registry:
            _registry[name] = _RouterReplayState()


def set_global_router_replay_action(action: RouterReplayAction) -> None:
    """Set *action* on every registered MoE block."""
    for state in _registry.values():
        state.action = action


def clear_global_router_replay_action() -> None:
    """Reset action to DISABLED on every registered MoE block."""
    for state in _registry.values():
        state.action = RouterReplayAction.DISABLED


def clear_global_indices() -> None:
    """Clear recorded / target indices on every registered MoE block."""
    for state in _registry.values():
        state.recorded_indices = None
        state.target_indices = None


def get_replay_state(block_name: str) -> Optional[_RouterReplayState]:
    """Return the replay state for *block_name*, or *None*."""
    return _registry.get(block_name)


def set_router_replay_data(
    batch_routed_experts: torch.Tensor,
    model: nn.Module,
) -> None:
    """Slice *batch_routed_experts* ``[1, total_seq, L, K]`` into per-block
    ``target_indices`` and inject them into the registered MoE blocks of *model*.

    Each block receives a ``[num_tokens, topk]`` slice covering the tokens
    processed by that layer.
    """
    if batch_routed_experts is None:
        return

    from .expert_parallel import find_moe_blocks_with_names

    blocks = list(find_moe_blocks_with_names(model))
    if not blocks:
        return

    # batch_routed_experts: [1, total_seq, num_moe_layers, topk]
    # Squeeze batch dim: [total_seq, num_moe_layers, topk]
    routed = batch_routed_experts.squeeze(0)
    if routed.dim() != 3:
        raise ValueError(
            f'Expected routed_experts with shape [1, seq, layers, topk], '
            f'got {tuple(batch_routed_experts.shape)}'
        )

    num_tokens = routed.shape[0]
    num_layers_in_data = routed.shape[1]

    for layer_idx, (name, _) in enumerate(blocks):
        state = _registry.get(name)
        if state is None:
            continue
        if layer_idx >= num_layers_in_data:
            break

        # Each layer gets [num_tokens, topk]
        target = routed[:, layer_idx, :].to(torch.int64)
        if target.numel() > 0:
            state.target_indices = target


def get_router_replay_data(
    model: nn.Module,
    ep_group: Optional[dist.ProcessGroup] = None,
) -> Optional[torch.Tensor]:
    """Collect ``recorded_indices`` from all registered MoE blocks in *model*.

    When *ep_group* is provided (EP > 1) the local routing data is all-gathered
    across EP ranks along the sequence dimension so the caller receives the
    full ``[1, total_seq, num_layers, topk]`` tensor.

    Returns *None* when no MoE blocks have recorded routing data.
    """
    from .expert_parallel import find_moe_blocks_with_names

    blocks = list(find_moe_blocks_with_names(model))
    layers = []
    for name, _ in blocks:
        state = _registry.get(name)
        if state is not None and state.recorded_indices is not None:
            layers.append(state.recorded_indices)  # each: [num_tokens, topk]

    if not layers:
        return None

    # Stack: [num_tokens, num_layers, topk] -> [1, num_tokens, num_layers, topk]
    local_data = torch.stack(layers, dim=1).unsqueeze(0)

    # EP all-gather along the sequence dimension (dim=1)
    if ep_group is not None and ep_group.size() > 1:
        local_len = torch.tensor(
            [local_data.shape[1]], dtype=torch.long, device=local_data.device
        )
        ep_world = ep_group.size()
        all_lens = [torch.zeros(1, dtype=torch.long, device=local_data.device) for _ in range(ep_world)]
        dist.all_gather(all_lens, local_len, group=ep_group)
        max_len = max(l.item() for l in all_lens)

        B, _, L, K = local_data.shape
        padded = torch.zeros(B, max_len, L, K, dtype=local_data.dtype, device=local_data.device)
        padded[:, :local_data.shape[1], :, :] = local_data

        gathered = [torch.zeros_like(padded) for _ in range(ep_world)]
        dist.all_gather(gathered, padded, group=ep_group)

        local_data = torch.cat(
            [g[:, :l.item(), :, :] for g, l in zip(gathered, all_lens)], dim=1
        )

    return local_data  # [1, total_seq, num_layers, topk]


def apply_router_replay_patch(model: nn.Module) -> None:
    """Register MoE blocks and (for EP=1) wrap their forwards through
    ``_run_router()`` so that routing replay works on the HF native path.

    When EP > 1, ``expert_parallel.py`` already patches the forward through
    ``_run_router()`` — only registration is needed.
    """
    from .expert_parallel import find_moe_blocks_with_names

    register_moe_blocks(model)

    # Determine if EP patches are already in place.
    # When EP > 1 the block forward has already been replaced by
    # patch_forward(); we only need to ensure replay_state is wired in.
    # When EP = 1 the original HF forward is intact — wrap it.
    blocks = list(find_moe_blocks_with_names(model))
    if not blocks:
        return

    # Check whether the first block's forward has already been EP-patched
    _first_name, first_block = blocks[0]
    if _is_ep_patched(first_block):
        logger.debug(
            'EP patches detected — routing replay piggy-backs on '
            'patch_forward() replay_state wiring.'
        )
        return

    # EP = 1: wrap each MoE block forward through _run_router()
    logger.info('Applying FSDP routing replay patch (EP=1 mode).')
    _wrap_all_moe_blocks(model, blocks)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_ep_patched(block: nn.Module) -> bool:
    """Return True if *block* has already been patched by expert_parallel."""
    return getattr(block, '_ep_patched', False)


def _wrap_all_moe_blocks(
    model: nn.Module,
    blocks: List[tuple[str, nn.Module]],
) -> None:
    """Replace each MoE block's forward with a wrapper that calls
    ``_run_router()`` instead of the original gate logic."""
    from .expert_parallel import _get_gate, _get_router_dtype, _run_router

    for name, block in blocks:
        gate = _get_gate(block)
        if gate is None:
            continue
        top_k = getattr(gate, 'top_k', 2)
        norm_topk_prob = getattr(
            block, 'norm_topk_prob',
            getattr(gate, 'norm_topk_prob', False),
        )

        original_forward = block.forward

        def _make_patched_forward(
            _original_forward,
            _block,
            _name,
            _gate,
            _top_k,
            _norm_topk_prob,
        ):
            def patched_forward(self, hidden_states):
                batch_size, sequence_length, hidden_dim = hidden_states.shape
                hidden_states_2d = hidden_states.view(-1, hidden_dim)

                replay_state = _registry.get(_name)
                _router_logits, routing_weights, selected_experts = _run_router(
                    gate=_gate,
                    hidden_states=hidden_states_2d,
                    top_k=_top_k,
                    router_dtype=_get_router_dtype(None, hidden_states_2d.dtype),
                    norm_topk_prob=_norm_topk_prob,
                    replay_state=replay_state,
                )
                routed_output = self.experts(
                    hidden_states_2d, selected_experts, routing_weights
                )
                return routed_output.reshape(batch_size, sequence_length, hidden_dim)

            return patched_forward

        routed_fn = _make_patched_forward(
            original_forward, block, name, gate, top_k, norm_topk_prob
        )

        # Wrap shared-expert logic when present
        if (
            hasattr(block, 'shared_expert')
            and hasattr(block, 'shared_expert_gate')
        ):
            def _full_forward(self, hidden_states):
                final = routed_fn(self, hidden_states)
                gate_val = self.shared_expert_gate(hidden_states)
                if isinstance(gate_val, torch.Tensor) and gate_val.dim() > 0:
                    gate_val = gate_val.sigmoid()
                shared = self.shared_expert(gate_val.mul(hidden_states)
                                            if gate_val.dim() <= hidden_states.dim()
                                            else hidden_states)
                return final + shared
            block.forward = _full_forward.__get__(block)
        else:
            block.forward = routed_fn.__get__(block)
