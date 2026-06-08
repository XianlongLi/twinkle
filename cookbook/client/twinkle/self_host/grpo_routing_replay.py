# Twinkle Client - GRPO Training with Routing Replay for MoE Models
#
# This script demonstrates GRPO training with MoE routing replay using the
# Twinkle client API. Routing replay ensures the training forward pass uses
# the same expert assignments as the rollout (R3) or old-policy pass (R2).
#
# Prerequisites:
#   1. The server must be configured with router_replay_mode:
#      In server_config.yaml, set:
#        router_replay_mode: "R2"  # or "R3" or "disabled"
#   2. For R3 mode, the vLLM sampler on the server must support
#      enable_return_routed_experts (vLLM >= 0.14.0)
#   3. The server must be running before executing this script
#
# Flow:
#   1. Initialize Twinkle client
#   2. Prepare dataset and dataloader
#   3. Configure model with GRPOLoss, optimizer
#   4. Training loop:
#      a. model.save() -> get twinkle_path
#      b. sampler.sample(inputs, adapter_uri=twinkle_path)
#      c. Compute rewards and advantages
#      d. (R2 only) model.forward_only(inputs)  # RECORD routing
#      e. model.forward_backward(inputs, advantages, old_logps)
#      f. model.clip_grad_and_step()
#
# Routing Replay Modes:
#   - disabled: No routing replay (default)
#   - R2: Record routing during forward_only, replay in forward_backward
#   - R3: vLLM returns routed_experts, automatically flows through HTTP to training

import dotenv

dotenv.load_dotenv('.env')

import gc
import os
import re
from peft import LoraConfig
from typing import List, Tuple, Dict, Any

from twinkle import get_logger
from twinkle.reward import GSM8KAccuracyReward
from twinkle.reward.base import Reward
from twinkle.advantage import GRPOAdvantage
from twinkle.dataset import DatasetMeta
from twinkle.metric import CompletionRewardMetric
from twinkle import init_twinkle_client
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset
from twinkle.preprocessor.llm import GSM8KProcessor
from twinkle_client.model import MultiLoraTransformersModel
from twinkle_client.sampler import vLLMSampler

logger = get_logger()

# ========== Configuration ==========
ROUTER_REPLAY_MODE = os.environ.get('ROUTER_REPLAY_MODE', 'disabled')
MODEL_ID = 'ms://Qwen/Qwen3.6-35B-A3B'
NUM_GENERATIONS = 4
MAX_NEW_TOKENS = 1024
LEARNING_RATE = 2e-5
MAX_STEPS = 100
BATCH_SIZE = 2
TEMPERATURE = 1.0
SYNC_INTERVAL = 1
GRADIENT_ACCUMULATION_STEPS = 1
DATA_NUM = 2000

# Validate configuration
if ROUTER_REPLAY_MODE not in ('disabled', 'R2', 'R3'):
    raise ValueError(f'Invalid ROUTER_REPLAY_MODE: {ROUTER_REPLAY_MODE}. '
                     f"Must be one of 'disabled', 'R2', 'R3'")
if ROUTER_REPLAY_MODE != 'disabled':
    logger.info(f'Routing replay mode: {ROUTER_REPLAY_MODE}')
    logger.info('Make sure the server is configured with the same router_replay_mode.')

SYSTEM_PROMPT = ('You are a helpful math assistant. Solve the problem with minimal but correct reasoning '
                 'and put your final answer within \\boxed{}.')


class GSM8KBrevityReward(Reward):
    """Brevity reward: rewards shorter completions that contain a valid answer."""

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for traj in trajectories:
            messages = traj.get('messages', [])
            completion = ''
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    completion = msg.get('content', '')
                    break

            has_answer = bool(
                re.search(r'\\boxed\{[^}]+\}', completion)
                or re.search(r'####\s*[\-\d,\.]+', completion)
            )

            if not has_answer:
                rewards.append(0.0)
            else:
                length = len(completion)
                if length <= 200:
                    rewards.append(1.0)
                else:
                    rewards.append(max(0.0, 1.0 - (length - 200) / 3000))
        return rewards


def create_gsm8k_dataset():
    dataset = Dataset(DatasetMeta('ms://modelscope/gsm8k', subset_name='main', split='train',
                                  data_slice=range(DATA_NUM)))
    dataset.set_template('Qwen3_5Template', model_id=MODEL_ID, max_length=2048, enable_thinking=False)
    dataset.map(GSM8KProcessor(system=SYSTEM_PROMPT))
    dataset.encode(add_generation_prompt=True)
    return dataset


def compute_rewards(
    trajectories: List[Dict[str, Any]],
) -> Tuple[List[float], List[float], List[float]]:
    accuracy_reward_fn = GSM8KAccuracyReward()
    brevity_reward_fn = GSM8KBrevityReward()

    accuracy_rewards = accuracy_reward_fn(trajectories)
    brevity_rewards = brevity_reward_fn(trajectories)
    total_rewards = [a + b for a, b in zip(accuracy_rewards, brevity_rewards)]
    return total_rewards, brevity_rewards, accuracy_rewards


def train():
    # Step 1: Initialize the Twinkle client
    client = init_twinkle_client(
        base_url='http://127.0.0.1:8000',
        api_key='EMPTY_TOKEN',
    )

    # Step 2: Prepare dataset and dataloader
    dataset = create_gsm8k_dataset()
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, num_workers=0)

    # Step 3: Configure the training model
    model = MultiLoraTransformersModel(model_id=MODEL_ID)

    lora_config = LoraConfig(
        target_modules='all-linear',
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
    )
    model.add_adapter_to_model(
        'default',
        lora_config,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    )

    model.set_loss('GRPOLoss', epsilon=0.2, beta=0.0)
    model.set_optimizer('Adam', lr=LEARNING_RATE)
    model.set_processor('InputProcessor')
    model.set_template('Qwen3_5Template', model_id=MODEL_ID)

    # Step 4: Configure the sampler
    sampler = vLLMSampler(model_id=MODEL_ID)
    sampler.set_template('Qwen3_5Template', model_id=MODEL_ID)

    # Step 5: Setup metrics and advantage function
    advantage_fn = GRPOAdvantage()
    metrics = CompletionRewardMetric()

    sampling_params = {
        'max_tokens': MAX_NEW_TOKENS,
        'temperature': TEMPERATURE,
        'top_p': 0.95,
        'num_samples': NUM_GENERATIONS,
        'logprobs': 1,
    }

    current_adapter_uri = None

    step = 0
    for batch in dataloader:
        if step >= MAX_STEPS:
            break

        metrics.reset()
        prompts = batch if isinstance(batch, list) else [batch]

        # ========== 1. Save weights and update adapter_uri ==========
        if step % SYNC_INTERVAL == 0:
            logger.info(f'Step {step}: Saving weights for sampler...')
            result = model.save(
                name='grpo-sampler-weights',
                save_optimizer=False,
                is_sampler=True,
            )
            current_adapter_uri = result.twinkle_path
            logger.info(f'Step {step}: Saved weights to {current_adapter_uri}')

        # ========== 2. Sample completions ==========
        sample_responses = sampler.sample(
            inputs=prompts,
            sampling_params=sampling_params,
            adapter_uri=current_adapter_uri,
        )

        all_input_data: List[Dict[str, Any]] = []
        all_old_logps: List[List[float]] = []
        all_completion_lengths: List[int] = []

        for sample_response in sample_responses:
            for sequence in sample_response.sequences:
                # In R3 mode, sequence.new_input_feature already contains
                # routed_experts data from vLLM, which flows automatically
                # through the HTTP path to the server
                all_input_data.append(sequence.new_input_feature)
                all_old_logps.append([logprob[0][1] for logprob in sequence.logprobs])
                all_completion_lengths.append(len(sequence.tokens))

        # ========== 3. Compute rewards ==========
        total_rewards, brevity_rewards, accuracy_rewards = compute_rewards(all_input_data)
        metrics.accumulate(
            completion_lengths=all_completion_lengths,
            rewards={
                'total': total_rewards,
                'brevity': brevity_rewards,
                'accuracy': accuracy_rewards,
            },
        )

        # ========== 4. Compute advantages ==========
        advantages = advantage_fn(
            total_rewards,
            num_generations=NUM_GENERATIONS,
            scale='group',
        ).tolist()

        frac_zero_std = (1.0 if all(abs(a) < 1e-8 for a in advantages) else 0.0)
        if frac_zero_std == 1.0:
            logger.info(f'Step {step}: All advantages are zero, skipping training')
            step += 1
            continue

        # ========== 5. Training step (GRPO) ==========
        # R2: forward_only RECORD pass → get routing data → inject into inputs
        if ROUTER_REPLAY_MODE == 'R2':
            fwd_result = model.forward_only(inputs=all_input_data)
            router_replay_data = fwd_result.result.get('routed_experts')
            if router_replay_data is not None:
                # NOTE: routing data is returned as a Python list via HTTP.
                # For large sequences this can be expensive; convert to
                # numpy for efficient slicing, then back to list for JSON.
                import numpy as np
                rrd = np.array(router_replay_data)  # [1, total_seq, L, K]
                offset = 0
                for inp in all_input_data:
                    seq_len = inp.get('length') or len(inp['input_ids'])
                    inp['routed_experts'] = rrd[:, offset:offset + seq_len, :, :].tolist()
                    offset += seq_len

        # Training forward_backward — for R3, routed_experts in InputFeature
        # automatically flows through HTTP to the server-side forward_step_func
        model.forward_backward(
            inputs=all_input_data,
            advantages=advantages,
            old_logps=all_old_logps,
        )

        # Gradient clipping and optimizer step
        model.clip_grad_and_step()

        gc.collect()

        # ========== 6. Log ==========
        log_dict = metrics.calculate()
        log_dict.update(model.calculate_metric(is_training=True).result)
        log_dict['train/frac_reward_zero_std'] = frac_zero_std
        logger.info(f'Step {step}: {log_dict}')

        step += 1

    logger.info(f'Training completed. Steps: {step}')
    model.save('grpo-routing-replay-final')


if __name__ == '__main__':
    train()
