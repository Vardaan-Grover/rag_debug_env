"""
baseline/train_grpo.py
----------------------
GRPO training script for RAGDebugEnv using Qwen2.5-1.5B-Instruct.

STATUS: Stub — structure and wiring are in place. Full implementation
follows once eval_agent.py confirms the reward signal is clean and the
environment behaves correctly across all three tasks.

## Why GRPO (Group Relative Policy Optimization)?

GRPO is the right fit for this environment because:
  - Episodes provide a scalar reward signal at every step (dense)
  - The environment is fast (~1ms/step), enabling many rollouts per batch
  - GRPO does not require a separate value network (unlike PPO), which
    simplifies the training loop
  - Qwen2.5-1.5B-Instruct is small enough to train on a single GPU (8-16GB)
    and has strong instruction-following that makes the action schema learnable

## Training Loop Overview

  for each batch:
    1. Sample G rollouts from the current policy
    2. Each rollout = one episode (reset → steps until done)
    3. Collect (prompt, action, reward) triples
    4. GRPO: normalize rewards within the group, compute policy gradient
    5. Update model weights

## Requirements

    pip install trl>=0.9.0 transformers>=4.45.0 torch peft accelerate

    # The environment server must be running:
    uvicorn rag_debug_env.server.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

# TODO: uncomment when ready to train
# import os
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from trl import GRPOTrainer, GRPOConfig
# from peft import LoraConfig, get_peft_model

from src import RagDebugEnv, RAGDebugAction, RAGDebugObservation

# ---------------------------------------------------------------------------
# Model and training config
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SERVER_URL  = "http://localhost:8000"
TASK_ID     = 1          # Start with task 1 (easiest), progress to 2 and 3
EPISODES_PER_BATCH = 8   # G (group size) for GRPO reward normalization
MAX_EPISODES = 1000      # Total training episodes
LEARNING_RATE = 2e-5
MAX_NEW_TOKENS = 256


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert RAG pipeline debugger. Given an observation of a broken
RAG pipeline, output ONE corrective action as JSON.

Action format: {"action_type": "<type>", "params": {<relevant params>}}

Action types: adjust_chunk_size, adjust_threshold, adjust_top_k,
swap_embedding_model, toggle_reranking, adjust_context_limit,
adjust_chunk_overlap, rewrite_query, submit

Always end with submit when mean_coverage >= 0.70.
"""


def format_prompt(obs: RAGDebugObservation) -> str:
    """Format observation into a model input string."""
    cfg = obs.pipeline_config
    m   = obs.metrics
    return (
        f"Pipeline config: chunk_size={cfg.chunk_size}, "
        f"threshold={cfg.similarity_threshold}, top_k={cfg.top_k}, "
        f"model={cfg.embedding_model.value}, reranking={cfg.use_reranking}\n"
        f"Metrics: coverage={m.mean_coverage:.3f}, precision={m.mean_precision:.3f}, "
        f"empty_retrievals={m.n_empty_retrievals}\n"
        f"Step {obs.steps_taken}/{obs.max_steps}\n"
        f"Action:"
    )


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def collect_rollout(env: RagDebugEnv, model, tokenizer, task_id: int) -> list[dict]:
    """
    Run one episode and collect (prompt, action_text, reward) triples.

    TODO: implement model.generate() call and action parsing.
    """
    # TODO:
    # result = env.reset(task_id=task_id)
    # obs = result.observation
    # rollout = []
    # while not obs.done:
    #     prompt = format_prompt(obs)
    #     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    #     with torch.no_grad():
    #         output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    #     action_text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    #     action = parse_action(action_text)
    #     step_result = env.step(action)
    #     rollout.append({"prompt": prompt, "response": action_text, "reward": step_result.reward or 0.0})
    #     obs = step_result.observation
    # return rollout
    raise NotImplementedError("collect_rollout is not yet implemented — see TODOs above")


def parse_action(action_text: str) -> RAGDebugAction:
    """
    Parse model output text into RAGDebugAction.

    TODO: add robust JSON extraction with fallback to SUBMIT on parse failure.
    """
    import json
    import re
    # Try to extract JSON from the generated text
    match = re.search(r'\{.*\}', action_text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return RAGDebugAction(
                action_type=data.get("action_type", "submit"),
                params=data.get("params", {}),
            )
        except (json.JSONDecodeError, Exception):
            pass
    # Fallback: submit
    return RAGDebugAction(action_type="submit", params={})


# ---------------------------------------------------------------------------
# Training loop (stub)
# ---------------------------------------------------------------------------

def train(server_url: str = SERVER_URL, task_id: int = TASK_ID) -> None:
    """
    Main training loop. Currently a stub.

    TODO: Implement with TRL GRPOTrainer.

    Full implementation steps:
      1. Load model + tokenizer with LoRA adapters
      2. Wrap with GRPOConfig (lr, batch_size, etc.)
      3. Define reward_fn: calls env.step() → returns scalar reward
      4. Instantiate GRPOTrainer(model, config, reward_fn)
      5. trainer.train() — collects rollouts, updates policy via GRPO
      6. Save adapter weights: model.save_pretrained("baseline/checkpoints/grpo_task1")
    """
    print(f"[train_grpo] Model: {MODEL_NAME}")
    print(f"[train_grpo] Task:  {task_id}")
    print(f"[train_grpo] Server: {server_url}")
    print()
    print("This script is a stub. Implement the TODOs above to enable training.")
    print()
    print("Quick start once TODOs are filled in:")
    print("  1. pip install trl transformers peft accelerate")
    print("  2. uvicorn rag_debug_env.server.app:app --host 0.0.0.0 --port 8000 &")
    print("  3. python -m rag_debug_env.baseline.train_grpo --task 1")
    print()
    print("Expected training time on a single A100: ~2h for task 1 to reach 80% success rate.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GRPO training for RAGDebugEnv (stub)")
    parser.add_argument("--task",   type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--server", default=SERVER_URL)
    args = parser.parse_args()
    train(server_url=args.server, task_id=args.task)
