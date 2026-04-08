"""
outputs/train_grpo.py
---------------------
GRPO-style training scaffold for RAGDebugEnv.

Implements Group Relative Policy Optimization (GRPO) using the OpenAI API
as the policy (no local gradient descent required). This is a zero-shot
RL evaluation loop that demonstrates the full GRPO training data structure:

  - Collect G rollouts per batch from the same task/seed (the "group").
  - Compute GRPO-normalized rewards: r_norm = (r - mean(group)) / (std(group) + eps)
  - Log each (prompt, response, reward, normalized_reward) tuple — exactly the
    format needed to fine-tune a model with TRL's GRPOTrainer.
  - Saves training data to outputs/grpo_data.jsonl for offline fine-tuning.

## Why GRPO?

GRPO is the right fit for this environment because:
  - Episodes provide a dense scalar reward at every step (not just terminal).
  - The environment is fast (~1ms/step), enabling many rollouts per batch.
  - GRPO requires no value network — it normalizes rewards within the group.
  - The resulting training data can fine-tune a small model (Qwen2.5-1.5B)
    to learn debugging strategies through reinforcement learning.

## Usage

    # Server must be running:
    uvicorn server.app:app --host 0.0.0.0 --port 7860

    # Run 3 batches of 4 rollouts on task 1:
    python outputs/train_grpo.py --task 1 --batches 3 --group-size 4

    # Run on all tasks:
    python outputs/train_grpo.py --task all --batches 2 --group-size 4

    # With a different server URL:
    python outputs/train_grpo.py --server http://localhost:7860 --task 2

## Training Data Format (outputs/grpo_data.jsonl)

Each line is a JSON object:
    {
        "task_id": int,
        "batch": int,
        "rollout": int,
        "episode_seed": int,
        "steps": [{"prompt": str, "response": str, "reward": float}],
        "total_reward": float,
        "normalized_reward": float,  # GRPO baseline subtracted
        "success": bool
    }

## Upgrading to Full Fine-Tuning

To run actual gradient descent with TRL:
    pip install trl>=0.9.0 transformers>=4.45.0 torch peft accelerate

Then adapt the data collected here with TRL's GRPOTrainer:
    from trl import GRPOTrainer, GRPOConfig
    trainer = GRPOTrainer(
        model=model,
        config=GRPOConfig(learning_rate=2e-5, ...),
        reward_fn=lambda prompts, responses: [r["reward"] for r in responses],
    )
    trainer.train()
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import textwrap
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from client import RAGDebugEnv
from models import RAGDebugAction, RAGDebugObservation, ActionType

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:7860")
DATA_PATH = Path(__file__).parent / "grpo_data.jsonl"

MAX_STEPS = 10
TEMPERATURE = 0.7       # Higher temperature for diverse rollouts (exploration)
MAX_TOKENS = 512
GRPO_EPS = 1e-8         # Numerical stability for reward normalisation

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert RAG pipeline debugger. Given an observation of a broken
    RAG pipeline, output exactly ONE corrective action as a JSON object.

    JSON format: {"action_type": "<type>", "params": {<relevant params>}}

    Available action types:
      adjust_chunk_size     {"value": int}       -- range 64-2048
      adjust_chunk_overlap  {"value": int}        -- range 0-500
      adjust_threshold      {"value": float}      -- range 0.0-1.0
      adjust_top_k          {"value": int}        -- range 1-50
      swap_embedding_model  {"model": "general"|"medical"|"legal"|"code"}
      toggle_reranking      {"enabled": bool}
      adjust_context_limit  {"value": int}        -- range 512-16384
      rewrite_query         {"query_id": int, "strategy": "rephrase"}
      submit                {}                    -- ends episode

    Diagnose first, then act:
      - Empty retrievals → lower threshold or increase top_k
      - Low coverage + decent precision → top_k too small
      - Compressed scores (std < 0.05) → wrong embedding model
      - Context overflow → increase context_limit
      - Submit when coverage >= 0.75 and no empty retrievals
""").strip()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    prompt: str
    response: str
    reward: float


@dataclass
class RolloutRecord:
    task_id: int
    batch: int
    rollout: int
    episode_seed: int
    steps: List[StepRecord] = field(default_factory=list)
    total_reward: float = 0.0
    normalized_reward: float = 0.0
    success: bool = False

    def to_dict(self):
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Observation formatter (compact for smaller context)
# ---------------------------------------------------------------------------

def _format_obs(obs: RAGDebugObservation, step: int) -> str:
    cfg = obs.pipeline_config
    m = obs.metrics
    lines = [
        f"Step {step}/{obs.max_steps} | Task {obs.task_id}",
        f"Config: chunk_size={cfg.chunk_size} overlap={cfg.chunk_overlap} "
        f"threshold={cfg.similarity_threshold} top_k={cfg.top_k} "
        f"model={cfg.embedding_model.value} reranking={cfg.use_reranking} "
        f"context_limit={cfg.context_window_limit}",
        f"Metrics: coverage={m.mean_coverage:.3f} precision={m.mean_precision:.3f} "
        f"empty={m.n_empty_retrievals} overflow={m.n_context_overflows}",
    ]
    if m.multi_hop_coverage is not None:
        lines.append(f"Multi-hop coverage: {m.multi_hop_coverage:.3f}")
    lines.append("Per-query: " + "  ".join(
        f"q{r.query_id}(cov={r.coverage_score:.2f},n={r.n_retrieved})"
        for r in obs.query_results
    ))
    if obs.diagnostic_hints:
        lines.append("Hints: " + " | ".join(obs.diagnostic_hints[:2]))
    lines.append("\nOutput exactly one JSON action:")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Action parsing with heuristic fallback
# ---------------------------------------------------------------------------

def _parse_action(text: str, obs: RAGDebugObservation) -> RAGDebugAction:
    text = text.strip()
    # Strip markdown fences
    if "```" in text:
        text = "\n".join(l for l in text.splitlines() if not l.strip().startswith("```"))
    # Find last JSON-like line
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and "action_type" in line:
            try:
                data = json.loads(line)
                return RAGDebugAction(
                    action_type=ActionType(data["action_type"]),
                    params=data.get("params", {}),
                )
            except Exception:
                continue
    # Fallback: full text as JSON
    try:
        data = json.loads(text)
        return RAGDebugAction(
            action_type=ActionType(data["action_type"]),
            params=data.get("params", {}),
        )
    except Exception:
        pass
    # Heuristic fallback
    m = obs.metrics
    cfg = obs.pipeline_config
    if m.n_empty_retrievals > 0 and cfg.similarity_threshold > 0.15:
        return RAGDebugAction(action_type=ActionType.ADJUST_THRESHOLD,
                              params={"value": round(max(0.05, cfg.similarity_threshold - 0.15), 2)})
    if obs.task_id == 3 and cfg.embedding_model.value != "medical":
        return RAGDebugAction(action_type=ActionType.SWAP_EMBEDDING_MODEL,
                              params={"model": "medical"})
    if m.mean_coverage < 0.6 and cfg.top_k < 20:
        return RAGDebugAction(action_type=ActionType.ADJUST_TOP_K,
                              params={"value": min(30, cfg.top_k + 5)})
    return RAGDebugAction(action_type=ActionType.SUBMIT, params={})


# ---------------------------------------------------------------------------
# Single rollout
# ---------------------------------------------------------------------------

def run_rollout(
    llm: OpenAI,
    env: RAGDebugEnv,
    task_id: int,
    seed: int,
    batch: int,
    rollout_idx: int,
) -> RolloutRecord:
    record = RolloutRecord(task_id=task_id, batch=batch, rollout=rollout_idx,
                           episode_seed=seed)

    result = env.reset(task_id=task_id, seed=seed)
    obs: RAGDebugObservation = result.observation
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for step in range(1, MAX_STEPS + 1):
        if obs.done:
            break

        prompt = _format_obs(obs, step)
        messages.append({"role": "user", "content": prompt})

        try:
            completion = llm.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            response_text = (completion.choices[0].message.content or "").strip()
        except Exception as exc:
            response_text = '{"action_type":"submit","params":{}}'
            print(f"  [LLM error] {exc}")

        messages.append({"role": "assistant", "content": response_text})
        action = _parse_action(response_text, obs)

        try:
            step_result = env.step(action)
        except RuntimeError:
            break

        reward = float(step_result.reward or 0.0)
        obs = step_result.observation
        record.steps.append(StepRecord(prompt=prompt, response=response_text, reward=reward))
        record.total_reward += reward

        if obs.done:
            break

    # Infer success: terminal reward >= 0.7 → success zone
    if record.steps:
        last_reward = record.steps[-1].reward
        record.success = last_reward >= 0.7
    return record


# ---------------------------------------------------------------------------
# GRPO batch
# ---------------------------------------------------------------------------

def grpo_normalize(rollouts: List[RolloutRecord]) -> None:
    """
    Apply GRPO reward normalisation in-place.

    Within the group, subtract the mean total reward and divide by std.
    This is the core GRPO operation: each rollout is evaluated relative
    to the other rollouts from the same task/batch, not against an absolute
    baseline. Rollouts better than average get positive normalised rewards;
    those worse than average get negative ones.
    """
    rewards = [r.total_reward for r in rollouts]
    mean_r = sum(rewards) / len(rewards)
    variance = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
    std_r = variance ** 0.5
    for rollout in rollouts:
        rollout.normalized_reward = (rollout.total_reward - mean_r) / (std_r + GRPO_EPS)


def run_batch(
    llm: OpenAI,
    env: RAGDebugEnv,
    task_id: int,
    batch_idx: int,
    group_size: int,
    seed_base: int,
) -> List[RolloutRecord]:
    rollouts = []
    for i in range(group_size):
        seed = seed_base + i
        print(f"  Rollout {i + 1}/{group_size} (seed={seed}) ...", end=" ", flush=True)
        rollout = run_rollout(llm, env, task_id, seed, batch_idx, i)
        rollouts.append(rollout)
        outcome = "SUCCESS" if rollout.success else "failed"
        print(f"{outcome}  total_reward={rollout.total_reward:.3f}  steps={len(rollout.steps)}")

    grpo_normalize(rollouts)
    return rollouts


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_batch_summary(rollouts: List[RolloutRecord], task_id: int, batch: int) -> None:
    successes = sum(1 for r in rollouts if r.success)
    avg_reward = sum(r.total_reward for r in rollouts) / len(rollouts)
    avg_steps = sum(len(r.steps) for r in rollouts) / len(rollouts)
    norm_rewards = [r.normalized_reward for r in rollouts]
    best = max(norm_rewards)
    worst = min(norm_rewards)
    print(f"  Batch {batch} summary (task {task_id}):")
    print(f"    Success rate:      {successes}/{len(rollouts)}")
    print(f"    Avg total reward:  {avg_reward:.3f}")
    print(f"    Avg steps:         {avg_steps:.1f}")
    print(f"    Norm reward range: [{worst:+.3f}, {best:+.3f}]")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO training scaffold for RAGDebugEnv")
    parser.add_argument("--task", choices=["1", "2", "3", "all"], default="1")
    parser.add_argument("--batches", type=int, default=2,
                        help="Number of GRPO batches per task (default: 2)")
    parser.add_argument("--group-size", type=int, default=4,
                        help="Rollouts per batch (GRPO group size G, default: 4)")
    parser.add_argument("--server", default=SERVER_URL,
                        help=f"Environment server URL (default: {SERVER_URL})")
    parser.add_argument("--seed", type=int, default=100,
                        help="Base seed (rollouts use seed + rollout_index, default: 100)")
    parser.add_argument("--output", default=str(DATA_PATH),
                        help="Path to write JSONL training data")
    args = parser.parse_args()

    if not API_KEY:
        raise RuntimeError("Set HF_TOKEN, OPENAI_API_KEY, or API_KEY in environment")

    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    tasks = [1, 2, 3] if args.task == "all" else [int(args.task)]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_rollouts: List[RolloutRecord] = []

    env = RAGDebugEnv(base_url=args.server)
    with env.sync() as env:
        for task_id in tasks:
            print(f"\n{'='*60}")
            print(f"  Task {task_id}  |  {args.batches} batches  x  {args.group_size} rollouts")
            print(f"{'='*60}")

            for batch_idx in range(1, args.batches + 1):
                print(f"\n  Batch {batch_idx}/{args.batches}")
                seed_base = args.seed + (batch_idx - 1) * args.group_size
                rollouts = run_batch(llm, env, task_id, batch_idx,
                                     args.group_size, seed_base)
                print_batch_summary(rollouts, task_id, batch_idx)
                all_rollouts.extend(rollouts)

    # Write JSONL training data
    with output_path.open("w") as f:
        for r in all_rollouts:
            f.write(json.dumps(r.to_dict()) + "\n")

    print(f"Training data written to: {output_path}")
    print(f"Total rollouts: {len(all_rollouts)}")
    print(f"Total steps recorded: {sum(len(r.steps) for r in all_rollouts)}")
    success_rate = sum(1 for r in all_rollouts if r.success) / len(all_rollouts)
    print(f"Overall success rate: {success_rate:.1%}")
    print()
    print("Next steps for full gradient-based GRPO training:")
    print("  pip install trl>=0.9.0 transformers>=4.45.0 torch peft accelerate")
    print("  Use grpo_data.jsonl as the reward signal dataset with TRL's GRPOTrainer.")


if __name__ == "__main__":
    main()
