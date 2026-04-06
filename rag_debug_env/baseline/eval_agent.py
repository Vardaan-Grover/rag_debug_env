"""
baseline/eval_agent.py
----------------------
Zero-shot eval agent using GPT-4o-mini to diagnose and fix broken RAG pipelines.

Purpose: validate the environment end-to-end — confirm reward signals are
meaningful, observations are interpretable, and the tasks are solvable by a
capable model before committing to GRPO training.

Usage:
    # Server must be running first:
    #   uvicorn rag_debug_env.server.app:app --host 0.0.0.0 --port 8000

    python baseline/eval_agent.py --task 1 --episodes 3
    python baseline/eval_agent.py --task all --episodes 2 --verbose
    python baseline/eval_agent.py --task 2 --seed 42 --server http://localhost:8000

Requirements:
    OPENAI_API_KEY environment variable must be set.
    pip install openai
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from enum import Enum
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from src import RagDebugEnv, RAGDebugAction, RAGDebugObservation

load_dotenv()

# ---------------------------------------------------------------------------
# OpenAI structured output schema
# ---------------------------------------------------------------------------

class _ActionType(str, Enum):
    ADJUST_CHUNK_SIZE    = "adjust_chunk_size"
    ADJUST_CHUNK_OVERLAP = "adjust_chunk_overlap"
    ADJUST_THRESHOLD     = "adjust_threshold"
    ADJUST_TOP_K         = "adjust_top_k"
    SWAP_EMBEDDING_MODEL = "swap_embedding_model"
    TOGGLE_RERANKING     = "toggle_reranking"
    ADJUST_CONTEXT_LIMIT = "adjust_context_limit"
    REWRITE_QUERY        = "rewrite_query"
    SUBMIT               = "submit"


class AgentDecision(BaseModel):
    """Structured output schema enforced by OpenAI's API."""
    reasoning: str
    action_type: _ActionType
    # Flat param fields — fill only the one(s) relevant to your action_type.
    # int_value   : chunk_size, top_k, context_limit, chunk_overlap
    # float_value : similarity_threshold
    # model_name  : embedding model ("general" | "medical" | "legal" | "code")
    # enabled     : reranking toggle (True/False)
    # query_id    : query to rewrite
    int_value:   Optional[int]   = None
    float_value: Optional[float] = None
    model_name:  Optional[str]   = None
    enabled:     Optional[bool]  = None
    query_id:    Optional[int]   = None


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert RAG (Retrieval-Augmented Generation) pipeline debugger.

Your job is to diagnose why a RAG pipeline is performing poorly and take
corrective actions to restore retrieval quality. You will be given an
observation describing the current pipeline state, per-query results, and
aggregate metrics.

## Available Actions

| Action               | Required param       | Effect                              |
|----------------------|----------------------|-------------------------------------|
| adjust_chunk_size    | int_value (64-2048)  | Change chunk size                   |
| adjust_chunk_overlap | int_value (0-500)    | Change chunk overlap                |
| adjust_threshold     | float_value (0.0-1.0)| Change similarity threshold         |
| adjust_top_k         | int_value (1-50)     | Change number of retrieved chunks   |
| swap_embedding_model | model_name           | Switch embedding model              |
| toggle_reranking     | enabled (bool)       | Enable/disable cross-encoder rerank |
| adjust_context_limit | int_value (512-16384)| Change context window limit         |
| rewrite_query        | query_id (int)       | Boost a specific query              |
| submit               | (none)               | Submit — ends the episode           |

## Embedding Models
- "general"  — all-purpose (sentence-transformers/all-MiniLM-L6-v2)
- "medical"  — biomedical text (PubMedBert-MS-MARCO)
- "legal"    — legal documents (legal-bert-base-uncased)
- "code"     — code + docstrings (codebert-base)

## Diagnostic Heuristics
- Low coverage + low precision + many empty retrievals → threshold may be too high, or top_k too small
- Low coverage + moderate precision → top_k too small, or embedding model mismatch
- Many retrieved chunks but low coverage → duplicate flooding, or threshold too low letting noise through
- Score distribution compressed (all scores similar) → wrong embedding model, or chunk too large
- Coverage plateaus despite config changes → wrong embedding model (especially on domain-specific text)
- Context overflow → increase context_limit or decrease top_k
- Submit only when mean_coverage >= 0.70 and no empty retrievals

Fill in only the param field relevant to your chosen action. Leave others as null.
"""


# ---------------------------------------------------------------------------
# Observation formatter
# ---------------------------------------------------------------------------

def _format_observation(obs: RAGDebugObservation, action_history: list[dict]) -> str:
    """Convert an observation to a readable prompt string for the model."""
    cfg = obs.pipeline_config
    m   = obs.metrics
    cs  = obs.corpus_stats

    lines = [
        f"## Task {obs.task_id}: {obs.task_description}",
        f"Step {obs.steps_taken} / {obs.max_steps}",
        "",
        "## Current Pipeline Config",
        f"  chunk_size        = {cfg.chunk_size}",
        f"  chunk_overlap     = {cfg.chunk_overlap}",
        f"  similarity_threshold = {cfg.similarity_threshold}",
        f"  top_k             = {cfg.top_k}",
        f"  embedding_model   = {cfg.embedding_model.value}",
        f"  use_reranking     = {cfg.use_reranking}",
        f"  context_window_limit = {cfg.context_window_limit}",
        "",
        "## Corpus Info",
        f"  domain = {cs.domain.value}  |  {cs.n_chunks} chunks  |  {cs.n_queries} queries",
        f"  multi-hop queries: {cs.n_multi_hop_queries}",
        "",
        "## Aggregate Metrics",
        f"  mean_coverage    = {m.mean_coverage:.3f}",
        f"  mean_precision   = {m.mean_precision:.3f}",
        f"  empty retrievals = {m.n_empty_retrievals}",
        f"  context overflows = {m.n_context_overflows}",
    ]
    if m.multi_hop_coverage is not None:
        lines.append(f"  multi_hop_coverage = {m.multi_hop_coverage:.3f}")

    lines += ["", "## Per-Query Results"]
    for qr in obs.query_results:
        mh_tag = " [multi-hop]" if qr.is_multi_hop else ""
        score_summary = ""
        if qr.retrieval_scores:
            score_summary = (
                f"  scores: min={min(qr.retrieval_scores):.3f} "
                f"max={max(qr.retrieval_scores):.3f} "
                f"mean={sum(qr.retrieval_scores)/len(qr.retrieval_scores):.3f}"
            )
        lines.append(
            f"  Q{qr.query_id}{mh_tag}: coverage={qr.coverage_score:.3f} "
            f"precision={qr.precision_score:.3f} "
            f"retrieved={qr.n_retrieved}{score_summary}"
        )
        if qr.n_retrieved == 0:
            lines.append(f"    !! empty retrieval — no chunks above threshold")

    if action_history:
        lines += ["", "## Actions Taken So Far"]
        for i, ah in enumerate(action_history, 1):
            lines.append(f"  {i}. {ah['action_type']}({ah.get('params', {})})  reward={ah['reward']:+.3f}")

    return "\n".join(lines)


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def _emit_log(message: str, log_file: Optional[str]) -> None:
    print(message)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(message)
            f.write("\n")


def _estimate_task_score(obs: RAGDebugObservation) -> tuple[float, float, bool]:
    """
    Estimate the environment success score from visible metrics.

    Returns (estimated_score, success_threshold, extra_condition_ok).
    """
    m = obs.metrics
    if obs.task_id in (1, 2):
        efficiency = 1.0 - (obs.steps_taken / max(obs.max_steps, 1))
        score = 0.60 * m.mean_coverage + 0.25 * m.mean_precision + 0.15 * efficiency
        threshold = 0.75
        return score, threshold, True

    # Task 3 has an extra multi-hop condition.
    mh_cov = m.multi_hop_coverage or 0.0
    score = 0.55 * m.mean_coverage + 0.25 * m.mean_precision + 0.20 * mh_cov
    threshold = 0.70
    return score, threshold, mh_cov > 0.60


def _submit_confidence(obs: RAGDebugObservation, ready_streak: int) -> tuple[float, bool, str]:
    """Return (confidence_0_to_1, ready_now, human_readable_reason)."""
    est_score, threshold, extra_ok = _estimate_task_score(obs)
    score_ratio = est_score / threshold if threshold > 0 else 0.0
    confidence = max(0.0, min(1.0, score_ratio))
    ready_now = est_score >= threshold and extra_ok and obs.metrics.n_empty_retrievals == 0

    reason = (
        f"score={est_score:.3f}/{threshold:.3f} "
        f"cov={obs.metrics.mean_coverage:.3f} "
        f"prec={obs.metrics.mean_precision:.3f} "
        f"empty={obs.metrics.n_empty_retrievals} "
        f"streak={ready_streak}"
    )
    if obs.task_id == 3:
        reason += f" mh_cov={(obs.metrics.multi_hop_coverage or 0.0):.3f}"
    return confidence, ready_now, reason


def _fallback_action_for_swap_cooldown(obs: RAGDebugObservation) -> RAGDebugAction:
    """Pick a safe non-swap action when a swap is blocked by cooldown."""
    cfg = obs.pipeline_config
    m = obs.metrics

    if not cfg.use_reranking:
        return RAGDebugAction(action_type="toggle_reranking", params={"enabled": True})

    if m.mean_precision < 0.12 and cfg.similarity_threshold < 0.70:
        return RAGDebugAction(
            action_type="adjust_threshold",
            params={"value": round(min(0.70, cfg.similarity_threshold + 0.05), 3)},
        )

    if m.mean_coverage < 0.70 and cfg.top_k < 30:
        return RAGDebugAction(action_type="adjust_top_k", params={"value": min(30, cfg.top_k + 5)})

    return RAGDebugAction(
        action_type="adjust_context_limit",
        params={"value": min(16384, cfg.context_window_limit + 512)},
    )


# ---------------------------------------------------------------------------
# Action builder
# ---------------------------------------------------------------------------

def _decision_to_action(decision: AgentDecision) -> RAGDebugAction:
    """Convert AgentDecision (structured output) to RAGDebugAction."""
    at = decision.action_type.value
    params: dict = {}

    if at in ("adjust_chunk_size", "adjust_top_k", "adjust_context_limit", "adjust_chunk_overlap"):
        if decision.int_value is not None:
            params["value"] = decision.int_value

    elif at == "adjust_threshold":
        if decision.float_value is not None:
            params["value"] = decision.float_value

    elif at == "swap_embedding_model":
        if decision.model_name:
            params["model"] = decision.model_name

    elif at == "toggle_reranking":
        if decision.enabled is not None:
            params["enabled"] = decision.enabled

    elif at == "rewrite_query":
        if decision.query_id is not None:
            params["query_id"] = decision.query_id

    # submit: no params needed

    return RAGDebugAction(action_type=at, params=params)


# ---------------------------------------------------------------------------
# Single episode
# ---------------------------------------------------------------------------

def run_episode(
    client: OpenAI,
    env: RagDebugEnv,
    task_id: int,
    seed: Optional[int],
    episode_num: int,
    verbose: bool = False,
    log_system_prompt: bool = False,
    log_llm_input: bool = False,
    log_llm_output: bool = False,
    log_llm_usage: bool = False,
    log_file: Optional[str] = None,
    log_max_chars: int = 4000,
    submit_streak_required: int = 2,
    swap_cooldown_steps: int = 2,
    log_submit_confidence: bool = False,
) -> dict:
    """
    Run one episode and return a result dict.

    Returns
    -------
    {task_id, episode, seed, steps, final_coverage, final_precision,
     success, total_reward, actions}
    """
    reset_kwargs: dict = {"task_id": task_id}
    if seed is not None:
        reset_kwargs["seed"] = seed

    result = env.reset(**reset_kwargs)
    obs: RAGDebugObservation = result.observation

    action_history: list[dict] = []
    total_reward = 0.0
    success = False
    ready_submit_streak = 0
    last_swap_step: Optional[int] = None

    print(f"\n  Episode {episode_num} (task={task_id})")
    print(f"  {'─'*50}")
    print(f"  Initial state: coverage={obs.metrics.mean_coverage:.3f}  "
          f"precision={obs.metrics.mean_precision:.3f}  "
          f"empty={obs.metrics.n_empty_retrievals}")

    if log_system_prompt:
        _emit_log("\n--- LLM System Prompt ---", log_file)
        _emit_log(_truncate(_SYSTEM_PROMPT, log_max_chars), log_file)

    while not obs.done and obs.steps_taken < obs.max_steps:
        confidence, ready_now, reason = _submit_confidence(obs, ready_submit_streak)
        ready_submit_streak = ready_submit_streak + 1 if ready_now else 0

        if log_submit_confidence:
            _emit_log(
                "\n--- Submit Confidence (step "
                f"{obs.steps_taken}) --- confidence={confidence:.2f} "
                f"ready={ready_now} {reason}",
                log_file,
            )

        observation_text = _format_observation(obs, action_history)

        if verbose:
            print(f"\n--- Observation (step {obs.steps_taken}) ---")
            print(observation_text)

        if log_llm_input:
            _emit_log(f"\n--- LLM Input (step {obs.steps_taken}) ---", log_file)
            _emit_log(_truncate(observation_text, log_max_chars), log_file)

        # Call GPT-4o-mini with structured output
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": observation_text},
            ],
            response_format=AgentDecision,
            temperature=0.2,
        )

        decision: AgentDecision = response.choices[0].message.parsed
        action = _decision_to_action(decision)
        override_reason: Optional[str] = None

        # If the environment looks solved for multiple consecutive steps,
        # force a SUBMIT so the policy doesn't waste steps and time out.
        if ready_submit_streak >= submit_streak_required and action.action_type != "submit":
            action = RAGDebugAction(action_type="submit", params={})
            override_reason = (
                f"forced_submit_after_streak={ready_submit_streak} "
                f"(required={submit_streak_required})"
            )

        # Prevent swap thrashing: block frequent back-to-back model swaps.
        if action.action_type == "swap_embedding_model" and last_swap_step is not None:
            delta = obs.steps_taken - last_swap_step
            if delta < swap_cooldown_steps:
                action = _fallback_action_for_swap_cooldown(obs)
                override_reason = (
                    f"swap_blocked_by_cooldown delta={delta} "
                    f"cooldown={swap_cooldown_steps}"
                )

        if log_llm_output and override_reason:
            _emit_log(f"Override: {override_reason}", log_file)

        if log_llm_output:
            raw_content = response.choices[0].message.content or ""
            output_record = {
                "reasoning": decision.reasoning,
                "action_type": decision.action_type.value,
                "int_value": decision.int_value,
                "float_value": decision.float_value,
                "model_name": decision.model_name,
                "enabled": decision.enabled,
                "query_id": decision.query_id,
                "raw_content": raw_content,
            }
            _emit_log(f"\n--- LLM Output (step {obs.steps_taken}) ---", log_file)
            _emit_log(
                _truncate(json.dumps(output_record, indent=2, ensure_ascii=False), log_max_chars),
                log_file,
            )

        if log_llm_usage and getattr(response, "usage", None) is not None:
            usage = response.usage
            _emit_log(
                "\n--- LLM Token Usage (step "
                f"{obs.steps_taken}) --- prompt={usage.prompt_tokens} "
                f"completion={usage.completion_tokens} total={usage.total_tokens}",
                log_file,
            )

        if verbose:
            print(f"\n  Reasoning: {decision.reasoning[:200]}")

        step_result = env.step(action)
        reward = step_result.reward or 0.0
        total_reward += reward
        obs = step_result.observation

        if action.action_type == "swap_embedding_model":
            last_swap_step = obs.steps_taken

        action_history.append({
            "action_type": action.action_type,
            "params": action.params,
            "reward": reward,
        })

        cov_str = f"coverage={obs.metrics.mean_coverage:.3f}"
        print(
            f"  Step {obs.steps_taken:2d}: {action.action_type:<22} "
            f"reward={reward:+.3f}  {cov_str}"
        )

        if obs.done or step_result.done:
            final_coverage = obs.metrics.mean_coverage
            final_precision = obs.metrics.mean_precision
            # Infer success from terminal reward
            success = reward >= 1.5
            break

    outcome = "SUCCESS ✓" if success else "failed ✗"
    print(f"  {'─'*50}")
    print(f"  {outcome}  |  total_reward={total_reward:+.3f}  "
          f"final_coverage={obs.metrics.mean_coverage:.3f}  "
          f"steps={obs.steps_taken}")

    return {
        "task_id":          task_id,
        "episode":          episode_num,
        "seed":             seed,
        "steps":            obs.steps_taken,
        "final_coverage":   obs.metrics.mean_coverage,
        "final_precision":  obs.metrics.mean_precision,
        "success":          success,
        "total_reward":     total_reward,
        "actions":          [ah["action_type"] for ah in action_history],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPT-4o-mini zero-shot eval agent for RAGDebugEnv"
    )
    parser.add_argument(
        "--task", choices=["1", "2", "3", "all"], default="1",
        help="Task ID to evaluate (default: 1)",
    )
    parser.add_argument(
        "--episodes", type=int, default=3,
        help="Number of episodes per task (default: 3)",
    )
    parser.add_argument(
        "--server", default="http://localhost:8000",
        help="Environment server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility (default: random)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print full observation each step",
    )
    parser.add_argument(
        "--log-system-prompt", action="store_true",
        help="Log the full system prompt once per episode",
    )
    parser.add_argument(
        "--log-llm-input", action="store_true",
        help="Log the observation text sent to the LLM every step",
    )
    parser.add_argument(
        "--log-llm-output", action="store_true",
        help="Log parsed decision payload (and raw content) returned by the LLM",
    )
    parser.add_argument(
        "--log-llm-usage", action="store_true",
        help="Log token usage per LLM call",
    )
    parser.add_argument(
        "--log-file", default=None,
        help="Optional file path to append logs to",
    )
    parser.add_argument(
        "--log-max-chars", type=int, default=4000,
        help="Max characters per logged block before truncation (default: 4000)",
    )
    parser.add_argument(
        "--submit-streak", type=int, default=2,
        help="Consecutive ready steps required before forcing submit (default: 2)",
    )
    parser.add_argument(
        "--swap-cooldown-steps", type=int, default=2,
        help="Minimum steps between embedding swaps (default: 2)",
    )
    parser.add_argument(
        "--log-submit-confidence", action="store_true",
        help="Log per-step submit confidence and estimated success score",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in environment.", file=sys.stderr)
        sys.exit(1)

    openai_client = OpenAI(api_key=api_key)

    tasks = [1, 2, 3] if args.task == "all" else [int(args.task)]
    all_results: list[dict] = []

    with RagDebugEnv(base_url=args.server).sync() as env:
        for task_id in tasks:
            print(f"\n{'='*60}")
            print(f"  Task {task_id}  ({args.episodes} episodes)")
            print(f"{'='*60}")

            for ep in range(1, args.episodes + 1):
                seed = args.seed if args.seed is not None else None
                try:
                    result = run_episode(
                        client=openai_client,
                        env=env,
                        task_id=task_id,
                        seed=seed,
                        episode_num=ep,
                        verbose=args.verbose,
                        log_system_prompt=args.log_system_prompt,
                        log_llm_input=args.log_llm_input,
                        log_llm_output=args.log_llm_output,
                        log_llm_usage=args.log_llm_usage,
                        log_file=args.log_file,
                        log_max_chars=args.log_max_chars,
                        submit_streak_required=args.submit_streak,
                        swap_cooldown_steps=args.swap_cooldown_steps,
                        log_submit_confidence=args.log_submit_confidence,
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"\n  ERROR in episode {ep}: {e}", file=sys.stderr)
                    import traceback
                    traceback.print_exc()

    # Summary table
    if all_results:
        print(f"\n{'='*60}")
        print(f"  Summary")
        print(f"{'='*60}")
        for task_id in tasks:
            task_results = [r for r in all_results if r["task_id"] == task_id]
            if not task_results:
                continue
            n_success = sum(1 for r in task_results if r["success"])
            avg_cov = sum(r["final_coverage"] for r in task_results) / len(task_results)
            avg_steps = sum(r["steps"] for r in task_results) / len(task_results)
            avg_reward = sum(r["total_reward"] for r in task_results) / len(task_results)
            print(
                f"  Task {task_id}: {n_success}/{len(task_results)} success  "
                f"avg_coverage={avg_cov:.3f}  avg_steps={avg_steps:.1f}  "
                f"avg_reward={avg_reward:+.3f}"
            )


if __name__ == "__main__":
    main()
