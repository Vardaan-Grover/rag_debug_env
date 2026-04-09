"""
Inference Script - RAGDebugEnv
==============================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL      The API endpoint for the LLM.
    MODEL_NAME        The model identifier to use for inference.
    HF_TOKEN          Your Hugging Face / API key.
    LOCAL_IMAGE_NAME  Optional local image name when using from_docker_image().

- Defaults are set only for API_BASE_URL and MODEL_NAME:
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")

- The inference script must be named inference.py and placed in the root directory of the project.
- Participants must use OpenAI Client for all LLM calls.

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - The task score must be in [0, 1].
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import textwrap
from typing import List, Optional, Tuple

from openai import OpenAI

from client import RAGDebugEnv
from models import ActionType, RAGDebugAction, RAGDebugObservation


API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "nebius/Qwen/Qwen2.5-72B-Instruct"

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:7860")

TASK_ID = int(os.getenv("RAG_DEBUG_TASK", "1"))
BENCHMARK = os.getenv("RAG_DEBUG_BENCHMARK", "rag_debug_env")
MAX_STEPS_OVERRIDE = int(os.getenv("RAG_DEBUG_MAX_STEPS", "10"))

TEMPERATURE = 0.2
MAX_TOKENS = 512

W = 64  # display width for visual output

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert RAG retrieval debugger. Your task is to diagnose and fix a broken RAG pipeline by analyzing retrieval metrics and adjusting configuration parameters.

    ## Output Format
    Reason step-by-step before acting:
    1. **Diagnosis**: What do the current metrics reveal? What fault(s) do you suspect?
    2. **History**: What have your previous actions changed? What hypotheses are confirmed or ruled out?
    3. **Plan**: What to try next and why?

    Then output exactly one JSON object on its own line:
    {"action_type":"<type>","params":{...}}

    ## Available Actions & Parameters
    - adjust_chunk_size: {"value": int}       — range 64-2048 (default 512)
    - adjust_chunk_overlap: {"value": int}    — range 0-500 (default 50)
    - adjust_threshold: {"value": float}      — range 0.0-1.0 (default 0.3)
    - adjust_top_k: {"value": int}            — range 1-50 (default 10)
    - swap_embedding_model: {"model": "general"|"medical"|"legal"|"code"}
    - toggle_reranking: {"enabled": bool}
    - adjust_context_limit: {"value": int}    — range 512-16384 (default 4096)
    - rewrite_query: {"query_id": int, "strategy": "rephrase"}
    - submit: {}                              — ends the episode; only submit when metrics are good

    ## Diagnostic Framework
    Analyze the observation systematically:

    1. **Empty retrievals** (n_retrieved=0 for some queries):
       - Primary cause: similarity_threshold is too high
       - Fix: lower threshold (try 0.15 or 0.10) or increase top_k

    2. **Low coverage + decent precision** (coverage < 0.6, precision > 0.4):
       - Primary cause: top_k is too small
       - Fix: increase top_k (try 15-20)

    3. **Low coverage + low precision** (both < 0.4):
       - Primary cause: wrong embedding model OR chunk_size too large
       - Fix: swap embedding model to match domain, reduce chunk_size to 256-384
       - For medical domain: swap to "medical" model
       - For legal domain: swap to "legal" model

    4. **Score compression** (all retrieval scores are similar, e.g., std < 0.05):
       - Primary cause: wrong embedding model or score compression fault
       - Fix: swap embedding model, then enable reranking

    5. **Context overflow** (n_context_overflows > 0):
       - Fix: increase context_window_limit (try 8192 or 16384) or reduce top_k

    6. **High scores but low precision** (mean score > 0.7 but precision < 0.5):
       - Primary cause: duplicate flooding or threshold too low
       - Fix: enable reranking, or raise threshold slightly

    ## Task Score (determines success/failure)
    Tasks 1 & 2: score = 0.60*coverage + 0.25*precision + 0.15*(1 - steps/max_steps)
      Success requires score >= 0.75. Fewer steps = higher efficiency bonus.
    Task 3: score = 0.55*coverage + 0.25*precision + 0.20*multi_hop_coverage
      Success requires score >= 0.70 AND multi_hop_coverage > 0.60

    ## Strategy
    - Act efficiently: each step reduces your efficiency bonus. Diagnose fast, fix precisely.
    - Fix the most impactful issue first (usually the config anomaly, empty retrievals, or wrong model).
    - Enable reranking if precision is low — it helps with multiple fault types.
    - For Tasks 1 & 2, avoid swapping embedding model unless there is very strong evidence of model mismatch.
    - If coverage is high (>=0.85) but precision is low (<0.35), stop increasing top_k; prioritize reranking and raising threshold.
    - For Task 3 (medical): always swap embedding model to "medical".
    - For Tasks 1 & 2, submit only when score >= 0.77, coverage >= 0.80, precision >= 0.40, and empty retrievals = 0.
    - For Task 3, submit only when score >= 0.78 and multi_hop_coverage > 0.60.

    ## Cross-Step Reasoning
    You will see your previous reasoning and actions in the conversation history.
    Use this to track hypotheses across steps:
    - If an action had no effect, the root cause is likely elsewhere — say so explicitly.
    - If lowering threshold didn't fix empty retrievals, suspect wrong embedding model.
    - If increasing top_k didn't improve coverage, the scores themselves are wrong (model mismatch).
    - Never repeat an action that produced no improvement. Build on what you've learned.
    """
).strip()


# ─── Rich Display (stderr) ─────────────────────────────────────────────────────

def _rich(text: str = "") -> None:
    """Print rich visual output to stderr (visible in terminal, invisible to stdout parsers)."""
    print(text, file=sys.stderr, flush=True)


def _bar(value: float, width: int = 25, fill: str = "█", empty: str = "░") -> str:
    """Render a horizontal bar for a 0.0–1.0 value."""
    n = int(round(value * width))
    return fill * n + empty * (width - n)


def _delta_str(old: float, new: float) -> str:
    """Format a metric delta with sign and color hint."""
    d = new - old
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.3f}"


def _show_banner() -> None:
    _rich()
    _rich(f"{'─' * W}")
    _rich(f"  RAGDebugEnv  │  RL Environment for Diagnosing Broken RAG Pipelines")
    _rich(f"{'─' * W}")
    _rich()
    _rich(f"  Model       {MODEL_NAME}")
    _rich(f"  Endpoint    {API_BASE_URL}")
    _rich(f"  Task        task_{TASK_ID}  ({BENCHMARK})")
    _rich()


def _show_task_info(obs: RAGDebugObservation) -> None:
    cs = obs.corpus_stats
    _rich(f"{'─' * W}")
    _rich(f"  TASK DESCRIPTION")
    # Wrap task description to fit nicely
    for line in textwrap.wrap(obs.task_description, width=W - 4):
        _rich(f"  {line}")
    _rich()
    mh_str = str(cs.n_multi_hop_queries) if cs.n_multi_hop_queries else "n/a"
    _rich(f"  CORPUS  domain={cs.domain.value}  docs={cs.n_documents}  chunks={cs.n_chunks}  "
          f"avg_tokens={cs.avg_chunk_tokens}  queries={cs.n_queries}  multi_hop={mh_str}")
    _rich(f"{'─' * W}")


def _show_config(obs: RAGDebugObservation) -> None:
    cfg = obs.pipeline_config
    _rich(f"  CONFIG  chunk_size={cfg.chunk_size}  overlap={cfg.chunk_overlap}  "
          f"threshold={cfg.similarity_threshold}  top_k={cfg.top_k}")
    _rich(f"          model={cfg.embedding_model.value}  reranking={'on' if cfg.use_reranking else 'off'}  "
          f"context_limit={cfg.context_window_limit}")


def _show_metrics(obs: RAGDebugObservation, label: str = "METRICS") -> None:
    m = obs.metrics
    mh = f"{m.multi_hop_coverage:.3f}" if m.multi_hop_coverage is not None else "n/a"
    _rich(f"  {label}")
    _rich(f"    coverage   {_bar(m.mean_coverage)} {m.mean_coverage:.3f}")
    _rich(f"    precision  {_bar(m.mean_precision)} {m.mean_precision:.3f}")
    _rich(f"    empty={m.n_empty_retrievals}  overflow={m.n_context_overflows}  multi_hop={mh}")


def _show_queries(obs: RAGDebugObservation) -> None:
    _rich(f"  QUERIES")
    for qr in obs.query_results:
        mh_flag = " [multi-hop]" if qr.is_multi_hop else ""
        _rich(f"    q{qr.query_id:>3}: cov={qr.coverage_score:.3f}  "
              f"prec={qr.precision_score:.3f}  n={qr.n_retrieved:>2}{mh_flag}")


def _show_initial_state(obs: RAGDebugObservation) -> None:
    _rich()
    _rich(f"  INITIAL STATE  (step 0/{obs.max_steps})")
    _show_config(obs)
    _rich()
    _show_metrics(obs, label="INITIAL METRICS")
    _rich()
    _show_queries(obs)
    _rich(f"{'─' * W}")


def _show_step(
    step: int,
    max_steps: int,
    action_str: str,
    reward: float,
    obs: RAGDebugObservation,
    prev_cov: float,
    prev_prec: float,
) -> None:
    m = obs.metrics
    cov_delta = _delta_str(prev_cov, m.mean_coverage)
    prec_delta = _delta_str(prev_prec, m.mean_precision)

    _rich()
    _rich(f"  ━━━ Step {step}/{max_steps} ━━━{'━' * (W - 20)}")
    _rich(f"  Action:  {action_str}")
    _rich(f"  Reward:  {reward:+.2f}")
    _rich(f"    coverage   {_bar(m.mean_coverage)} {m.mean_coverage:.3f}  ({cov_delta})")
    _rich(f"    precision  {_bar(m.mean_precision)} {m.mean_precision:.3f}  ({prec_delta})")
    if m.multi_hop_coverage is not None:
        _rich(f"    multi_hop  {_bar(m.multi_hop_coverage)} {m.multi_hop_coverage:.3f}")


def _show_summary(
    success: bool,
    score: float,
    steps: int,
    max_steps: int,
    rewards: List[float],
    initial_obs: Optional[RAGDebugObservation],
    final_obs: Optional[RAGDebugObservation],
) -> None:
    _rich()
    _rich(f"{'═' * W}")
    result_label = "SUCCESS" if success else "FAILURE"
    _rich(f"  RESULT: {result_label}    Score: {score:.3f}    Steps: {steps}/{max_steps}")
    _rich(f"{'═' * W}")

    if final_obs is not None and initial_obs is not None:
        mi = initial_obs.metrics
        mf = final_obs.metrics

        _rich(f"  METRIC TRAJECTORY")
        _rich(f"    coverage   {mi.mean_coverage:.3f}  -->  {mf.mean_coverage:.3f}  "
              f"({_delta_str(mi.mean_coverage, mf.mean_coverage)})")
        _rich(f"    precision  {mi.mean_precision:.3f}  -->  {mf.mean_precision:.3f}  "
              f"({_delta_str(mi.mean_precision, mf.mean_precision)})")
        if mf.multi_hop_coverage is not None:
            mhi = mi.multi_hop_coverage or 0.0
            _rich(f"    multi_hop  {mhi:.3f}  -->  {mf.multi_hop_coverage:.3f}  "
                  f"({_delta_str(mhi, mf.multi_hop_coverage)})")
        _rich()

        ci = initial_obs.pipeline_config
        cf = final_obs.pipeline_config
        changes = []
        if ci.chunk_size != cf.chunk_size:
            changes.append(f"chunk_size {ci.chunk_size} -> {cf.chunk_size}")
        if ci.chunk_overlap != cf.chunk_overlap:
            changes.append(f"overlap {ci.chunk_overlap} -> {cf.chunk_overlap}")
        if ci.similarity_threshold != cf.similarity_threshold:
            changes.append(f"threshold {ci.similarity_threshold} -> {cf.similarity_threshold}")
        if ci.top_k != cf.top_k:
            changes.append(f"top_k {ci.top_k} -> {cf.top_k}")
        if ci.embedding_model != cf.embedding_model:
            changes.append(f"model {ci.embedding_model.value} -> {cf.embedding_model.value}")
        if ci.use_reranking != cf.use_reranking:
            changes.append(f"reranking {'on' if ci.use_reranking else 'off'} -> {'on' if cf.use_reranking else 'off'}")
        if ci.context_window_limit != cf.context_window_limit:
            changes.append(f"context_limit {ci.context_window_limit} -> {cf.context_window_limit}")

        if changes:
            _rich(f"  CONFIG CHANGES")
            for c in changes:
                _rich(f"    {c}")
        else:
            _rich(f"  CONFIG CHANGES  (none)")

    _rich()
    rewards_str = "  ".join(f"{r:+.2f}" for r in rewards)
    _rich(f"  REWARDS  [{rewards_str}]")
    _rich(f"{'═' * W}")
    _rich()


# ─── Mandatory stdout logging ──────────────────────────────────────────────────

def _one_line(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={_one_line(task)} env={_one_line(env)} model={_one_line(model)}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    done_val = str(done).lower()
    error_val = _one_line(error) if error else "null"
    action_val = _one_line(action)
    print(
        f"[STEP] step={step} action={action_val} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ─── Agent logic ────────────────────────────────────────────────────────────────

def _format_observation(obs: RAGDebugObservation, last_reward: Optional[float] = None) -> str:
    cfg = obs.pipeline_config
    m = obs.metrics

    lines = []
    if last_reward is not None:
        lines.append(f"Previous action reward: {last_reward:+.3f}")
        lines.append("")
    lines += [
        f"Task ID: {obs.task_id}",
        f"Task: {obs.task_description}",
        f"Step: {obs.steps_taken}/{obs.max_steps}",
        "",
        "## Current Config",
        f"  chunk_size={cfg.chunk_size}, overlap={cfg.chunk_overlap}, threshold={cfg.similarity_threshold}",
        f"  top_k={cfg.top_k}, model={cfg.embedding_model.value}, reranking={'on' if cfg.use_reranking else 'off'}, context_limit={cfg.context_window_limit}",
        "",
        "## Aggregate Metrics",
        f"  coverage={m.mean_coverage:.3f}, precision={m.mean_precision:.3f}",
        f"  empty_retrievals={m.n_empty_retrievals}, context_overflows={m.n_context_overflows}",
    ]
    if m.multi_hop_coverage is not None:
        lines.append(f"  multi_hop_coverage={m.multi_hop_coverage:.3f}")

    lines.append("")
    lines.append("## Per-Query Results")
    all_scores = []
    for qr in obs.query_results:
        mh_flag = " [MULTI-HOP]" if qr.is_multi_hop else ""
        score_info = ""
        if qr.retrieval_scores:
            all_scores.extend(qr.retrieval_scores)
            s_min = min(qr.retrieval_scores)
            s_max = max(qr.retrieval_scores)
            s_mean = sum(qr.retrieval_scores) / len(qr.retrieval_scores)
            score_info = f"  scores: min={s_min:.3f} max={s_max:.3f} mean={s_mean:.3f}"
        lines.append(
            f"  q{qr.query_id}{mh_flag}: cov={qr.coverage_score:.3f}, prec={qr.precision_score:.3f}, n={qr.n_retrieved}{score_info}"
        )
        if qr.n_retrieved == 0:
            lines.append(f"    !! EMPTY — no chunks above threshold")

    # Score distribution summary (helps diagnose wrong model / compression)
    if all_scores:
        import statistics
        s_std = statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0
        lines.append(f"\n## Score Distribution: mean={sum(all_scores)/len(all_scores):.3f}, std={s_std:.3f}")
        if s_std < 0.05:
            lines.append("  WARNING: Scores are tightly compressed — possible wrong embedding model")

    # Diagnostic hints from environment
    hints = getattr(obs, "diagnostic_hints", [])
    if hints:
        lines.append("\n## Diagnostic Hints")
        for h in hints:
            lines.append(f"  - {h}")

    # Action error feedback
    err = getattr(obs, "last_action_error", None)
    if err:
        lines.append(f"\n## Last Action Error: {err}")

    # Estimated task score so the agent knows whether to submit
    efficiency = max(0.0, 1.0 - obs.steps_taken / max(obs.max_steps, 1))
    if obs.task_id in (1, 2):
        est_score = 0.60 * m.mean_coverage + 0.25 * m.mean_precision + 0.15 * efficiency
        lines.append(f"\n## Estimated Task Score: {est_score:.3f} (need >= 0.75, aim for >= 0.78)")
    else:
        mh = m.multi_hop_coverage or 0.0
        est_score = 0.55 * m.mean_coverage + 0.25 * m.mean_precision + 0.20 * mh
        lines.append(f"\n## Estimated Task Score: {est_score:.3f} (need >= 0.70, multi_hop > 0.60)")

    lines.append("\nAnalyze the situation and return your reasoning followed by a JSON action.")
    return "\n".join(lines)


def _extract_last_action_error(obs: RAGDebugObservation) -> Optional[str]:
    err = getattr(obs, "last_action_error", None)
    if err is None:
        return None
    return str(err)


def _estimate_task_score(obs: RAGDebugObservation) -> float:
    m = obs.metrics
    efficiency = max(0.0, 1.0 - obs.steps_taken / max(obs.max_steps, 1))
    if obs.task_id in (1, 2):
        score = 0.60 * m.mean_coverage + 0.25 * m.mean_precision + 0.15 * efficiency
    else:
        mh = m.multi_hop_coverage or 0.0
        score = 0.55 * m.mean_coverage + 0.25 * m.mean_precision + 0.20 * mh
    return min(max(score, 0.0), 1.0)


def _submit_ready(obs: RAGDebugObservation) -> bool:
    m = obs.metrics
    est_score = _estimate_task_score(obs)
    if obs.task_id in (1, 2):
        return (
            est_score >= 0.77
            and m.mean_coverage >= 0.80
            and m.mean_precision >= 0.40
            and m.n_empty_retrievals == 0
        )
    mh = m.multi_hop_coverage or 0.0
    return est_score >= 0.78 and mh > 0.60 and m.n_empty_retrievals == 0


def _precision_recovery_action(obs: RAGDebugObservation) -> RAGDebugAction:
    """Recover precision after high-coverage / low-precision plateaus."""
    metrics = obs.metrics
    cfg = obs.pipeline_config

    if not cfg.use_reranking:
        return RAGDebugAction(
            action_type=ActionType.TOGGLE_RERANKING,
            params={"enabled": True},
        )

    if cfg.similarity_threshold < 0.62:
        bump = 0.12 if metrics.mean_precision < 0.30 else 0.08
        return RAGDebugAction(
            action_type=ActionType.ADJUST_THRESHOLD,
            params={"value": round(min(0.65, cfg.similarity_threshold + bump), 2)},
        )

    if cfg.top_k > 7:
        return RAGDebugAction(
            action_type=ActionType.ADJUST_TOP_K,
            params={"value": max(5, cfg.top_k - 2)},
        )

    return RAGDebugAction(
        action_type=ActionType.ADJUST_THRESHOLD,
        params={"value": round(min(0.70, cfg.similarity_threshold + 0.05), 2)},
    )


def _task12_policy_action(obs: RAGDebugObservation) -> RAGDebugAction:
    """Deterministic recovery policy tuned for Tasks 1/2 stability."""
    metrics = obs.metrics
    cfg = obs.pipeline_config
    est_score = _estimate_task_score(obs)

    if _submit_ready(obs):
        return RAGDebugAction(action_type=ActionType.SUBMIT, params={})

    # 1) Empty retrievals are highest-priority blockers.
    if metrics.n_empty_retrievals > 0:
        if cfg.similarity_threshold > 0.10:
            return RAGDebugAction(
                action_type=ActionType.ADJUST_THRESHOLD,
                params={"value": round(max(0.05, cfg.similarity_threshold - 0.10), 2)},
            )
        if cfg.top_k < 18:
            return RAGDebugAction(
                action_type=ActionType.ADJUST_TOP_K,
                params={"value": min(20, cfg.top_k + 4)},
            )

    # 2) Context overflow must be resolved before precision can stabilize.
    if metrics.n_context_overflows > 0:
        if cfg.context_window_limit < 16384:
            return RAGDebugAction(
                action_type=ActionType.ADJUST_CONTEXT_LIMIT,
                params={"value": min(16384, cfg.context_window_limit * 2)},
            )
        if cfg.top_k > 8:
            return RAGDebugAction(
                action_type=ActionType.ADJUST_TOP_K,
                params={"value": max(6, cfg.top_k - 2)},
            )

    # 3) Low coverage recovery.
    if metrics.mean_coverage < 0.75:
        if cfg.top_k < 16:
            return RAGDebugAction(
                action_type=ActionType.ADJUST_TOP_K,
                params={"value": min(18, cfg.top_k + 3)},
            )
        if cfg.similarity_threshold > 0.12:
            return RAGDebugAction(
                action_type=ActionType.ADJUST_THRESHOLD,
                params={"value": round(max(0.05, cfg.similarity_threshold - 0.08), 2)},
            )
        if cfg.chunk_size > 384:
            return RAGDebugAction(
                action_type=ActionType.ADJUST_CHUNK_SIZE,
                params={"value": max(256, int(cfg.chunk_size * 0.75))},
            )

    # 4) High coverage but low precision plateau.
    if metrics.mean_coverage >= 0.80 and metrics.mean_precision < 0.40:
        return _precision_recovery_action(obs)

    # 5) If close to threshold and end of budget, submit to reduce variance.
    if (
        obs.steps_taken >= obs.max_steps - 2
        and est_score >= 0.73
        and metrics.n_empty_retrievals == 0
        and metrics.mean_coverage >= 0.75
    ):
        return RAGDebugAction(action_type=ActionType.SUBMIT, params={})

    # 6) Generic cleanup before submit readiness.
    if not cfg.use_reranking and metrics.mean_precision < 0.45:
        return RAGDebugAction(
            action_type=ActionType.TOGGLE_RERANKING,
            params={"enabled": True},
        )
    if metrics.mean_precision < 0.45 and cfg.similarity_threshold < 0.60:
        return RAGDebugAction(
            action_type=ActionType.ADJUST_THRESHOLD,
            params={"value": round(min(0.65, cfg.similarity_threshold + 0.08), 2)},
        )

    return RAGDebugAction(
        action_type=ActionType.ADJUST_TOP_K,
        params={"value": min(16, cfg.top_k + 1)},
    )


def _stabilize_action(obs: RAGDebugObservation, action: RAGDebugAction) -> RAGDebugAction:
    """Apply conservative policy guardrails to reduce baseline variance."""
    metrics = obs.metrics
    cfg = obs.pipeline_config

    # Never allow premature submit; use deterministic fallback policy instead.
    if action.action_type == ActionType.SUBMIT and not _submit_ready(obs):
        return _heuristic_action(obs)

    if obs.task_id in (1, 2):
        planner_action = _task12_policy_action(obs)

        # Tasks 1/2 are config faults; model swaps are usually wasted steps.
        if action.action_type in (
            ActionType.SWAP_EMBEDDING_MODEL,
            ActionType.REWRITE_QUERY,
            ActionType.ADJUST_CHUNK_OVERLAP,
        ):
            return planner_action

        # In high-coverage / low-precision states, avoid actions that commonly worsen precision.
        if metrics.mean_coverage >= 0.85 and metrics.mean_precision < 0.35:
            if action.action_type == ActionType.ADJUST_TOP_K:
                try:
                    requested_top_k = int(action.params.get("value", cfg.top_k))
                except Exception:
                    requested_top_k = cfg.top_k
                if requested_top_k >= cfg.top_k:
                    return planner_action

            if action.action_type == ActionType.ADJUST_THRESHOLD:
                try:
                    requested_threshold = float(action.params.get("value", cfg.similarity_threshold))
                except Exception:
                    requested_threshold = cfg.similarity_threshold
                if requested_threshold < cfg.similarity_threshold:
                    return planner_action

        # When coverage is clearly low, prefer planner's deterministic recovery path.
        if metrics.mean_coverage < 0.50 and action.action_type in (
            ActionType.ADJUST_THRESHOLD,
            ActionType.ADJUST_TOP_K,
            ActionType.ADJUST_CHUNK_SIZE,
            ActionType.TOGGLE_RERANKING,
        ):
            return planner_action

        # If we're at the end of the budget and already good enough, submit deterministically.
        if obs.steps_taken >= obs.max_steps - 1 and _submit_ready(obs):
            return RAGDebugAction(action_type=ActionType.SUBMIT, params={})

    return action


def _heuristic_action(obs: RAGDebugObservation) -> RAGDebugAction:
    """Smart heuristic fallback when LLM parsing fails."""
    metrics = obs.metrics
    cfg = obs.pipeline_config

    if obs.task_id in (1, 2):
        return _task12_policy_action(obs)

    # Priority 1: Fix empty retrievals (most impactful)
    if metrics.n_empty_retrievals > 0:
        if cfg.similarity_threshold > 0.15:
            return RAGDebugAction(
                action_type=ActionType.ADJUST_THRESHOLD,
                params={"value": round(max(0.05, cfg.similarity_threshold - 0.15), 2)},
            )
        if cfg.top_k < 20:
            return RAGDebugAction(
                action_type=ActionType.ADJUST_TOP_K,
                params={"value": min(30, cfg.top_k + 5)},
            )

    # Priority 2: Wrong embedding model (Task 3 specific)
    if obs.task_id == 3 and cfg.embedding_model.value != "medical":
        return RAGDebugAction(
            action_type=ActionType.SWAP_EMBEDDING_MODEL,
            params={"model": "medical"},
        )

    # Priority 3: Score compression detection → swap model or enable reranking
    all_scores = [s for qr in obs.query_results for s in qr.retrieval_scores]
    if all_scores and len(all_scores) > 3:
        import statistics
        score_std = statistics.stdev(all_scores)
        if score_std < 0.05 and not cfg.use_reranking:
            return RAGDebugAction(
                action_type=ActionType.TOGGLE_RERANKING,
                params={"enabled": True},
            )

    # Priority 4: Fix context overflow
    if metrics.n_context_overflows > 0:
        if cfg.context_window_limit < 16384:
            return RAGDebugAction(
                action_type=ActionType.ADJUST_CONTEXT_LIMIT,
                params={"value": min(16384, cfg.context_window_limit * 2)},
            )
        elif cfg.top_k > 5:
            return RAGDebugAction(
                action_type=ActionType.ADJUST_TOP_K,
                params={"value": max(5, cfg.top_k - 3)},
            )

    # Priority 5: Tasks 1/2 precision recovery once coverage is already strong.
    if obs.task_id in (1, 2) and metrics.mean_coverage >= 0.82 and metrics.mean_precision < 0.40:
        return _precision_recovery_action(obs)

    # Priority 6: High coverage but low precision → raise threshold to filter irrelevant chunks
    # This handles the top_k_too_small recovery: after increasing top_k to fix coverage,
    # the threshold is too permissive and lets in too many irrelevant chunks.
    if metrics.mean_coverage > 0.85 and metrics.mean_precision < 0.35:
        new_threshold = round(min(0.70, cfg.similarity_threshold + 0.12), 2)
        if new_threshold > cfg.similarity_threshold:
            return RAGDebugAction(
                action_type=ActionType.ADJUST_THRESHOLD,
                params={"value": new_threshold},
            )

    # Priority 7: Low coverage → increase top_k
    if metrics.mean_coverage < 0.6 and cfg.top_k < 20:
        return RAGDebugAction(
            action_type=ActionType.ADJUST_TOP_K,
            params={"value": min(30, cfg.top_k + 5)},
        )

    # Priority 8: Enable reranking if precision is poor
    if metrics.mean_precision < 0.4 and not cfg.use_reranking:
        return RAGDebugAction(
            action_type=ActionType.TOGGLE_RERANKING,
            params={"enabled": True},
        )

    # Priority 9: Chunk size reduction for large chunks
    if metrics.mean_coverage < 0.6 and cfg.chunk_size > 384:
        return RAGDebugAction(
            action_type=ActionType.ADJUST_CHUNK_SIZE,
            params={"value": max(256, cfg.chunk_size // 2)},
        )

    # Priority 10: Reduce chunk size if it's large and coverage is low
    if metrics.mean_coverage < 0.7 and cfg.chunk_size > 300:
        return RAGDebugAction(
            action_type=ActionType.ADJUST_CHUNK_SIZE,
            params={"value": max(256, cfg.chunk_size // 2)},
        )

    # Priority 11: Lower threshold further if coverage is still not great
    if metrics.mean_coverage < 0.7 and cfg.similarity_threshold > 0.10:
        return RAGDebugAction(
            action_type=ActionType.ADJUST_THRESHOLD,
            params={"value": round(max(0.05, cfg.similarity_threshold - 0.10), 2)},
        )

    # Priority 12: Submit once deterministic readiness checks pass.
    if _submit_ready(obs):
        return RAGDebugAction(action_type=ActionType.SUBMIT, params={})

    # Fallback: if coverage is OK but precision still bad, keep raising threshold
    if metrics.mean_precision < 0.5 and cfg.similarity_threshold < 0.65:
        return RAGDebugAction(
            action_type=ActionType.ADJUST_THRESHOLD,
            params={"value": round(min(0.65, cfg.similarity_threshold + 0.10), 2)},
        )

    # Fallback: increase top_k if coverage is still low
    if metrics.mean_coverage < 0.85 and cfg.top_k < 30:
        return RAGDebugAction(
            action_type=ActionType.ADJUST_TOP_K,
            params={"value": min(30, cfg.top_k + 5)},
        )

    # Last resort submit
    return RAGDebugAction(action_type=ActionType.SUBMIT, params={})


def _parse_action_json(raw_text: str, obs: RAGDebugObservation) -> Tuple[RAGDebugAction, str]:
    text = raw_text.strip()
    # Strip markdown code fences
    if "```" in text:
        text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("```"))

    action = None
    # Search lines bottom-up for a JSON action (reasoning comes before the JSON)
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and "action_type" in line:
            try:
                data = json.loads(line)
                action = RAGDebugAction(
                    action_type=ActionType(data["action_type"]),
                    params=data.get("params", {}),
                )
                break
            except Exception:
                continue

    # Fallback: try parsing the full text as JSON
    if action is None:
        try:
            data = json.loads(text)
            action = RAGDebugAction(
                action_type=ActionType(data["action_type"]),
                params=data.get("params", {}),
            )
        except Exception:
            action = _heuristic_action(obs)

    action_str = json.dumps(
        {"action_type": action.action_type.value, "params": action.params},
        separators=(",", ":"),
    )
    return action, action_str


def _compute_score(obs: RAGDebugObservation) -> float:
    m = obs.metrics
    efficiency = max(0.0, 1.0 - obs.steps_taken / max(obs.max_steps, 1))

    if obs.task_id == 3:
        mh = m.multi_hop_coverage or 0.0
        score = 0.55 * m.mean_coverage + 0.25 * m.mean_precision + 0.20 * mh
    else:
        score = 0.60 * m.mean_coverage + 0.25 * m.mean_precision + 0.15 * efficiency

    return min(max(score, 0.0), 1.0)


def _compute_success(obs: RAGDebugObservation, score: float) -> bool:
    if obs.task_id in (1, 2):
        return score >= 0.75
    mh = obs.metrics.multi_hop_coverage or 0.0
    return score >= 0.70 and mh > 0.60


async def _connect_env() -> RAGDebugEnv:
    if LOCAL_IMAGE_NAME:
        return await RAGDebugEnv.from_docker_image(LOCAL_IMAGE_NAME)
    return RAGDebugEnv(base_url=SERVER_URL)


async def _next_action(
    client: OpenAI,
    obs: RAGDebugObservation,
    messages: List[dict],
    last_reward: Optional[float] = None,
) -> Tuple[RAGDebugAction, str]:
    prompt = _format_observation(obs, last_reward=last_reward)
    messages.append({"role": "user", "content": prompt})
    last_exc = None
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            raw = completion.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": raw})
            parsed_action, _ = _parse_action_json(raw, obs)
            action = _stabilize_action(obs, parsed_action)
            action_str = json.dumps(
                {"action_type": action.action_type.value, "params": action.params},
                separators=(",", ":"),
            )
            return action, action_str
        except Exception as exc:
            last_exc = exc
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
                continue
    # All retries exhausted — raise so the caller sees the real error
    raise RuntimeError(f"LLM failed after 3 attempts: {last_exc}") from last_exc


# ─── Main ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    task_name = f"task_{TASK_ID}"

    rewards: List[float] = []
    messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    steps_taken = 0
    score = 0.0
    success = False
    fatal_error: Optional[str] = None

    env: Optional[RAGDebugEnv] = None
    obs: Optional[RAGDebugObservation] = None
    initial_obs: Optional[RAGDebugObservation] = None
    max_steps = MAX_STEPS_OVERRIDE

    _show_banner()
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        if not API_KEY:
            raise RuntimeError("Missing HF_TOKEN/API_KEY for OpenAI client")

        llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        env = await _connect_env()

        result = await env.reset(task_id=TASK_ID)
        obs = result.observation
        initial_obs = obs
        max_steps = min(obs.max_steps, MAX_STEPS_OVERRIDE)

        _show_task_info(obs)
        _show_initial_state(obs)

        prev_cov = obs.metrics.mean_coverage
        prev_prec = obs.metrics.mean_precision

        for step in range(1, max_steps + 1):
            if result.done:
                break

            last_reward = rewards[-1] if rewards else None
            action, action_str = await _next_action(llm_client, obs, messages, last_reward=last_reward)

            try:
                result = await env.step(action)
            except Exception as exc:
                log_step(step=step, action=action_str, reward=0.0, done=False, error=str(exc))
                break

            obs = result.observation
            reward = float(result.reward or 0.0)
            done = bool(result.done)
            step_error = _extract_last_action_error(obs)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=step_error)

            _show_step(step, max_steps, action_str, reward, obs, prev_cov, prev_prec)

            prev_cov = obs.metrics.mean_coverage
            prev_prec = obs.metrics.mean_precision

            if done:
                break

        if obs is not None:
            score = _compute_score(obs)
            success = _compute_success(obs, score)

    except Exception as exc:
        # Never let runtime faults bubble out as a non-zero exit code in evaluator mode.
        fatal_error = _one_line(str(exc))
        _rich(f"  [FATAL] {fatal_error}")
        if obs is not None:
            score = _compute_score(obs)
            success = _compute_success(obs, score)
        else:
            score = 0.0
            success = False

    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

        try:
            _show_summary(
                success=success,
                score=score,
                steps=steps_taken,
                max_steps=max_steps,
                rewards=rewards,
                initial_obs=initial_obs,
                final_obs=obs,
            )
        except Exception as summary_exc:
            _rich(f"  [SUMMARY ERROR] {_one_line(str(summary_exc))}")

        if fatal_error:
            _rich(f"  [END ERROR] {fatal_error}")


if __name__ == "__main__":
    asyncio.run(main())
