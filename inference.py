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


API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")

TASK_ID = int(os.getenv("RAG_DEBUG_TASK", "1"))
BENCHMARK = os.getenv("RAG_DEBUG_BENCHMARK", "rag_debug_env")
MAX_STEPS_OVERRIDE = int(os.getenv("RAG_DEBUG_MAX_STEPS", "10"))

TEMPERATURE = 0.2
MAX_TOKENS = 220

W = 64  # display width for visual output

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert RAG retrieval debugger.

    Output exactly one JSON object on one line:
    {"action_type":"<type>","params":{...}}

    Allowed action_type values:
    adjust_chunk_size, adjust_chunk_overlap, adjust_threshold, adjust_top_k,
    swap_embedding_model, toggle_reranking, adjust_context_limit, rewrite_query, submit

    Param rules:
    - adjust_chunk_size: {"value": int}
    - adjust_chunk_overlap: {"value": int}
    - adjust_threshold: {"value": float}
    - adjust_top_k: {"value": int}
    - swap_embedding_model: {"model": "general"|"medical"|"legal"|"code"}
    - toggle_reranking: {"enabled": bool}
    - adjust_context_limit: {"value": int}
    - rewrite_query: {"query_id": int, "strategy": "rephrase"}
    - submit: {}

    Heuristics:
    - If empty retrievals are high, lower threshold or increase top_k.
    - If context overflows are high, increase context window or reduce top_k.
    - For medical task, consider swapping to medical embedding model.
    - Submit when metrics look stable and good.
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
    initial_obs: RAGDebugObservation,
    final_obs: Optional[RAGDebugObservation],
) -> None:
    _rich()
    _rich(f"{'═' * W}")
    result_label = "SUCCESS" if success else "FAILURE"
    _rich(f"  RESULT: {result_label}    Score: {score:.3f}    Steps: {steps}/{max_steps}")
    _rich(f"{'═' * W}")

    if final_obs is not None:
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

def _format_observation(obs: RAGDebugObservation, history: List[str]) -> str:
    cfg = obs.pipeline_config
    m = obs.metrics

    lines = [
        f"Task ID: {obs.task_id}",
        f"Task: {obs.task_description}",
        f"Step: {obs.steps_taken}/{obs.max_steps}",
        f"Config: chunk_size={cfg.chunk_size}, overlap={cfg.chunk_overlap}, threshold={cfg.similarity_threshold}, top_k={cfg.top_k}, model={cfg.embedding_model.value}, reranking={cfg.use_reranking}, context_limit={cfg.context_window_limit}",
        f"Metrics: coverage={m.mean_coverage:.3f}, precision={m.mean_precision:.3f}, recall={m.mean_recall:.3f}, empty={m.n_empty_retrievals}, overflow={m.n_context_overflows}, multi_hop={m.multi_hop_coverage}",
        "Per query:",
    ]

    for qr in obs.query_results:
        lines.append(
            f"- q{qr.query_id}: cov={qr.coverage_score:.3f}, prec={qr.precision_score:.3f}, n={qr.n_retrieved}, mh={qr.is_multi_hop}"
        )

    if history:
        lines.append("Recent actions:")
        lines.extend(f"- {item}" for item in history[-4:])

    lines.append("Return only one JSON action.")
    return "\n".join(lines)


def _extract_last_action_error(obs: RAGDebugObservation) -> Optional[str]:
    err = getattr(obs, "last_action_error", None)
    if err is None:
        return None
    return str(err)


def _heuristic_action(obs: RAGDebugObservation) -> RAGDebugAction:
    metrics = obs.metrics
    cfg = obs.pipeline_config

    if metrics.n_empty_retrievals > 0 and cfg.similarity_threshold > 0.1:
        return RAGDebugAction(action_type=ActionType.ADJUST_THRESHOLD, params={"value": max(0.05, cfg.similarity_threshold - 0.1)})

    if metrics.n_context_overflows > 0 and cfg.context_window_limit < 16384:
        return RAGDebugAction(action_type=ActionType.ADJUST_CONTEXT_LIMIT, params={"value": min(16384, cfg.context_window_limit + 1024)})

    if obs.task_id == 3 and cfg.embedding_model.value != "medical":
        return RAGDebugAction(action_type=ActionType.SWAP_EMBEDDING_MODEL, params={"model": "medical"})

    if cfg.top_k < 15:
        return RAGDebugAction(action_type=ActionType.ADJUST_TOP_K, params={"value": min(50, cfg.top_k + 3)})

    return RAGDebugAction(action_type=ActionType.SUBMIT, params={})


def _parse_action_json(raw_text: str, obs: RAGDebugObservation) -> Tuple[RAGDebugAction, str]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = "\n".join(line for line in text.splitlines() if not line.startswith("```"))

    try:
        data = json.loads(text)
        action_type = ActionType(data["action_type"])
        params = data.get("params", {})
        action = RAGDebugAction(action_type=action_type, params=params)
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


async def _next_action(client: OpenAI, obs: RAGDebugObservation, history: List[str]) -> Tuple[RAGDebugAction, str]:
    prompt = _format_observation(obs, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = completion.choices[0].message.content or ""
        return _parse_action_json(raw, obs)
    except Exception:
        action = _heuristic_action(obs)
        action_str = json.dumps({"action_type": action.action_type.value, "params": action.params}, separators=(",", ":"))
        return action, action_str


# ─── Main ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    task_name = f"task_{TASK_ID}"

    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    score = 0.0
    success = False

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

            action, action_str = await _next_action(llm_client, obs, history)

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

            history.append(f"{action_str} -> {reward:+.2f}")
            if done:
                break

        if obs is not None:
            score = _compute_score(obs)
            success = _compute_success(obs, score)

    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

        _show_summary(
            success=success,
            score=score,
            steps=steps_taken,
            max_steps=max_steps,
            rewards=rewards,
            initial_obs=initial_obs,
            final_obs=obs,
        )


if __name__ == "__main__":
    asyncio.run(main())
