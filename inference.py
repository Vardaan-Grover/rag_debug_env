"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL    The API endpoint for the LLM.
    MODEL_NAME      The model identifier to use for inference.
    API_KEY         Your API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment
                     if you are using from_docker_image().

- Defaults are set only for API_BASE_URL and MODEL_NAME:
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")

- The inference script must be named inference.py and placed in the root directory.
- OpenAI Client is used for all LLM calls.

STDOUT FORMAT
- The script emits exactly three line types to stdout, in this order:

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
    - Each task returns score in [0, 1].
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

from client import RAGDebugEnv
from models import ActionType, RAGDebugAction, RAGDebugObservation


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required for authentication")

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

SERVER_URL = os.getenv("SERVER_URL", "http://localhost:7860")
BENCHMARK = os.getenv("RAG_DEBUG_BENCHMARK", "rag_debug_env")
DEFAULT_TASK_IDS = (1, 2, 3)
MAX_STEPS_OVERRIDE = int(os.getenv("RAG_DEBUG_MAX_STEPS", "10"))

TEMPERATURE = float(os.getenv("RAG_DEBUG_TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("RAG_DEBUG_MAX_TOKENS", "256"))
HUMAN_LOGS_ENABLED = os.getenv("RAG_DEBUG_HUMAN_LOGS", "1").strip().lower() not in {
    "0",
    "false",
    "off",
    "no",
}

REQUIRED_ENV_VARS = ("API_BASE_URL", "HF_TOKEN", "MODEL_NAME")


SYSTEM_PROMPT = """You are an expert RAG retrieval debugger.
Goal: maximize final task score within the available step budget.

You will receive only observed state (metrics, config, recent history, and errors).
Infer root causes from these observations without assuming any known fault labels.

Cross-step policy:
1) compare current state against recent_history,
2) avoid repeating actions that produced no gain,
3) prioritize actions with measurable improvement,
4) submit only when metrics indicate likely success.

Output contract:
- return exactly one JSON object and nothing else,
- no markdown, no prose, no code fences,
- schema: {"action_type":"<type>","params":{...}}.

Valid action_type values:
- adjust_chunk_size with params {"value": int in [64,2048]}
- adjust_chunk_overlap with params {"value": int in [0,500]}
- adjust_threshold with params {"value": float in [0.0,1.0]}
- adjust_top_k with params {"value": int in [1,50]}
- swap_embedding_model with params {"model": "general"|"medical"|"legal"|"code"}
- toggle_reranking with params {"enabled": bool}
- adjust_context_limit with params {"value": int in [512,16384]}
- rewrite_query with params {"query_id": int, "strategy": "rephrase"}
- submit with params {}
"""


def _validate_required_env_vars() -> None:
    missing: List[str] = []

    for name in REQUIRED_ENV_VARS:
        value = os.getenv(name)
        if value is None or not value.strip():
            missing.append(name)

    api_base_url_value = (API_BASE_URL or "").strip()
    model_name_value = (MODEL_NAME or "").strip()

    placeholder_values: List[str] = []
    if api_base_url_value == "<your-active-endpoint>":
        placeholder_values.append("API_BASE_URL")
    if model_name_value == "<your-active-model>":
        placeholder_values.append("MODEL_NAME")

    if missing or placeholder_values:
        parts: List[str] = []
        if missing:
            parts.append(f"missing={','.join(missing)}")
        if placeholder_values:
            parts.append(f"placeholder={','.join(placeholder_values)}")
        raise EnvironmentError(
            "Required environment configuration is invalid: " + " ".join(parts)
        )


def _parse_task_ids(value: str) -> Tuple[int, ...]:
    """
    Parse RAG_DEBUG_TASK_IDS.

    Accepted values:
    - "all" (default): runs tasks 1,2,3
    - Comma list: "1", "1,2", "2,3", "1,3", "1,2,3"
    """
    raw = (value or "all").strip().lower()
    if raw in {"", "all"}:
        return DEFAULT_TASK_IDS

    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if not tokens:
        raise ValueError(
            "RAG_DEBUG_TASK_IDS is empty. Use 'all' or a comma list like '1,2,3'."
        )

    task_ids: List[int] = []
    seen: set[int] = set()
    for token in tokens:
        if token not in {"1", "2", "3"}:
            raise ValueError(
                "RAG_DEBUG_TASK_IDS contains invalid task id "
                f"'{token}'. Allowed values are 1,2,3 or 'all'."
            )
        task_id = int(token)
        if task_id not in seen:
            seen.add(task_id)
            task_ids.append(task_id)

    return tuple(task_ids)


def _stderr(line: str = "") -> None:
    if HUMAN_LOGS_ENABLED:
        print(line, file=sys.stderr, flush=True)


def _progress_bar(value: float, width: int = 26) -> str:
    clamped = max(0.0, min(1.0, float(value)))
    filled = int(round(clamped * width))
    return "#" * filled + "-" * (width - filled)


def _fmt_opt_float(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _fmt_delta(current: float, previous: Optional[float]) -> str:
    if previous is None:
        return "n/a"
    return f"{(current - previous):+.3f}"


def _log_episode_start(obs: RAGDebugObservation, max_steps: int) -> None:
    cfg = obs.pipeline_config
    m = obs.metrics
    score_est = _estimated_score(obs)

    _stderr()
    _stderr("=" * 76)
    _stderr(
        f"RAG Debug Run | task=task_{obs.task_id} | env={BENCHMARK} | model={MODEL_NAME}"
    )
    _stderr(f"Description: {_as_single_line(obs.task_description)}")
    _stderr(
        "Config: "
        f"chunk_size={cfg.chunk_size} overlap={cfg.chunk_overlap} "
        f"threshold={cfg.similarity_threshold:.2f} top_k={cfg.top_k} "
        f"model={cfg.embedding_model.value} reranking={_bool_text(cfg.use_reranking)} "
        f"context_limit={cfg.context_window_limit}"
    )
    _stderr(f"Budget: steps=0/{max_steps}")
    _stderr(
        f"coverage  [{_progress_bar(m.mean_coverage)}] {m.mean_coverage:.3f}"
    )
    _stderr(
        f"precision [{_progress_bar(m.mean_precision)}] {m.mean_precision:.3f}"
    )
    if m.multi_hop_coverage is not None:
        _stderr(
            f"multi_hop [{_progress_bar(m.multi_hop_coverage)}] {m.multi_hop_coverage:.3f}"
        )
    _stderr(
        f"est_score [{_progress_bar(score_est)}] {score_est:.3f} "
        f"empty={m.n_empty_retrievals} overflow={m.n_context_overflows}"
    )
    _stderr("=" * 76)


def _log_step_details(
    step: int,
    max_steps: int,
    action_text: str,
    reward: float,
    done: bool,
    obs: RAGDebugObservation,
    prev_coverage: Optional[float],
    prev_precision: Optional[float],
) -> None:
    m = obs.metrics
    step_progress = step / max(1, max_steps)
    score_est = _estimated_score(obs)

    _stderr()
    _stderr("-" * 76)
    _stderr(
        f"Step {step:02d}/{max_steps:02d} "
        f"[{_progress_bar(step_progress, width=18)}] reward={reward:+.2f} done={_bool_text(done)}"
    )
    _stderr(f"action: {action_text}")
    _stderr(
        f"coverage  [{_progress_bar(m.mean_coverage)}] {m.mean_coverage:.3f} "
        f"({_fmt_delta(m.mean_coverage, prev_coverage)})"
    )
    _stderr(
        f"precision [{_progress_bar(m.mean_precision)}] {m.mean_precision:.3f} "
        f"({_fmt_delta(m.mean_precision, prev_precision)})"
    )
    if m.multi_hop_coverage is not None:
        _stderr(
            f"multi_hop [{_progress_bar(m.multi_hop_coverage)}] {m.multi_hop_coverage:.3f}"
        )
    _stderr(
        f"est_score [{_progress_bar(score_est)}] {score_est:.3f} "
        f"empty={m.n_empty_retrievals} overflow={m.n_context_overflows}"
    )

    if obs.last_action_error:
        _stderr(f"last_action_error: {str(obs.last_action_error).replace(chr(10), ' ')}")
    _stderr("-" * 76)


def _log_final_summary(
    success: bool,
    score: float,
    steps_taken: int,
    max_steps: int,
    rewards: List[float],
    initial_obs: Optional[RAGDebugObservation],
    obs: Optional[RAGDebugObservation],
) -> None:
    _stderr()
    _stderr("-" * 76)
    _stderr(
        f"Final | success={_bool_text(success)} "
        f"steps={steps_taken}/{max_steps} score={score:.3f}"
    )
    if rewards:
        avg_reward = sum(rewards) / len(rewards)
        _stderr(
            f"rewards: count={len(rewards)} avg={avg_reward:.3f} "
            f"min={min(rewards):.3f} max={max(rewards):.3f}"
        )
    else:
        _stderr("rewards: count=0")

    if obs is not None:
        m = obs.metrics
        _stderr(
            f"final_metrics: coverage={m.mean_coverage:.3f} precision={m.mean_precision:.3f} "
            f"multi_hop={_fmt_opt_float(m.multi_hop_coverage)} "
            f"empty={m.n_empty_retrievals} overflow={m.n_context_overflows}"
        )

    if initial_obs is not None and obs is not None:
        m0 = initial_obs.metrics
        m1 = obs.metrics
        _stderr("metric_change:")
        _stderr(
            f"  coverage:  {m0.mean_coverage:.3f} -> {m1.mean_coverage:.3f} "
            f"({_fmt_delta(m1.mean_coverage, m0.mean_coverage)})"
        )
        _stderr(
            f"  precision: {m0.mean_precision:.3f} -> {m1.mean_precision:.3f} "
            f"({_fmt_delta(m1.mean_precision, m0.mean_precision)})"
        )
        if m0.multi_hop_coverage is not None or m1.multi_hop_coverage is not None:
            mh0 = m0.multi_hop_coverage or 0.0
            mh1 = m1.multi_hop_coverage or 0.0
            _stderr(
                f"  multi_hop: {mh0:.3f} -> {mh1:.3f} "
                f"({_fmt_delta(mh1, mh0)})"
            )
        _stderr(
            f"  empty:     {m0.n_empty_retrievals} -> {m1.n_empty_retrievals} "
            f"({m1.n_empty_retrievals - m0.n_empty_retrievals:+d})"
        )
        _stderr(
            f"  overflow:  {m0.n_context_overflows} -> {m1.n_context_overflows} "
            f"({m1.n_context_overflows - m0.n_context_overflows:+d})"
        )
        start_score = _estimated_score(initial_obs)
        end_score = _estimated_score(obs)
        _stderr(
            f"  est_score: {start_score:.3f} -> {end_score:.3f} "
            f"({_fmt_delta(end_score, start_score)})"
        )
    _stderr("-" * 76)


def _as_single_line(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def log_start(task: str, env: str, model: str) -> None:
    print(
        f"[START] task={_as_single_line(task)} env={_as_single_line(env)} model={_as_single_line(model)}",
        flush=True,
    )


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    action_text = _as_single_line(action)
    error_text = "null" if error is None else str(error).replace("\n", "\\n")
    print(
        f"[STEP] step={step} action={action_text} reward={reward:.2f} done={_bool_text(done)} error={error_text}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_text = ",".join(f"{r:.2f}" for r in rewards)
    clipped_score = min(max(float(score), 0.0), 1.0)
    print(
        f"[END] success={_bool_text(success)} steps={steps} score={clipped_score:.2f} rewards={rewards_text}",
        flush=True,
    )


def _build_observation_prompt(
    obs: RAGDebugObservation,
    previous_reward: Optional[float],
    recent_history: List[Dict[str, Any]],
) -> str:
    m = obs.metrics
    cfg = obs.pipeline_config
    query_brief = [
        {
            "query_id": q.query_id,
            "n_retrieved": q.n_retrieved,
            "coverage": round(q.coverage_score, 4),
            "precision": round(q.precision_score, 4),
            "is_multi_hop": q.is_multi_hop,
        }
        for q in obs.query_results
    ]

    payload: Dict[str, Any] = {
        "task_id": obs.task_id,
        "steps_taken": obs.steps_taken,
        "max_steps": obs.max_steps,
        "pipeline_config": {
            "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap,
            "similarity_threshold": cfg.similarity_threshold,
            "top_k": cfg.top_k,
            "embedding_model": cfg.embedding_model.value,
            "use_reranking": cfg.use_reranking,
            "context_window_limit": cfg.context_window_limit,
        },
        "metrics": {
            "mean_coverage": m.mean_coverage,
            "mean_precision": m.mean_precision,
            "n_empty_retrievals": m.n_empty_retrievals,
            "n_context_overflows": m.n_context_overflows,
            "multi_hop_coverage": m.multi_hop_coverage,
        },
        "query_results": query_brief,
        "last_action_error": obs.last_action_error,
        "previous_reward": previous_reward,
        "recent_history": recent_history,
    }

    return json.dumps(payload, ensure_ascii=True)


def _clamp_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(low, min(high, parsed))


def _clamp_float(value: Any, low: float, high: float, default: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = default
    return max(low, min(high, parsed))


def _estimated_score(obs: RAGDebugObservation) -> float:
    m = obs.metrics
    if obs.task_id in (1, 2):
        efficiency = max(0.0, 1.0 - obs.steps_taken / max(obs.max_steps, 1))
        score = 0.60 * m.mean_coverage + 0.25 * m.mean_precision + 0.15 * efficiency
    else:
        mh = m.multi_hop_coverage or 0.0
        score = 0.55 * m.mean_coverage + 0.25 * m.mean_precision + 0.20 * mh
    return min(max(score, 0.0), 1.0)


def _is_submit_ready(obs: RAGDebugObservation) -> bool:
    m = obs.metrics
    score = _estimated_score(obs)
    if obs.task_id in (1, 2):
        return (
            score >= 0.77
            and m.mean_coverage >= 0.80
            and m.mean_precision >= 0.35
            and m.n_empty_retrievals == 0
        )
    return (
        score >= 0.72
        and m.mean_coverage >= 0.70
        and m.mean_precision >= 0.30
        and (m.multi_hop_coverage or 0.0) > 0.60
        and m.n_empty_retrievals == 0
    )


def _validate_submit_or_raise(action: RAGDebugAction, obs: RAGDebugObservation) -> None:
    """Reject premature submit actions instead of falling back to heuristic recovery."""
    if action.action_type == ActionType.SUBMIT and not _is_submit_ready(obs):
        raise ValueError("submit requested before readiness criteria were met")


def _sanitize_action(action: RAGDebugAction, obs: RAGDebugObservation) -> RAGDebugAction:
    cfg = obs.pipeline_config
    params = dict(action.params or {})

    if action.action_type == ActionType.ADJUST_CHUNK_SIZE:
        params["value"] = _clamp_int(params.get("value"), 64, 2048, cfg.chunk_size)
    elif action.action_type == ActionType.ADJUST_CHUNK_OVERLAP:
        params["value"] = _clamp_int(params.get("value"), 0, 500, cfg.chunk_overlap)
    elif action.action_type == ActionType.ADJUST_THRESHOLD:
        params["value"] = round(
            _clamp_float(params.get("value"), 0.0, 1.0, cfg.similarity_threshold),
            2,
        )
    elif action.action_type == ActionType.ADJUST_TOP_K:
        params["value"] = _clamp_int(params.get("value"), 1, 50, cfg.top_k)
    elif action.action_type == ActionType.SWAP_EMBEDDING_MODEL:
        model = str(params.get("model", cfg.embedding_model.value)).lower()
        if model not in {"general", "medical", "legal", "code"}:
            model = cfg.embedding_model.value
        params["model"] = model
    elif action.action_type == ActionType.TOGGLE_RERANKING:
        params["enabled"] = bool(params.get("enabled", True))
    elif action.action_type == ActionType.ADJUST_CONTEXT_LIMIT:
        params["value"] = _clamp_int(params.get("value"), 512, 16384, cfg.context_window_limit)
    elif action.action_type == ActionType.REWRITE_QUERY:
        valid_ids = {q.query_id for q in obs.query_results}
        qid = _clamp_int(params.get("query_id"), 0, 10_000, min(valid_ids) if valid_ids else 0)
        if qid not in valid_ids and valid_ids:
            qid = min(valid_ids)
        params["query_id"] = qid
        params["strategy"] = "rephrase"
    else:
        params = {}

    return RAGDebugAction(action_type=action.action_type, params=params)


def _extract_action_json(raw_text: str) -> Optional[Dict[str, Any]]:
    text = raw_text.strip()
    if not text:
        return None

    if "```" in text:
        text = "\n".join(
            line for line in text.splitlines() if not line.strip().startswith("```")
        ).strip()

    for line in reversed(text.splitlines()):
        candidate = line.strip()
        if not (candidate.startswith("{") and "action_type" in candidate):
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return None


def _parse_action(raw_text: str, obs: RAGDebugObservation) -> RAGDebugAction:
    payload = _extract_action_json(raw_text)
    if payload is None:
        raise ValueError("no valid JSON action found in model output")

    try:
        action = RAGDebugAction(
            action_type=ActionType(str(payload.get("action_type", "submit"))),
            params=payload.get("params", {}),
        )
    except Exception as exc:
        raise ValueError(f"invalid action payload: {exc}") from exc

    sanitized = _sanitize_action(action, obs)
    _validate_submit_or_raise(sanitized, obs)
    return sanitized


def _action_text(action: RAGDebugAction) -> str:
    return json.dumps(
        {"action_type": action.action_type.value, "params": action.params},
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _compute_score(obs: RAGDebugObservation) -> float:
    return _estimated_score(obs)


def _compute_success(obs: RAGDebugObservation, score: float) -> bool:
    if obs.task_id in (1, 2):
        return score >= 0.75
    return score >= 0.70 and (obs.metrics.multi_hop_coverage or 0.0) > 0.60


async def _connect_env() -> RAGDebugEnv:
    if LOCAL_IMAGE_NAME:
        return await RAGDebugEnv.from_docker_image(LOCAL_IMAGE_NAME)
    return RAGDebugEnv(base_url=SERVER_URL)


async def _choose_action(
    llm_client: OpenAI,
    observation: RAGDebugObservation,
    messages: List[Dict[str, str]],
    previous_reward: Optional[float],
    recent_history: List[Dict[str, Any]],
) -> Tuple[RAGDebugAction, str]:
    user_prompt = _build_observation_prompt(
        observation,
        previous_reward=previous_reward,
        recent_history=recent_history,
    )
    messages.append({"role": "user", "content": user_prompt})

    for attempt in range(3):
        try:
            completion = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            raw = completion.choices[0].message.content or ""
        except Exception as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        messages.append({"role": "assistant", "content": raw})
        try:
            action = _parse_action(raw, observation)
            return action, _action_text(action)
        except ValueError as exc:
            if attempt >= 2:
                raise RuntimeError(f"LLM produced invalid action after 3 attempts: {exc}") from exc
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous output was invalid: "
                        f"{_as_single_line(str(exc))}. "
                        "Return exactly one valid JSON action object with schema "
                        '{"action_type":"<type>","params":{...}} and no extra text.'
                    ),
                }
            )

    raise RuntimeError("failed to produce a valid action")


async def _run_single_task(task_id: int, llm_client: OpenAI) -> None:
    task_name = f"task_{task_id}"

    rewards: List[float] = []
    step_history: List[Dict[str, Any]] = []
    steps_taken = 0
    success = False
    score = 0.0

    env: Optional[RAGDebugEnv] = None
    initial_obs: Optional[RAGDebugObservation] = None
    obs: Optional[RAGDebugObservation] = None
    max_steps = MAX_STEPS_OVERRIDE
    prev_coverage: Optional[float] = None
    prev_precision: Optional[float] = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        env = await _connect_env()

        reset_result = await env.reset(task_id=task_id)
        obs = reset_result.observation
        initial_obs = obs
        done = bool(reset_result.done)

        max_steps = min(obs.max_steps, MAX_STEPS_OVERRIDE)
        _log_episode_start(obs, max_steps=max_steps)
        prev_coverage = obs.metrics.mean_coverage
        prev_precision = obs.metrics.mean_precision
        messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        while not done and steps_taken < max_steps:
            previous_reward = rewards[-1] if rewards else None
            action, action_text = await _choose_action(
                llm_client=llm_client,
                observation=obs,
                messages=messages,
                previous_reward=previous_reward,
                recent_history=step_history[-6:],
            )

            step_result = await env.step(action)
            obs = step_result.observation
            reward = float(step_result.reward or 0.0)
            done = bool(step_result.done)

            steps_taken += 1
            rewards.append(reward)

            last_error = getattr(obs, "last_action_error", None)
            log_step(
                step=steps_taken,
                action=action_text,
                reward=reward,
                done=done,
                error=last_error,
            )
            _log_step_details(
                step=steps_taken,
                max_steps=max_steps,
                action_text=action_text,
                reward=reward,
                done=done,
                obs=obs,
                prev_coverage=prev_coverage,
                prev_precision=prev_precision,
            )
            step_history.append(
                {
                    "step": steps_taken,
                    "action": action_text,
                    "reward": round(reward, 4),
                    "done": done,
                    "error": last_error,
                    "coverage": round(obs.metrics.mean_coverage, 4),
                    "precision": round(obs.metrics.mean_precision, 4),
                    "est_score": round(_estimated_score(obs), 4),
                }
            )
            prev_coverage = obs.metrics.mean_coverage
            prev_precision = obs.metrics.mean_precision

        if obs is not None:
            score = _compute_score(obs)
            success = _compute_success(obs, score)

    except Exception as exc:
        _stderr(
            f"runtime_error: {exc.__class__.__name__}: "
            f"{_as_single_line(str(exc))}"
        )
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
        _log_final_summary(
            success=success,
            score=score,
            steps_taken=steps_taken,
            max_steps=max_steps,
            rewards=rewards,
            initial_obs=initial_obs,
            obs=obs,
        )


async def main() -> None:
    try:
        task_ids = _parse_task_ids(os.getenv("RAG_DEBUG_TASK_IDS", "all"))
    except Exception as exc:
        _stderr(
            f"startup_warning: invalid RAG_DEBUG_TASK_IDS; defaulting to all ({exc.__class__.__name__}: "
            f"{_as_single_line(str(exc))})"
        )
        task_ids = DEFAULT_TASK_IDS

    try:
        _validate_required_env_vars()
    except Exception as exc:
        _stderr(
            f"startup_error: {exc.__class__.__name__}: "
            f"{_as_single_line(str(exc))}"
        )
        for task_id in task_ids:
            log_start(task=f"task_{task_id}", env=BENCHMARK, model=MODEL_NAME or "unknown")
            log_end(success=False, steps=0, score=0.0, rewards=[])
        return

    try:
        # Required by user request: initialize OpenAI client directly from injected env vars.
        llm_client = OpenAI(
            base_url=os.environ["API_BASE_URL"],
            api_key=os.environ["API_KEY"],
        )
    except Exception as exc:
        _stderr(
            f"startup_error: failed to initialize OpenAI client ({exc.__class__.__name__}: "
            f"{_as_single_line(str(exc))})"
        )
        for task_id in task_ids:
            log_start(task=f"task_{task_id}", env=BENCHMARK, model=MODEL_NAME or "unknown")
            log_end(success=False, steps=0, score=0.0, rewards=[])
        return

    _stderr("Planned tasks: " + ", ".join(f"task_{task_id}" for task_id in task_ids))
    for task_id in task_ids:
        try:
            await _run_single_task(task_id=task_id, llm_client=llm_client)
        except Exception as exc:
            _stderr(
                f"task_runtime_error: task_{task_id} failed with {exc.__class__.__name__}: "
                f"{_as_single_line(str(exc))}"
            )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        _stderr(f"fatal_error: {exc.__class__.__name__}: {_as_single_line(str(exc))}")
