"""
Inference Script — RAGDebugEnv
==============================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment:
    API_BASE_URL       The API endpoint for the LLM.
    MODEL_NAME         The model identifier to use for inference.
    HF_TOKEN           Your Hugging Face / API key.
    IMAGE_NAME         Docker image name (if using from_docker_image()).
    RAG_DEBUG_TASK     Task difficulty: 1 (easy), 2 (medium), 3 (hard). Default: 1.

- The inference script must be named `inference.py` and placed in the root directory.
- Participants must use the OpenAI client for all LLM calls.

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
    - error is the raw error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - score is in [0, 1].
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from client import RAGDebugEnv
from models import ActionType, RAGDebugAction, RAGDebugObservation

IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_ID = int(os.getenv("RAG_DEBUG_TASK", "1"))
BENCHMARK = os.getenv("RAG_DEBUG_BENCHMARK", "rag_debug_env")
MAX_STEPS = 10
TEMPERATURE = 0.3
MAX_TOKENS = 256

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert RAG pipeline debugger. Your job is to diagnose faults in a
    retrieval-augmented generation pipeline and fix them by adjusting configuration.

    At each step you receive the current pipeline state: configuration knobs, per-query
    retrieval results, and aggregate quality metrics. Your goal is to maximise
    mean_coverage (the fraction of relevant chunks retrieved across all queries).

    Available actions and their parameter schemas:
      adjust_chunk_size      {"value": int}                       # 64–2048
      adjust_chunk_overlap   {"value": int}                       # 0–500, must be < chunk_size
      adjust_threshold       {"value": float}                     # 0.0–1.0
      adjust_top_k           {"value": int}                       # 1–50
      swap_embedding_model   {"model": str}                       # "general"|"medical"|"legal"|"code"
      toggle_reranking       {"enabled": bool}
      adjust_context_limit   {"value": int}                       # 512–16384
      rewrite_query          {"query_id": int, "strategy": str}   # "expand"|"rephrase"|"decompose"
      submit                 {}                                   # end episode (only when satisfied)

    Diagnostic hints:
    - High n_empty_retrievals → threshold is too high; lower it.
    - Low coverage AND low precision → chunk_size is wrong or wrong embedding model.
    - n_context_overflows > 0 → raise context_window_limit.
    - Low precision despite good coverage → threshold is too low; raise it, or enable reranking.
    - For medical/legal corpora → prefer the domain-specific embedding model.

    Respond with ONLY a JSON object on a single line:
      {"action_type": "<type>", "params": {<params>}}
    No explanation. No markdown fences. Just the raw JSON.
    """
).strip()


# ── Logging helpers ───────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Observation formatting ────────────────────────────────────────────────────

def _format_observation(obs: RAGDebugObservation) -> str:
    cfg = obs.pipeline_config
    m = obs.metrics
    lines = [
        f"Task {obs.task_id}: {obs.task_description}",
        f"Step {obs.steps_taken}/{obs.max_steps}",
        "",
        "=== Current Pipeline Config ===",
        f"  chunk_size={cfg.chunk_size}  chunk_overlap={cfg.chunk_overlap}",
        f"  similarity_threshold={cfg.similarity_threshold}  top_k={cfg.top_k}",
        f"  embedding_model={cfg.embedding_model.value}  use_reranking={cfg.use_reranking}",
        f"  context_window_limit={cfg.context_window_limit}",
        "",
        "=== Quality Metrics ===",
        f"  mean_coverage={m.mean_coverage:.3f}  (primary objective — higher is better)",
        f"  mean_precision={m.mean_precision:.3f}",
        f"  mean_recall={m.mean_recall:.3f}",
        f"  n_empty_retrievals={m.n_empty_retrievals}",
        f"  n_context_overflows={m.n_context_overflows}",
    ]
    if m.multi_hop_coverage is not None:
        lines.append(f"  multi_hop_coverage={m.multi_hop_coverage:.3f}")
    lines += [
        "",
        "=== Per-Query Results ===",
    ]
    for qr in obs.query_results:
        tag = " [multi-hop]" if qr.is_multi_hop else ""
        lines.append(
            f"  [q{qr.query_id}] coverage={qr.coverage_score:.3f}  "
            f"precision={qr.precision_score:.3f}  n_retrieved={qr.n_retrieved}{tag}"
        )
    lines += [
        "",
        f"Corpus: {obs.corpus_stats.domain.value}  "
        f"{obs.corpus_stats.n_chunks} chunks  "
        f"{obs.corpus_stats.avg_chunk_tokens} avg tokens/chunk  "
        f"near_duplicates={obs.corpus_stats.has_near_duplicates}",
    ]
    return "\n".join(lines)


# ── Score computation ─────────────────────────────────────────────────────────

def _compute_score(obs: RAGDebugObservation) -> float:
    """Mirror environment's _compute_task_score from the final observation."""
    m = obs.metrics
    efficiency = max(0.0, 1.0 - obs.steps_taken / obs.max_steps)
    if obs.task_id == 3 and m.multi_hop_coverage is not None:
        return 0.55 * m.mean_coverage + 0.25 * m.mean_precision + 0.20 * m.multi_hop_coverage
    return 0.60 * m.mean_coverage + 0.25 * m.mean_precision + 0.15 * efficiency


# ── Action parsing ────────────────────────────────────────────────────────────

def _parse_action(llm_text: str) -> Optional[RAGDebugAction]:
    """Parse LLM JSON response into a RAGDebugAction. Returns None on failure."""
    text = llm_text.strip()
    # Strip markdown fences if the model added them despite instructions
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines() if not line.startswith("```")
        ).strip()
    try:
        data = json.loads(text)
        return RAGDebugAction(
            action_type=ActionType(data["action_type"]),
            params=data.get("params", {}),
        )
    except Exception:
        return None


# ── LLM call ─────────────────────────────────────────────────────────────────

def _get_model_action(
    llm_client: OpenAI,
    obs: RAGDebugObservation,
    history: List[str],
) -> tuple[Optional[RAGDebugAction], str]:
    user_prompt = _format_observation(obs)
    if history:
        user_prompt += "\n\nRecent actions:\n" + "\n".join(history[-4:])
    user_prompt += "\n\nWhat is your next action?"

    try:
        completion = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", flush=True)
        raw = ""

    action = _parse_action(raw)
    if action is None:
        # Fallback: submit to end episode gracefully rather than crashing
        print(f"[DEBUG] Failed to parse LLM output: {raw!r}", flush=True)
        action = RAGDebugAction(action_type=ActionType.SUBMIT, params={})
        raw = '{"action_type": "submit", "params": {}}'

    return action, raw


# ── Main episode loop ─────────────────────────────────────────────────────────

async def main() -> None:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = await RAGDebugEnv.from_docker_image(IMAGE_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    final_obs: Optional[RAGDebugObservation] = None
    history: List[str] = []

    log_start(task=f"task_{TASK_ID}", env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_id=TASK_ID)
        obs: RAGDebugObservation = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action, _ = _get_model_action(llm_client, obs, history)
            result = await env.step(action)
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            rewards.append(reward)
            steps_taken = step
            final_obs = obs

            log_step(step=step, action=str(action), reward=reward, done=done, error=None)
            history.append(f"Step {step}: {action} -> reward {reward:+.2f}")

            if done:
                break

        if final_obs is not None:
            score = min(max(_compute_score(final_obs), 0.0), 1.0)

        # Task 1/2 threshold is 0.75; Task 3 is 0.70
        success_threshold = 0.70 if TASK_ID == 3 else 0.75
        success = score >= success_threshold

    finally:
        try:
            await env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error: {exc}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
