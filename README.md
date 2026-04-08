---
title: RAGDebugEnv
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# RAGDebugEnv

**An OpenEnv environment for training and evaluating AI agents that diagnose and fix broken Retrieval-Augmented Generation (RAG) pipelines.**

RAG pipelines are the backbone of modern AI applications — from enterprise search to customer support chatbots. But when they break (wrong embedding model, bad chunk sizes, misconfigured thresholds), diagnosing the root cause is a time-consuming, expert-level task. **RAGDebugEnv turns this real-world debugging challenge into a structured RL environment.**

## Why This Matters

RAG pipeline debugging is a genuine industry pain point:
- **Production RAG systems break silently** — retrieval quality degrades without obvious errors
- **Root cause diagnosis requires understanding** the interplay between embedding models, chunking strategies, similarity thresholds, and reranking
- **No existing training environments** target this critical skill for AI agents

RAGDebugEnv fills this gap by simulating realistic retrieval degradation through mathematically grounded fault injection, letting agents learn systematic debugging strategies.

## Architecture

```
Agent (inference.py)
  --> client.py (OpenEnv EnvClient over WebSocket)
  --> server/app.py (FastAPI + OpenEnv HTTP server)
  --> RAGDebugEnvironment (server/rag_debug_env_environment.py)
      |-- Precomputed S_true matrices (query-chunk cosine similarity)
      |-- Fault injection math (server/fault_math.py)
      |-- Ground truth R* sets (corpora/<domain>/ground_truth.json)
```

**Key design choice:** The environment does NOT call a live vector database. Instead, it uses precomputed similarity matrices (`S_true`) and applies fault transformations mathematically. This makes each `step()` execute in ~1ms, enabling rapid agent training and evaluation.

## Action Space

| Action | Parameters | Effect |
|--------|-----------|--------|
| `adjust_chunk_size` | `{"value": int}` (64-2048) | Changes chunk size; affects score smearing |
| `adjust_chunk_overlap` | `{"value": int}` (0-500) | Changes overlap; stabilizes boundary embeddings |
| `adjust_threshold` | `{"value": float}` (0.0-1.0) | Changes similarity threshold filter |
| `adjust_top_k` | `{"value": int}` (1-50) | Changes number of retrieved chunks |
| `swap_embedding_model` | `{"model": str}` | Switches between general/medical/legal/code models |
| `toggle_reranking` | `{"enabled": bool}` | Enables/disables cross-encoder reranking |
| `adjust_context_limit` | `{"value": int}` (512-16384) | Changes context window token limit |
| `rewrite_query` | `{"query_id": int, "strategy": str}` | Boosts a specific query's retrieval |
| `submit` | `{}` | Ends the episode and triggers grading |

## Observation Space

Each observation includes:
- **`pipeline_config`**: Current configuration parameters the agent can modify
- **`query_results`**: Per-query retrieval results (chunk IDs, scores, coverage, precision)
- **`metrics`**: Aggregate quality metrics (mean coverage, precision, empty retrievals, overflows)
- **`corpus_stats`**: Static metadata about the corpus (domain, size, multi-hop count)
- **`diagnostic_hints`**: Context-aware hints based on metric patterns
- **`reward_components`**: Named breakdown of the reward signal for interpretability
- **`last_action_error`**: Feedback when an action was invalid

## Tasks

| Task | Domain | Difficulty | Faults | Success Condition |
|------|--------|-----------|--------|-------------------|
| **Task 1** | Software (Python docs) | Easy | 1-2 config faults (sampled) | `task_score >= 0.75` |
| **Task 2** | Climate (IPCC reports) | Medium | Compound config faults (sampled) | `task_score >= 0.75` |
| **Task 3** | Medical (MedRAG textbooks) | Hard | Wrong embedding model + config faults + multi-hop | `task_score >= 0.70` AND `multi_hop_coverage > 0.60` |

### Fault Types (9 total)

| Fault | Mechanism | Fix Strategy |
|-------|-----------|-------------|
| `chunk_too_large` | Score smearing via box filter | Reduce chunk_size |
| `chunk_too_small` | Gaussian noise from unstable embeddings | Increase chunk_size + overlap |
| `threshold_too_high` | Score deflation (*0.55) | Lower threshold |
| `threshold_too_low` | Additive noise clutters retrieval | Raise threshold |
| `top_k_too_small` | Score compression toward 0.5 | Increase top_k + enable reranking |
| `duplicate_flooding` | Boosted duplicate chunks crowd results | Enable reranking |
| `context_overflow` | Zeroed tail columns | Increase context_limit |
| `no_reranking` | Additive noise without cross-encoder | Enable reranking |
| `wrong_embedding_model` | Fundamentally wrong score distribution | Swap to domain-appropriate model |

## Reward Design

Dense per-step reward with interpretable components:

| Component | Weight | Description |
|-----------|--------|-------------|
| `coverage_delta` | 0.6x | Change in mean retrieval coverage |
| `precision_delta` | 0.3x | Change in mean retrieval precision |
| `step_cost` | -0.03 | Fixed cost per step (efficiency pressure) |
| `redundancy_penalty` | -0.10 | Repeating the same action type |
| `empty_retrieval_penalty` | -0.15/each | Newly introduced empty retrievals |
| `empty_recovery_bonus` | +0.15/each | Recovering from empty retrievals |
| `overflow_penalty` | -0.10/each | Newly introduced context overflows |
| `multi_hop_bonus` | 0.15x | Multi-hop coverage improvement (Task 3) |
| `invalid_action_penalty` | -0.05 | Invalid action parameters |
| `terminal_bonus` | +2.0 | Successful submit |
| `premature_submit_penalty` | -0.50 | Submitting below threshold |

**Task score formula:**
- Tasks 1 & 2: `0.60 * coverage + 0.25 * precision + 0.15 * efficiency`
- Task 3: `0.55 * coverage + 0.25 * precision + 0.20 * multi_hop_coverage`

## Setup

### Required Environment Variables

```bash
export API_BASE_URL="https://router.huggingface.co/v1"   # LLM endpoint
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"            # Model identifier
export HF_TOKEN="your-hugging-face-token"                 # API key
```

### Install & Run

```bash
# Install dependencies
uv sync

# Run the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run inference (in another terminal)
python inference.py
```

### Docker

```bash
docker build -t rag_debug_env .
docker run -p 7860:7860 -e HF_TOKEN=$HF_TOKEN rag_debug_env
```

### Validate OpenEnv Spec

```bash
openenv validate
```

## Repository Layout

```
rag_debug_env/
  inference.py           # Competition inference script ([START]/[STEP]/[END] logging)
  client.py              # OpenEnv client (WebSocket-based)
  models.py              # Typed Pydantic models (Action, Observation, Config, etc.)
  openenv.yaml           # OpenEnv manifest
  pyproject.toml         # Package metadata and dependencies
  Dockerfile             # Multi-stage Docker build for HF Spaces
  server/
    app.py               # FastAPI application (create_app with OpenEnv)
    rag_debug_env_environment.py  # Core environment (reset/step/state)
    constants.py          # Task definitions, fault sets, thresholds
    fault_math.py         # Pure fault injection math (S_true -> S_faulted)
    corpus.py             # Corpus loader with synthetic fallback
  corpora/
    software/             # Pre-built corpus artifacts (271 chunks, 48 queries)
    climate/              # Pre-built corpus artifacts (612 chunks, 44 queries)
    medical/              # Pre-built corpus artifacts (359 chunks, 44 queries, 6 multi-hop)
    stages/               # 6-stage corpus build pipeline
  outputs/
    eval_agent.py         # GPT-4o-mini zero-shot evaluation agent
    train_grpo.py         # GRPO training scaffold
  docs/
    ARCHITECTURE.md       # Detailed architecture documentation
    MODELS_REFERENCE.md   # Embedding model details
```

## Corpus

Each domain includes pre-built artifacts:
- `chunks.json` — chunked documents with text and token counts
- `queries.json` — generated queries (direct and multi-hop)
- `ground_truth.json` — relevant chunk IDs (R*) per query
- `S_true_{general,medical,legal,code}.npy` — precomputed cosine similarity matrices
- `corpus_stats.json` — corpus metadata

To rebuild:
```bash
python -m corpora.build_corpus --domain all
```

## Example Agent Interaction

```
[START] task=task_1 env=rag_debug_env model=Qwen/Qwen2.5-72B-Instruct

Initial state: coverage=0.340, precision=0.280, empty=2
Diagnostic hint: "2 queries have empty retrievals — lower threshold or increase top_k"

Step 1: adjust_threshold(value=0.15)  -> coverage=0.620, precision=0.450  reward=+0.17
Step 2: toggle_reranking(enabled=true) -> coverage=0.720, precision=0.580  reward=+0.10
Step 3: adjust_top_k(value=15)         -> coverage=0.840, precision=0.610  reward=+0.07
Step 4: submit()                       -> SUCCESS! task_score=0.82

[END] success=true steps=4 score=0.820 rewards=0.17,0.10,0.07,2.00
```
