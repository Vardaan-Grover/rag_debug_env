# Architecture

## Runtime Topology

```text
Agent / baseline script
  -> client.RAGDebugEnv (openenv.core.EnvClient)
  -> WebSocket/HTTP to FastAPI app (server/app.py)
  -> RAGDebugEnvironment (server/rag_debug_env_environment.py)
  -> Corpus artifacts (corpora/<domain>/*)
```

Server construction uses `openenv.core.env_server.http_server.create_app`:

- Environment class: `RagDebugEnvironment` aliasing `RAGDebugEnvironment`
- Action schema: `RAGDebugAction`
- Observation schema: `RAGDebugObservation`
- `env_name="rag_debug_env"`
- `max_concurrent_envs=1` in `server/app.py`

## Core Simulation Contract

The environment does not call a live vector database during episodes.

Episode-time retrieval is simulated from precomputed matrices:

- `S_true_{general,medical,legal,code}.npy`: query-chunk cosine matrices
- `ground_truth.json`: relevant chunk IDs (`R*`) per query

At reset:

1. Load one domain corpus (`software`, `climate`, `medical`)
2. Sample episode queries (5 total per task)
3. Slice full `S_true` matrices down to episode query rows
4. Sample injected faults
5. Build `S_faulted` via `server/fault_math.py`
6. Return initial `RAGDebugObservation`

At step:

1. Apply action to config/model/rewrite overlay
2. Recompute `S_faulted` when required
3. Simulate retrieval (`top_k` then threshold)
4. Compute per-query coverage/precision and aggregate metrics
5. Compute dense reward (or terminal submit reward)

## Task Configuration

Values below are sourced from `server/constants.py` and `server/rag_debug_env_environment.py`.

### Shared limits

- Episode queries: 5 (`_N_EPISODE_QUERIES` for all tasks)
- Max steps: 10 (`_MAX_STEPS`)

### Task 1 (software)

- Domain: `software`
- Faults sampled from:
  - `[chunk_too_large, no_reranking]`
  - `[threshold_too_high]`
  - `[top_k_too_small]`
  - `[chunk_too_large]`
- Success check on submit: `task_score >= 0.75`

### Task 2 (climate)

- Domain: `climate`
- Faults sampled from:
  - `[threshold_too_low, duplicate_flooding]`
  - `[top_k_too_small, context_overflow]`
  - `[duplicate_flooding]`
  - `[context_overflow]`
- Success check on submit: `task_score >= 0.75`

### Task 3 (medical)

- Domain: `medical`
- Fixed fault set:
  - `wrong_embedding_model`
  - `chunk_too_large`
  - `threshold_too_high`
- Initial active model is `legal` (intentional mismatch)
- Query sampling forces up to 2 multi-hop queries per episode
- Success check on submit:
  - `task_score >= 0.70`
  - `multi_hop_coverage > 0.60`

## Reward and Scoring

Dense step reward (`_compute_reward`):

- `coverage_delta = (new.mean_coverage - prev.mean_coverage) * 0.6`
- `precision_delta = (new.mean_precision - prev.mean_precision) * 0.3`
- `step_cost = -0.02`
- `redundancy_penalty = -0.10` for same action type twice in a row
- `empty_retrieval_penalty = -0.15 * new_empty_retrievals`

Submit reward override (`_apply_action`):

- `+2.0` if success condition is met
- `-0.5` otherwise

Task score (`_compute_task_score`):

- Task 1/2: `0.60*coverage + 0.25*precision + 0.15*efficiency`
- Task 3: `0.55*coverage + 0.25*precision + 0.20*multi_hop_coverage`

## Fault Math (Implemented)

All transformations are in `server/fault_math.py`.

- `CHUNK_TOO_LARGE`: 1D uniform filter along chunk axis; severity scales with `chunk_size`
- `CHUNK_TOO_SMALL`: gaussian noise scaled by small chunk size, mitigated by overlap
- `THRESHOLD_TOO_LOW`: additive gaussian noise
- `THRESHOLD_TOO_HIGH`: multiplicative score deflation (`* 0.55`)
- `TOP_K_TOO_SMALL`: score compression toward 0.5; less severe if reranking enabled
- `DUPLICATE_FLOODING`: boosts random duplicate columns; reduced if reranking enabled
- `CONTEXT_OVERFLOW`: zeroes tail columns based on `context_window_limit`
- `NO_RERANKING`: additive noise only when reranking is off
- `WRONG_EMBEDDING_MODEL`: implicit by selecting wrong matrix (not a direct transform)

## Determinism and Fallbacks

- Noise arrays and duplicate indices are sampled once at reset and reused during recomputation for deterministic intra-episode behavior.
- If required corpus files are missing, `server/corpus.py` falls back to synthetic data and emits warnings.
- Synthetic fallback is for smoke testing only, not for real training/evaluation.
