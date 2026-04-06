# Build Status

## Phase 0 — Project Skeleton ✅

All files created, package installable.

```bash
pip install -e .  # works
```

Files: `pyproject.toml`, `openenv.yaml`, `Dockerfile`, `server/requirements.txt`, all `__init__.py` files.

---

## Phase 1 — Pydantic Models ✅

All models written and tested. 35+ tests passing.

```bash
pytest tests/test_models.py -v  # all pass (requires openenv-core installed)
```

**Tier 1 (OpenEnv interface):**
- `RAGDebugAction(Action)` — inherits from `openenv.core.env_server.types.Action`
- `RAGDebugObservation(Observation)` — inherits from `openenv.core.env_server.types.Observation`

**Tier 2 (internal):**
- `PipelineConfig` — 7 fields, all validated with bounds
- `QueryResult` — per-query retrieval output including coverage_score
- `QualityMetrics` — aggregate metrics including multi_hop_coverage
- `CorpusStats` — static corpus metadata
- `Reward` — scalar + named component breakdown
- `FaultConfig` — single fault parameters (never sent to agent)
- `InternalState` — full server-side state including injected faults
- `EpisodeResult` — post-episode grade

**Key constraint:** `chunk_overlap < chunk_size` enforced by `model_validator`.

---

## Phase 2a — Server Environment ✅

`server/rag_debug_env_environment.py` — `RAGDebugEnvironment(Environment)` fully implemented.

- `reset(task_id, seed)` — loads corpus, samples faults, applies them, returns initial observation
- `step(action)` — applies action to config or S_matrix, recomputes metrics, computes reward
- `state` property — returns OpenEnv `State(episode_id, step_count)`
- `get_internal_state()` — returns `InternalState` for graders (not sent to agent)
- `get_last_reward()` — float, used by app.py to build StepResult
- `grade()` — computes `EpisodeResult` after episode ends

**Fault injection:** 9 fault types, each modeled as a mathematical transformation of S_true:
- `CHUNK_TOO_LARGE` — average pool (scipy uniform_filter1d)
- `THRESHOLD_TOO_LOW` — Gaussian noise injection
- `WRONG_EMBEDDING_MODEL` — row permutation of similarity scores
- `DUPLICATE_FLOODING` — boost scores for random chunk subset
- etc.

**Synthetic fallback:** If corpus files not found, generates synthetic NumPy data. Allows all tests to run before `build_corpus.py` completes.

---

## Phase 2b — FastAPI App + Client ✅

`server/app.py`:
```python
app = create_env_app(env, RAGDebugAction, RAGDebugObservation)
# Adds custom GET /grade endpoint
```

`client.py` — `RAGDebugEnv(HTTPEnvClient[RAGDebugAction, RAGDebugObservation])`:
- `reset(task_id=1)` — async, sends task_id in payload
- `step(action)` — standard
- `grade()` — custom endpoint call, returns `EpisodeResult`

`__init__.py` exports: `RAGDebugEnv`, `RAGDebugAction`, `RAGDebugObservation`

---

## Phase 3 — Corpus Build ✅

### Stage Status

- Stage 1 (`s1_load.py`) — ✅ implemented and cached per domain
- Stage 2 (`s2_chunk.py`) — ✅ implemented and cached per domain
- Stage 3 (`s3_queries.py`) — ✅ implemented and cached per domain
- Stage 4 (`s4_multihop.py`) — ✅ implemented (medical-only multi-hop)
- Stage 5 (`s5_embed.py`) — ✅ implemented and generated `S_true_{general,medical,legal,code}.npy`
- Stage 6 (`s6_grade.py`) — ⏳ pending

Current corpus artifacts exist for `software`, `climate`, and `medical` through Stage 5.
See `docs/CORPUS_BUILD_PLAN.md` for Stage 6 and verification specs.

---

## Phase 4 — Tests ⏳

`tests/test_environment.py` and `tests/test_faults.py` are written but not yet validated against the real environment (only runs with synthetic fallback currently). Will be fully validated after corpus build completes.

---

## Phase 5 — Baseline Script ⏳

`baseline/run_baseline.py` — not yet written.

Design: GPT-4o (or GPT-4o-mini) agent using structured JSON outputs matching `RAGDebugAction` schema. Loop: format observation as prompt → LLM response → parse action → env.step() → repeat until done. Report per-task scores.

---

## Phase 6 — Deployment ⏳

- `openenv validate` — not yet run (requires openenv CLI)
- HuggingFace Space — not yet deployed
- Docker build/run — Dockerfile written, not yet tested end-to-end
