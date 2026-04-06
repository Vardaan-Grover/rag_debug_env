# Models Reference

Complete field-by-field reference for every model in `models.py`.

## Quick Navigation

- [Enums](#enums)
- [RAGDebugAction](#ragdebugaction) — Tier 1, OpenEnv interface
- [RAGDebugObservation](#ragdebugobservation) — Tier 1, OpenEnv interface
- [PipelineConfig](#pipelineconfig) — Tier 2, internal
- [QueryResult](#queryresult) — Tier 2, internal
- [QualityMetrics](#qualitymetrics) — Tier 2, internal
- [CorpusStats](#corpusstats) — Tier 2, internal
- [Reward](#reward) — Tier 2, internal
- [FaultConfig](#faultconfig) — Tier 2, internal (never sent to agent)
- [InternalState](#internalstate) — Tier 2, internal (never sent to agent)
- [EpisodeResult](#episoderesult) — Tier 2, post-episode

---

## Enums

### `EmbeddingModel`
```
GENERAL  → sentence-transformers/all-MiniLM-L6-v2   General purpose, fast, 384-dim
MEDICAL  → pritamdeka/S-PubMedBert-MS-MARCO         Biomedical domain
LEGAL    → nlpaueb/legal-bert-base-uncased          Legal domain
CODE     → sentence-transformers/multi-qa-mpnet-base-dot-v1   Contrast retriever slot
```

### `Domain`
```
SOFTWARE → Python docs + HF docs     Task 1 (easy)
CLIMATE  → Wikipedia climate         Task 2 (medium)
MEDICAL  → MedRAG textbooks          Task 3 (hard)
```

### `ActionType`
```
ADJUST_CHUNK_SIZE      params: {"value": int}           64-2048
ADJUST_CHUNK_OVERLAP   params: {"value": int}           0-500
ADJUST_THRESHOLD       params: {"value": float}         0.0-1.0
ADJUST_TOP_K           params: {"value": int}           1-50
SWAP_EMBEDDING_MODEL   params: {"model": str}           EmbeddingModel value
TOGGLE_RERANKING       params: {"enabled": bool}
ADJUST_CONTEXT_LIMIT   params: {"value": int}           512-16384
REWRITE_QUERY          params: {"query_id": int, "strategy": str}
                        strategy: "expand"|"rephrase"|"decompose"
SUBMIT                 params: {}
```

### `FaultType`
```
CHUNK_TOO_LARGE        Large chunks dilute relevance signal
CHUNK_TOO_SMALL        Answers split across too many chunks
THRESHOLD_TOO_LOW      Noise floods retrieved set
THRESHOLD_TOO_HIGH     Almost nothing gets retrieved
TOP_K_TOO_SMALL        Correct chunk just out of reach
CONTEXT_OVERFLOW       Retrieved chunks truncated before reaching LLM
DUPLICATE_FLOODING     Near-duplicates dominate top-K
WRONG_EMBEDDING_MODEL  General model on domain-specific text
NO_RERANKING           No second-pass cleanup, noisy retrieval
```

---

## RAGDebugAction

**Inherits from:** `openenv.core.env_server.types.Action`

| Field | Type | Required | Description |
|---|---|---|---|
| `action_type` | `ActionType` | Yes | Which operation to perform |
| `params` | `dict[str, Any]` | No (default: `{}`) | Arguments for the action |

**Usage:**
```python
RAGDebugAction(action_type=ActionType.ADJUST_THRESHOLD, params={"value": 0.72})
RAGDebugAction(action_type=ActionType.SUBMIT)
```

**Created by:** The agent / baseline script  
**Consumed by:** `env.step(action)` → `_apply_action()`

---

## RAGDebugObservation

**Inherits from:** `openenv.core.env_server.types.Observation`

| Field | Type | Description |
|---|---|---|
| `pipeline_config` | `PipelineConfig` | Current pipeline parameters |
| `query_results` | `list[QueryResult]` | Per-query retrieval detail |
| `metrics` | `QualityMetrics` | Aggregate quality — primary signal |
| `corpus_stats` | `CorpusStats` | Static corpus metadata |
| `steps_taken` | `int` | Actions taken this episode |
| `max_steps` | `int` | Total budget |
| `task_id` | `int` | 1=easy, 2=medium, 3=hard |
| `task_description` | `str` | Plain language objective |
| `done` | `bool` | True when episode has ended |

**Created by:** `env.reset()` and `env.step()`  
**Consumed by:** Agent, baseline script

**What the agent should NOT see:** `InternalState.injected_faults`. The agent infers faults from metrics alone.

---

## PipelineConfig

All fields have validated bounds. `chunk_overlap < chunk_size` enforced by `model_validator`.

| Field | Type | Default | Bounds | Effect |
|---|---|---|---|---|
| `chunk_size` | `int` | 512 | 64–2048 | Size of text chunks in tokens |
| `chunk_overlap` | `int` | 50 | 0–500 | Shared tokens between adjacent chunks |
| `similarity_threshold` | `float` | 0.7 | 0.0–1.0 | Min cosine similarity to retrieve a chunk |
| `top_k` | `int` | 10 | 1–50 | Max chunks to retrieve per query |
| `embedding_model` | `EmbeddingModel` | GENERAL | enum | Which model encoded the chunks |
| `use_reranking` | `bool` | False | — | Whether to apply cross-encoder reranking |
| `context_window_limit` | `int` | 4096 | 512–16384 | Max tokens in assembled context |

**Mathematical effects:**
- `chunk_size` too large → dilution via average pooling of scores
- `similarity_threshold` too low → Gaussian noise elevates irrelevant chunks
- `top_k` too small → score range compression makes ranking differences tiny
- `embedding_model=GENERAL` on medical text → row permutation of query scores (WRONG_EMBEDDING_MODEL fault)

---

## QueryResult

One per query per step.

| Field | Type | Description |
|---|---|---|
| `query_id` | `int` | Which query |
| `query_text` | `str` | The query text |
| `retrieved_chunk_ids` | `list[int]` | Indices of retrieved chunks |
| `retrieval_scores` | `list[float]` | Scores parallel to chunk_ids |
| `n_retrieved` | `int` | Count (0 = empty retrieval) |
| `coverage_score` | `float` | `|R_agent ∩ R*| / |R*|` — primary metric |
| `precision_score` | `float` | `|R_agent ∩ R*| / |R_agent|` |
| `is_multi_hop` | `bool` | Requires 2 chunks (Task 3 only) |

**coverage_score math:**
```
R_agent = set of retrieved chunk IDs
R*      = set of ground-truth relevant chunk IDs
coverage = len(R_agent ∩ R*) / len(R*)
         = 0.0 if nothing relevant retrieved
         = 1.0 if all relevant chunks retrieved
```

---

## QualityMetrics

Aggregate across all queries.

| Field | Type | Description |
|---|---|---|
| `mean_coverage` | `float` | Primary optimization target |
| `mean_precision` | `float` | Fraction of retrieved that are relevant |
| `mean_recall` | `float` | Same as mean_coverage in this environment |
| `n_empty_retrievals` | `int` | Queries where n_retrieved == 0 |
| `n_context_overflows` | `int` | Queries where context was truncated |
| `multi_hop_coverage` | `float \| None` | Coverage on multi-hop queries only (Task 3) |

**Success thresholds (on mean_coverage):**
- Task 1: 0.80
- Task 2: 0.75
- Task 3: 0.70 (plus multi_hop_coverage > 0.60)

---

## CorpusStats

Static, loaded at `reset()`, doesn't change during episode.

| Field | Type | Description |
|---|---|---|
| `domain` | `Domain` | SOFTWARE, CLIMATE, or MEDICAL |
| `n_documents` | `int` | Source documents in corpus |
| `n_chunks` | `int` | Total chunks after splitting |
| `avg_chunk_tokens` | `int` | Used to estimate context overflow |
| `has_near_duplicates` | `bool` | Signals DUPLICATE_FLOODING possible |
| `n_queries` | `int` | Total queries being evaluated |
| `n_multi_hop_queries` | `int` | Multi-hop count (0 for Tasks 1, 2) |

---

## Reward

| Field | Type | Description |
|---|---|---|
| `value` | `float` | Scalar reward for RL algorithm |
| `components` | `dict[str, float]` | Named breakdown |

**Component names and values:**
```
coverage_delta           Δmean_coverage × 0.6
precision_delta          Δmean_precision × 0.3
step_cost                -0.02 (always)
redundancy_penalty       -0.10 (same action_type twice)
empty_retrieval_penalty  -0.15 × new_empty_retrievals
terminal_bonus           +2.0 (successful SUBMIT)
premature_submit_penalty -0.5 (SUBMIT before threshold)
```

---

## FaultConfig

**Never sent to agent.** Stored in `InternalState`.

| Field | Type | Description |
|---|---|---|
| `fault_type` | `FaultType` | Which fault |
| `params` | `dict[str, Any]` | Fault parameters (e.g. `{"scale": 4}`) |
| `description` | `str` | Human-readable description |

---

## InternalState

**Never sent to agent.** Accessible via `env.get_internal_state()`.

| Field | Type | Description |
|---|---|---|
| `injected_faults` | `list[FaultConfig]` | What was broken this episode |
| `episode_seed` | `int` | Random seed for reproducibility |
| `action_history` | `list[RAGDebugAction]` | All actions taken so far |
| `reward_history` | `list[float]` | All rewards received so far |

**Properties:**
- `total_reward` — sum of reward_history
- `fault_names` — list of `fault_type.value` strings

---

## EpisodeResult

Returned by `env.grade()` after episode ends.

| Field | Type | Description |
|---|---|---|
| `task_id` | `int` | Which task was solved |
| `task_score` | `float` | 0.0–1.0 from grader |
| `success` | `bool` | True if mean_coverage >= threshold |
| `n_steps` | `int` | Steps taken |
| `total_reward` | `float` | Sum of step rewards |
| `final_metrics` | `QualityMetrics` | Quality at episode end |
| `fault_names` | `list[str]` | Which faults were injected (revealed) |
| `action_history` | `list[RAGDebugAction]` | Full action sequence |

**Grading formula:**
```python
# Tasks 1 and 2:
task_score = (
    0.60 * final_metrics.mean_coverage +
    0.25 * final_metrics.mean_precision +
    0.15 * (1 - n_steps / max_steps)   # efficiency bonus
)

# Task 3 additionally weights multi_hop_coverage:
task_score = (
    0.55 * final_metrics.mean_coverage +
    0.25 * final_metrics.mean_precision +
    0.20 * final_metrics.multi_hop_coverage
)
```
