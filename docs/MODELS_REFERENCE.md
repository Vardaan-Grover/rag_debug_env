# Models Reference

Field-by-field reference for the Pydantic models in `models.py`.

## Enums

### `EmbeddingModel`

- `GENERAL` -> `sentence-transformers/all-MiniLM-L6-v2`
- `MEDICAL` -> `NeuML/pubmedbert-base-embeddings`
- `LEGAL` -> `nlpaueb/legal-bert-base-uncased`
- `CODE` -> `sentence-transformers/multi-qa-mpnet-base-dot-v1`

### `Domain`

- `SOFTWARE`
- `CLIMATE`
- `MEDICAL`

### `ActionType`

- `ADJUST_CHUNK_SIZE` (`params.value: int`)
- `ADJUST_CHUNK_OVERLAP` (`params.value: int`)
- `ADJUST_THRESHOLD` (`params.value: float`)
- `ADJUST_TOP_K` (`params.value: int`)
- `SWAP_EMBEDDING_MODEL` (`params.model: str`)
- `TOGGLE_RERANKING` (`params.enabled: bool`)
- `ADJUST_CONTEXT_LIMIT` (`params.value: int`)
- `REWRITE_QUERY` (`params.query_id: int`, optional `strategy` accepted by callers)
- `SUBMIT` (no params)

### `FaultType`

- `CHUNK_TOO_LARGE`
- `CHUNK_TOO_SMALL`
- `THRESHOLD_TOO_LOW`
- `THRESHOLD_TOO_HIGH`
- `TOP_K_TOO_SMALL`
- `CONTEXT_OVERFLOW`
- `DUPLICATE_FLOODING`
- `WRONG_EMBEDDING_MODEL`
- `NO_RERANKING`

## Tier 1 OpenEnv Interface Models

### `RAGDebugAction(Action)`

Fields:

- `action_type: ActionType`
- `params: dict[str, Any] = {}`

Notes:

- `params` accepts dict or JSON-stringified dict (validator coercion)
- used by server step routing in `_apply_action`

### `RAGDebugObservation(Observation)`

Fields:

- `pipeline_config: PipelineConfig`
- `query_results: list[QueryResult]`
- `metrics: QualityMetrics`
- `corpus_stats: CorpusStats`
- `steps_taken: int`
- `max_steps: int`
- `task_id: int`
- `task_description: str`
- `done: bool`
- `last_action_error: str | None`
- `diagnostic_hints: list[str]`
- `reward_components: dict[str, float]`

Design note:

- injected faults are intentionally omitted from observation and remain internal.

## Tier 2 Internal Models

### `PipelineConfig`

Defaults and bounds:

- `chunk_size: int = 512` (`64..2048`)
- `chunk_overlap: int = 50` (`0..500`)
- `similarity_threshold: float = 0.3` (`0.0..1.0`)
- `top_k: int = 10` (`1..50`)
- `embedding_model: EmbeddingModel = GENERAL`
- `use_reranking: bool = False`
- `context_window_limit: int = 4096` (`512..16384`)

Validation:

- `chunk_overlap < chunk_size` (model validator)

### `QueryResult`

Per-query retrieval result:

- `query_id: int`
- `query_text: str`
- `retrieved_chunk_ids: list[int]`
- `retrieval_scores: list[float]`
- `n_retrieved: int`
- `coverage_score: float` (`0..1`)
- `precision_score: float` (`0..1`)
- `is_multi_hop: bool = False`

Metric definitions:

- coverage = `|R_agent ∩ R*| / |R*|`
- precision = `|R_agent ∩ R*| / |R_agent|`

### `QualityMetrics`

Aggregate metrics:

- `mean_coverage: float`
- `mean_precision: float`
- `mean_recall: float`
- `n_empty_retrievals: int`
- `n_context_overflows: int`
- `multi_hop_coverage: float | None`

### `CorpusStats`

Static corpus metadata:

- `domain: Domain`
- `n_documents: int`
- `n_chunks: int`
- `avg_chunk_tokens: int`
- `has_near_duplicates: bool`
- `n_queries: int`
- `n_multi_hop_queries: int`

### `Reward`

- `value: float`
- `components: dict[str, float]`

Component names emitted by environment reward logic:

- `progress_reward`
- `delta_bonus`
- `empty_retrieval_signal`
- `overflow_signal`
- `step_cost`
- `redundancy_penalty`
- `invalid_action_penalty` (only when the last action had invalid params)

Terminal submit components:

- `terminal_success` (successful submit)
- `terminal_failure` (unsuccessful submit)

Terminal submit rewards are handled directly in action routing:

- success: `0.7 + 0.3 * task_score` (clipped to `[0.7, 1.0]`)
- failure: `0.2 * task_score` (clipped to `[0.0, 0.2]`)

### `FaultConfig`

Internal fault descriptor:

- `fault_type: FaultType`
- `params: dict[str, Any] = {}`
- `description: str = ""`

### `InternalState`

Server-side state:

- `injected_faults: list[FaultConfig]`
- `episode_seed: int`
- `action_history: list[RAGDebugAction]`
- `reward_history: list[float]`

Properties:

- `total_reward`
- `fault_names`

### `EpisodeResult`

Post-episode summary model (not currently exposed via custom app endpoint):

- `task_id: int`
- `task_score: float` (`0..1`)
- `success: bool`
- `n_steps: int`
- `total_reward: float`
- `final_metrics: QualityMetrics`
- `fault_names: list[str]`
- `action_history: list[RAGDebugAction]`

## Runtime Scoring Rules (Environment)

From `server/rag_debug_env_environment.py`:

Task score:

- Task 1 and 2:
  - `0.60 * mean_coverage + 0.25 * mean_precision + 0.15 * (1 - n_steps/max_steps)`
- Task 3:
  - `0.55 * mean_coverage + 0.25 * mean_precision + 0.20 * multi_hop_coverage`

Success checks:

- Task 1: `task_score >= 0.75`
- Task 2: `task_score >= 0.75`
- Task 3: `task_score >= 0.70` and `multi_hop_coverage > 0.60`
