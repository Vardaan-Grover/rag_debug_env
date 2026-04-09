"""
server/rag_debug_env_environment.py
------------------------------------
Real RAGDebugEnvironment implementation.

Replaces the echo stub with a full RL environment for training agents to
diagnose and fix broken RAG pipelines.

Architecture summary
--------------------
- Each episode samples N queries from one domain's corpus.
- S_true_{model}.npy matrices (precomputed by Stage 5) hold the ground-truth
  cosine similarity scores for every (query, chunk) pair.
- Faults are applied as mathematical transformations: S_faulted = f(S_true, config, faults).
- The agent's job is to modify PipelineConfig (and/or query rewrites) until
  the retrieval simulation—run on S_faulted—recovers adequate coverage of R*.
- Every config change triggers _recompute_S_faulted() so the fault math can
  modulate its severity based on the new config values.
- Noise arrays are pre-generated at reset() time for determinism across
  recomputation calls within a single episode.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata, State
from pydantic import ValidationError

from server.constants import (
    _TASK_DOMAIN,
    _TASK_DESCRIPTION,
    _N_EPISODE_QUERIES,
    _MAX_STEPS,
    _MODEL_FILE,
    _TASK1_FAULT_SETS,
    _TASK2_FAULT_SETS,
    _TASK3_FAULTS,
)
from server.corpus import _load_corpus
from server.fault_math import apply_faults
from models import (
    RAGDebugAction,
    RAGDebugObservation,
    ActionType,
    EmbeddingModel,
    FaultType,
    FaultConfig,
    InternalState,
    PipelineConfig,
    QueryResult,
    QualityMetrics,
    CorpusStats,
    Domain,
    Reward,
)

class RAGDebugEnvironment(Environment):
    """
    RL environment for diagnosing and fixing broken RAG pipelines.

    Each episode samples a small query set from one domain, injects faults
    into the similarity matrix, and rewards the agent for recovering
    retrieval quality through pipeline config changes.

    Tasks
    -----
    Task 1 (software): One or two config faults. Success threshold ≥ 0.80.
    Task 2 (climate):  Compound config faults. Success threshold ≥ 0.75.
    Task 3 (medical):  Wrong embedding model + config faults + multi-hop.
                       Success threshold ≥ 0.70 AND multi_hop_coverage > 0.60.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # Episode-level data, set during reset()
        self._task_id: int = 1
        self._domain: str = "software"
        self._chunks: List[Dict] = []
        self._episode_queries: List[Dict] = []
        self._ground_truth: Dict[str, List[int]] = {}
        self._corpus_stats_dict: Dict = {}

        # Per-episode numpy data
        self._chunk_ids: List[int] = []
        self._chunk_tokens: List[int] = []
        self._chunk_id_to_tokens: Dict[int, int] = {}  # O(1) token lookup
        self._s_true_episode: Dict[str, np.ndarray] = {}  # model_name → (n_q, n_c) float32
        self._active_model: EmbeddingModel = EmbeddingModel.GENERAL

        # Deterministic noise (pre-generated at reset, scaled during recompute)
        self._noise: Dict[FaultType, np.ndarray] = {}
        self._dupe_ids: np.ndarray = np.array([], dtype=int)

        # REWRITE_QUERY persistent overlay
        self._rewrite_boosts: np.ndarray = np.zeros((0, 0), dtype=np.float32)

        # Current faulted matrix (rebuilt by _recompute_S_faulted)
        self._S_faulted: np.ndarray = np.zeros((0, 0), dtype=np.float32)

        # Episode config & state
        self._config: PipelineConfig = PipelineConfig()
        self._injected_faults: List[FaultConfig] = []
        self._internal_state: InternalState = InternalState(
            injected_faults=[], episode_seed=0
        )
        self._prev_metrics: Optional[QualityMetrics] = None
        self._prev_action_type: Optional[ActionType] = None
        self._done: bool = False
        self._last_action_error: Optional[str] = None
        self._last_reward_components: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # OpenEnv required interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> RAGDebugObservation:
        """Reset the environment and return the initial observation."""
        # Inherited from openenv.core.env_server.interfaces.Environment.
        # Clears any OpenEnv framework-level episode state (e.g. step counters
        # tracked by the base class) before we re-initialise domain-specific state.
        self._reset_rubric()

        # Task selection
        task_id = int(kwargs.get("task_id", 1))
        if task_id not in (1, 2, 3):
            raise ValueError(f"task_id must be 1, 2, or 3; got {task_id}")
        self._task_id = task_id

        # RNG
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**31))
        rng = np.random.default_rng(seed)

        # State bookkeeping
        ep_id = episode_id or str(uuid4())
        self._state = State(episode_id=ep_id, step_count=0)
        self._done = False
        self._prev_action_type = None
        self._last_action_error = None
        self._last_reward_components = {}

        # Domain & corpus
        self._domain = _TASK_DOMAIN[task_id].value
        corpus = _load_corpus(self._domain)
        self._chunks = corpus["chunks"]
        self._ground_truth = corpus["ground_truth"]
        self._corpus_stats_dict = corpus["corpus_stats"]

        all_queries: List[Dict] = corpus["queries"]

        # Sample episode queries
        self._episode_queries = self._sample_queries(all_queries, task_id, rng)

        # Build episode index structures
        self._chunk_ids = [c["chunk_id"] for c in self._chunks]
        self._chunk_tokens = [c.get("n_tokens", 100) for c in self._chunks]
        self._chunk_id_to_tokens = {
            c["chunk_id"]: c.get("n_tokens", 100) for c in self._chunks
        }
        n_q = len(self._episode_queries)
        n_c = len(self._chunks)

        # Slice S_true to episode query rows
        # Full S_true shape: (all_queries, n_chunks); query_id == row index in S_true
        self._s_true_episode = {}
        all_query_ids = [q["query_id"] for q in all_queries]
        ep_query_ids = [q["query_id"] for q in self._episode_queries]
        # Build index map: query_id → row in the full S_true matrix
        qid_to_row = {qid: i for i, qid in enumerate(all_query_ids)}
        ep_rows = [qid_to_row[qid] for qid in ep_query_ids]

        s_true_full = corpus["s_true"]
        for model_name, s_full in s_true_full.items():
            self._s_true_episode[model_name] = s_full[ep_rows, :].copy()

        # Pre-generate noise (unit normal; scaled during recompute)
        shape = (n_q, n_c)
        self._noise = {
            FaultType.CHUNK_TOO_SMALL:   rng.standard_normal(shape).astype(np.float32),
            FaultType.THRESHOLD_TOO_LOW: rng.standard_normal(shape).astype(np.float32),
            FaultType.NO_RERANKING:      rng.standard_normal(shape).astype(np.float32),
        }
        self._dupe_ids = rng.choice(n_c, size=max(1, n_c // 7), replace=False)

        # Reset overlay and config
        self._rewrite_boosts = np.zeros(shape, dtype=np.float32)
        self._config = PipelineConfig()

        # Task 3 starts with the LEGAL embedding model (wrong model for medical text).
        # S_true_legal has ~0.62 coverage vs ~0.90 for GENERAL on medical data.
        # The agent must discover this and swap to GENERAL or MEDICAL.
        self._active_model = EmbeddingModel.LEGAL if task_id == 3 else EmbeddingModel.GENERAL
        self._config = self._config.model_copy(update={"embedding_model": self._active_model})

        # Start from a mildly constrained baseline so most episodes leave
        # headroom for meaningful improvement steps.
        self._config = self._config.model_copy(
            update={
                "top_k": int(rng.integers(5, 9)),
                "similarity_threshold": float(rng.uniform(0.34, 0.48)),
            }
        )

        # Sample and inject faults
        self._injected_faults = self._sample_faults(task_id, rng)
        self._internal_state = InternalState(
            injected_faults=self._injected_faults,
            episode_seed=seed,
        )

        # Fault-specific initial config: some faults only degrade coverage when
        # the related parameter starts at a bad value.
        # TOP_K_TOO_SMALL: score compression preserves rank order, so coverage
        #   stays high at top_k=10. Start with a small top_k so the agent must
        #   increase it to recover coverage.
        # DUPLICATE_FLOODING: flooded chunks can't displace high-scoring relevant
        #   chunks in a top_k=10 pool. Start with reduced top_k so the flooding
        #   actually crowds out relevant chunks.
        fault_types_active = {f.fault_type for f in self._injected_faults}
        if FaultType.TOP_K_TOO_SMALL in fault_types_active:
            self._config = self._config.model_copy(update={"top_k": int(rng.integers(2, 4))})
        elif FaultType.DUPLICATE_FLOODING in fault_types_active:
            self._config = self._config.model_copy(update={"top_k": int(rng.integers(4, 8))})

        # If reset still lands in a very strong state, add one extra nudge
        # so the agent usually has room to improve from step 1.
        self._calibrate_initial_difficulty(rng)

        # Initial matrix + metrics
        self._recompute_S_faulted()
        initial_results = self._simulate_retrieval()
        self._prev_metrics = self._compute_metrics(initial_results)

        return self._build_observation(initial_results, reward=None)

    def step(
        self,
        action: RAGDebugAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> RAGDebugObservation:
        """Execute one agent action and return the updated observation."""
        if self._done:
            raise RuntimeError("Episode is already done. Call reset() to start a new episode.")

        self._state.step_count += 1
        prev_metrics = self._prev_metrics

        # Route action (returns Reward for SUBMIT, None otherwise)
        reward_obj = self._apply_action(action)

        # Recompute retrieval results and metrics
        new_results = self._simulate_retrieval()
        new_metrics = self._compute_metrics(new_results)

        # Compute reward if not already set by SUBMIT handler
        if reward_obj is None:
            reward_obj = self._compute_reward(prev_metrics, new_metrics, action)

        # Keep per-step reward components available in observations.
        self._last_reward_components = dict(reward_obj.components)

        self._internal_state.action_history.append(action)
        self._internal_state.reward_history.append(reward_obj.value)

        self._prev_metrics = new_metrics
        self._prev_action_type = action.action_type

        # Auto-terminate on max steps
        if self._state.step_count >= _MAX_STEPS and not self._done:
            self._done = True

        return self._build_observation(new_results, reward=reward_obj.value)

    @property
    def state(self) -> State:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        readme_path = Path(__file__).parent.parent / "README.md"
        readme_content: Optional[str] = None
        if readme_path.exists():
            raw = readme_path.read_text(encoding="utf-8")
            # Strip YAML frontmatter (--- ... ---) so the UI renders clean Markdown
            if raw.startswith("---"):
                end = raw.find("---", 3)
                if end != -1:
                    raw = raw[end + 3:].lstrip("\n")
            readme_content = raw
        return EnvironmentMetadata(
            name="RAGDebugEnv",
            description="Debug broken RAG pipelines by tuning config and swapping embedding models.",
            readme_content=readme_content,
            version="1.0.0",
        )

    # ------------------------------------------------------------------
    # Action routing
    # ------------------------------------------------------------------

    def _update_config(self, **updates) -> Optional[str]:
        """
        Apply updates to PipelineConfig using the constructor, which runs all
        Pydantic validators (including model_validators).

        model_copy(update=...) does NOT run validators in Pydantic v2, so
        invalid combinations (e.g. chunk_overlap >= chunk_size) are silently
        accepted and later cause crashes when the config is embedded in an
        observation.  Using the constructor guarantees validation always runs.

        Returns an error string if validation fails, None on success.
        """
        try:
            self._config = PipelineConfig(**{**self._config.model_dump(), **updates})
            return None
        except (ValueError, TypeError, ValidationError) as exc:
            return str(exc)

    def _apply_action(self, action: RAGDebugAction) -> Optional[Reward]:
        """
        Apply action to config / overlays.
        Returns a Reward if the action is SUBMIT, else None.
        """
        self._last_action_error = None
        t = action.action_type
        p = action.params

        if t == ActionType.ADJUST_CHUNK_SIZE:
            value = int(p.get("value", self._config.chunk_size))
            err = self._update_config(chunk_size=value)
            if err:
                self._last_action_error = f"Invalid chunk_size {value}: {err}"
            self._recompute_S_faulted()

        elif t == ActionType.ADJUST_CHUNK_OVERLAP:
            value = int(p.get("value", self._config.chunk_overlap))
            err = self._update_config(chunk_overlap=value)
            if err:
                self._last_action_error = f"Invalid chunk_overlap {value}: {err}"
            # Recompute required: fault_math.apply_faults() uses chunk_overlap to
            # modulate CHUNK_TOO_SMALL noise sigma (higher overlap stabilises boundary
            # embeddings, reducing noise severity). Without recompute, the config change
            # has no visible effect on retrieval scores until another action triggers it.
            self._recompute_S_faulted()

        elif t == ActionType.ADJUST_THRESHOLD:
            value = float(p.get("value", self._config.similarity_threshold))
            err = self._update_config(similarity_threshold=value)
            if err:
                self._last_action_error = f"Invalid threshold {value}: {err}"
            # Threshold is applied in retrieval simulation; no matrix recompute

        elif t == ActionType.ADJUST_TOP_K:
            value = int(p.get("value", self._config.top_k))
            err = self._update_config(top_k=value)
            if err:
                self._last_action_error = f"Invalid top_k {value}: {err}"

        elif t == ActionType.ADJUST_CONTEXT_LIMIT:
            value = int(p.get("value", self._config.context_window_limit))
            err = self._update_config(context_window_limit=value)
            if err:
                self._last_action_error = f"Invalid context_limit {value}: {err}"
            self._recompute_S_faulted()

        elif t == ActionType.SWAP_EMBEDDING_MODEL:
            model_str = str(p.get("model", self._active_model.value))
            try:
                new_model = EmbeddingModel(model_str)
            except (ValueError, KeyError) as exc:
                self._last_action_error = f"Invalid embedding model '{model_str}': {exc}"
                new_model = None
            if new_model is not None:
                self._active_model = new_model
                self._update_config(embedding_model=new_model)
            self._recompute_S_faulted()

        elif t == ActionType.TOGGLE_RERANKING:
            enabled = bool(p.get("enabled", not self._config.use_reranking))
            self._update_config(use_reranking=enabled)
            self._recompute_S_faulted()

        elif t == ActionType.REWRITE_QUERY:
            query_id = p.get("query_id")
            ep_ids = [q["query_id"] for q in self._episode_queries]
            if query_id in ep_ids:
                row = ep_ids.index(query_id)
                r_star = self._ground_truth.get(str(query_id), [])
                # Map chunk_id → column index
                cid_to_col = {cid: col for col, cid in enumerate(self._chunk_ids)}
                cols = [cid_to_col[cid] for cid in r_star if cid in cid_to_col]
                if cols:
                    self._rewrite_boosts[row, cols] = 0.20
            self._recompute_S_faulted()

        elif t == ActionType.SUBMIT:
            # Compute final metrics at SUBMIT time for accurate grading.
            results = self._simulate_retrieval()
            metrics = self._compute_metrics(results)
            task_score = self._compute_task_score(metrics)
            self._done = True

            success = self._check_success(metrics, task_score)
            if success:
                terminal_value = float(np.clip(0.7 + 0.3 * task_score, 0.7, 1.0))
                return Reward(
                    value=terminal_value,
                    components={"terminal_success": terminal_value},
                )
            else:
                terminal_value = float(np.clip(0.2 * task_score, 0.0, 0.2))
                return Reward(
                    value=terminal_value,
                    components={"terminal_failure": terminal_value},
                )

        return None  # signal caller to compute reward via _compute_reward

    # ------------------------------------------------------------------
    # Fault math
    # ------------------------------------------------------------------

    def _recompute_S_faulted(self) -> None:
        """
        Apply all active faults to the S_true matrix for the current active model.

        WRONG_EMBEDDING_MODEL is implicit: Task 3 starts with active_model=LEGAL,
        whose score distribution on medical text is fundamentally wrong (compressed,
        mean≈0.84, std≈0.033 vs GENERAL std≈0.13). The agent must diagnose this
        from retrieval score distributions and swap models. All other faults are
        applied as matrix transformations by apply_faults().
        """
        # Lock to GENERAL on tasks without WRONG_EMBEDDING_MODEL so that
        # swap_embedding_model actions don't accidentally shift coverage/precision
        # via raw score-level differences between embedding model matrices.
        fault_types = {f.fault_type for f in self._injected_faults}
        if FaultType.WRONG_EMBEDDING_MODEL in fault_types:
            model_key = _MODEL_FILE[self._active_model]
        else:
            model_key = "general"
        S = self._s_true_episode.get(model_key)
        if S is None:
            S = self._s_true_episode["general"]

        self._S_faulted = apply_faults(
            S=S,
            fault_types=fault_types,
            config_chunk_size=self._config.chunk_size,
            config_context_limit=self._config.context_window_limit,
            config_use_reranking=self._config.use_reranking,
            config_chunk_overlap=self._config.chunk_overlap,
            noise=self._noise,
            dupe_ids=self._dupe_ids,
            rewrite_boosts=self._rewrite_boosts,
        )

    # ------------------------------------------------------------------
    # Retrieval simulation
    # ------------------------------------------------------------------

    def _simulate_retrieval(self) -> List[QueryResult]:
        """Run retrieval simulation over all episode queries using S_faulted."""
        results = []
        config = self._config
        n_c = len(self._chunk_ids)

        for i, query in enumerate(self._episode_queries):
            query_id = query["query_id"]
            scores = self._S_faulted[i]  # (n_chunks,)

            # Top-K by descending score
            top_k = min(config.top_k, n_c)
            top_indices = np.argsort(scores)[::-1][:top_k]

            # Filter by threshold
            retrieved: List[Tuple[int, float]] = [
                (self._chunk_ids[j], float(scores[j]))
                for j in top_indices
                if scores[j] >= config.similarity_threshold
            ]

            retrieved_ids = [cid for cid, _ in retrieved]
            retrieved_scores = [s for _, s in retrieved]

            # Coverage and precision
            r_star = set(self._ground_truth.get(str(query_id), []))
            r_agent = set(retrieved_ids)
            coverage = len(r_agent & r_star) / len(r_star) if r_star else 0.0
            precision = len(r_agent & r_star) / len(r_agent) if r_agent else 0.0

            # Context overflow check
            total_tokens = sum(
                self._chunk_id_to_tokens.get(cid, 100)
                for cid in retrieved_ids
            )

            results.append(
                QueryResult(
                    query_id=query_id,
                    query_text=query["text"],
                    retrieved_chunk_ids=retrieved_ids,
                    retrieval_scores=retrieved_scores,
                    n_retrieved=len(retrieved_ids),
                    coverage_score=float(np.clip(coverage, 0.0, 1.0)),
                    precision_score=float(np.clip(precision, 0.0, 1.0)),
                    is_multi_hop=bool(query.get("is_multi_hop", False)),
                )
            )

        return results

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(self, results: List[QueryResult]) -> QualityMetrics:
        coverages = [r.coverage_score for r in results]
        precisions = [r.precision_score for r in results]
        n_empty = sum(1 for r in results if r.n_retrieved == 0)

        # Count context overflows by re-checking token counts
        n_overflow = 0
        config = self._config
        for r in results:
            total = sum(
                self._chunk_id_to_tokens.get(cid, 100)
                for cid in r.retrieved_chunk_ids
            )
            if total > config.context_window_limit:
                n_overflow += 1

        multi_hop_covs = [r.coverage_score for r in results if r.is_multi_hop]
        multi_hop_cov = float(np.mean(multi_hop_covs)) if multi_hop_covs else None

        return QualityMetrics(
            mean_coverage=float(np.mean(coverages)) if coverages else 0.0,
            mean_precision=float(np.mean(precisions)) if precisions else 0.0,
            mean_recall=float(np.mean(coverages)) if coverages else 0.0,
            n_empty_retrievals=n_empty,
            n_context_overflows=n_overflow,
            multi_hop_coverage=multi_hop_cov,
        )

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        prev: Optional[QualityMetrics],
        new: QualityMetrics,
        action: RAGDebugAction,
    ) -> Reward:
        """Compute a progress-based reward in [0.0, 1.0].

        Design: reward reflects the absolute quality level (progress toward the
        success threshold) PLUS a small bonus for the direction of change.

        This ensures the full [0.0, ~0.89] range is utilised for non-terminal
        steps, giving the RL agent a strong per-step learning signal:
          - Terrible state, no improvement → reward ≈ 0.09
          - Mid quality, no change        → reward ≈ 0.42
          - At success threshold, no change → reward ≈ 0.64
          - Large improvement step        → reward up to  0.89
          - Large regression + penalties  → reward clipped to 0.00

        Terminal rewards (SUBMIT) remain in their own zone [0.7, 1.0] for
        success and [0.0, 0.15] for failure, as before.
        """
        components: Dict[str, float] = {}
        n_queries = max(len(self._episode_queries), 1)

        # --- Progress reward: absolute quality level signal ---
        # Maps quality_score to [0.10, 0.65] proportional to how close we are
        # to the task's success threshold.  Spans the reward range across the
        # full episode regardless of per-step delta magnitude.
        quality_target = 0.75 if self._task_id in (1, 2) else 0.70
        current_quality = self._quality_score(new)
        progress = min(1.0, current_quality / quality_target)
        components["progress_reward"] = 0.10 + 0.55 * progress

        # --- Delta bonus: immediate direction feedback ---
        # Distinguishes an improving step from a no-op at the same quality level.
        # Multiplied delta capped at ±0.15 so a single step never dominates.
        if prev is not None:
            prev_quality = self._quality_score(prev)
            q_delta = current_quality - prev_quality
            components["delta_bonus"] = float(np.clip(q_delta * 2.0, -0.15, 0.15))

            # Empty retrieval signal: bidirectional (weight 0.06)
            empty_change = prev.n_empty_retrievals - new.n_empty_retrievals
            components["empty_retrieval_signal"] = float(np.clip(empty_change / n_queries, -1.0, 1.0)) * 0.06

            # Context overflow signal: bidirectional (weight 0.04)
            overflow_change = prev.n_context_overflows - new.n_context_overflows
            components["overflow_signal"] = float(np.clip(overflow_change / n_queries, -1.0, 1.0)) * 0.04
        else:
            components["delta_bonus"] = 0.0
            components["empty_retrieval_signal"] = 0.0
            components["overflow_signal"] = 0.0

        # --- Efficiency penalties ---
        components["step_cost"] = -0.01

        # Redundancy penalty for repeating the same action type consecutively
        if self._prev_action_type is not None and action.action_type == self._prev_action_type:
            components["redundancy_penalty"] = -0.04
        else:
            components["redundancy_penalty"] = 0.0

        # Penalty for invalid action parameters
        if self._last_action_error is not None:
            components["invalid_action_penalty"] = -0.05

        # --- Combine: no fixed base — progress_reward IS the base ---
        raw = sum(components.values())
        value = float(np.clip(raw, 0.0, 1.0))
        return Reward(value=value, components=components)

    def _compute_task_score(self, metrics: QualityMetrics) -> float:
        """Compute the scalar task score used for grading."""
        n_steps = self._state.step_count
        if self._task_id in (1, 2):
            efficiency = 1.0 - n_steps / _MAX_STEPS
            return (
                0.60 * metrics.mean_coverage
                + 0.25 * metrics.mean_precision
                + 0.15 * efficiency
            )
        else:  # task 3
            mh_cov = metrics.multi_hop_coverage or 0.0
            return (
                0.55 * metrics.mean_coverage
                + 0.25 * metrics.mean_precision
                + 0.20 * mh_cov
            )

    def _quality_score(self, metrics: QualityMetrics) -> float:
        """Quality portion of task_score, excluding efficiency. Normalized to [0, 1].

        Uses the same coverage:precision weighting as _compute_task_score so that
        step rewards are aligned with the terminal success criterion.
        """
        if self._task_id in (1, 2):
            # 0.60 + 0.25 = 0.85 max; normalize to [0, 1]
            return (0.60 * metrics.mean_coverage + 0.25 * metrics.mean_precision) / 0.85
        else:  # task 3: includes multi-hop coverage, already sums to 1.0
            mh_cov = metrics.multi_hop_coverage or 0.0
            return 0.55 * metrics.mean_coverage + 0.25 * metrics.mean_precision + 0.20 * mh_cov

    def _check_success(self, metrics: QualityMetrics, task_score: float) -> bool:
        if self._task_id == 1:
            return task_score >= 0.75
        elif self._task_id == 2:
            return task_score >= 0.75
        else:  # task 3
            mh_cov = metrics.multi_hop_coverage or 0.0
            return task_score >= 0.70 and mh_cov > 0.60

    # ------------------------------------------------------------------
    # Fault sampling
    # ------------------------------------------------------------------

    def _sample_faults(self, task_id: int, rng: np.random.Generator) -> List[FaultConfig]:
        if task_id == 1:
            idx = int(rng.integers(0, len(_TASK1_FAULT_SETS)))
            fault_types = _TASK1_FAULT_SETS[idx]
        elif task_id == 2:
            idx = int(rng.integers(0, len(_TASK2_FAULT_SETS)))
            fault_types = _TASK2_FAULT_SETS[idx]
        else:
            fault_types = _TASK3_FAULTS

        return [FaultConfig(fault_type=ft) for ft in fault_types]

    # ------------------------------------------------------------------
    # Query sampling
    # ------------------------------------------------------------------

    def _sample_queries(
        self, all_queries: List[Dict], task_id: int, rng: np.random.Generator
    ) -> List[Dict]:
        n = _N_EPISODE_QUERIES[task_id]

        if task_id == 3:
            regular = [q for q in all_queries if not q.get("is_multi_hop")]
            multi_hop = [q for q in all_queries if q.get("is_multi_hop")]

            n_mh = min(2, len(multi_hop))
            n_reg = n - n_mh

            reg_sample = list(rng.choice(len(regular), size=min(n_reg, len(regular)), replace=False))
            mh_sample = list(rng.choice(len(multi_hop), size=n_mh, replace=False))

            sampled = [regular[i] for i in reg_sample] + [multi_hop[i] for i in mh_sample]
        else:
            indices = list(rng.choice(len(all_queries), size=min(n, len(all_queries)), replace=False))
            sampled = [all_queries[i] for i in indices]

        return sampled

    def _calibrate_initial_difficulty(self, rng: np.random.Generator) -> None:
        """Nudge overly-strong reset states toward improvable starting points."""
        self._recompute_S_faulted()
        results = self._simulate_retrieval()
        metrics = self._compute_metrics(results)

        if not results:
            return

        full_cov_rate = float(
            np.mean([1.0 if r.coverage_score >= 0.999 else 0.0 for r in results])
        )
        cov_caps = {1: 0.60, 2: 0.52, 3: 0.48}
        full_cov_caps = {1: 0.50, 2: 0.45, 3: 0.40}

        cov_cap = cov_caps.get(self._task_id, 0.60)
        full_cov_cap = full_cov_caps.get(self._task_id, 0.50)
        if metrics.mean_coverage <= cov_cap and full_cov_rate <= full_cov_cap:
            return

        updates: Dict[str, Any] = {}
        if self._config.top_k > 3:
            shrink = int(rng.integers(1, 3))
            updates["top_k"] = max(3, self._config.top_k - shrink)

        fault_types_active = {f.fault_type for f in self._injected_faults}
        if FaultType.THRESHOLD_TOO_HIGH not in fault_types_active:
            bump = float(rng.uniform(0.05, 0.12))
            updates["similarity_threshold"] = min(
                0.75, self._config.similarity_threshold + bump
            )

        if updates:
            self._config = self._config.model_copy(update=updates)

    # ------------------------------------------------------------------
    # Diagnostic hints
    # ------------------------------------------------------------------

    def _generate_diagnostic_hints(
        self, metrics: QualityMetrics, results: List[QueryResult]
    ) -> List[str]:
        """Generate context-aware hints based on current metric patterns."""
        hints: List[str] = []
        cfg = self._config

        if metrics.n_empty_retrievals > 0:
            hints.append(
                f"{metrics.n_empty_retrievals} queries have empty retrievals — "
                "consider lowering similarity_threshold or increasing top_k."
            )

        if metrics.n_context_overflows > 0:
            hints.append(
                f"{metrics.n_context_overflows} queries exceed context window — "
                "consider increasing context_window_limit or reducing top_k."
            )

        if metrics.mean_coverage < 0.5 and metrics.mean_precision > 0.3:
            hints.append(
                "Low coverage with moderate precision suggests top_k is too small "
                "or the embedding model may not suit this domain."
            )

        if metrics.mean_coverage < 0.4 and metrics.mean_precision < 0.3:
            hints.append(
                "Both coverage and precision are low — check if the similarity threshold "
                "is filtering out too many chunks, or if the embedding model is mismatched."
            )

        # Check for score compression (sign of wrong embedding model or TOP_K_TOO_SMALL)
        all_scores = [s for r in results for s in r.retrieval_scores]
        if all_scores:
            score_std = float(np.std(all_scores))
            score_mean = float(np.mean(all_scores))
            if score_std < 0.05 and len(all_scores) > 3:
                hints.append(
                    f"Retrieval scores are tightly compressed (std={score_std:.3f}) — "
                    "this may indicate the wrong embedding model or score compression fault."
                )
            if score_mean > 0.7 and metrics.mean_precision < 0.5:
                hints.append(
                    "High retrieval scores but low precision — many irrelevant chunks are "
                    "scoring high. Consider enabling reranking or checking for duplicate flooding."
                )

        # Task 3 multi-hop hint
        if self._task_id == 3:
            mh_cov = metrics.multi_hop_coverage
            if mh_cov is not None and mh_cov < 0.5:
                hints.append(
                    f"Multi-hop coverage is low ({mh_cov:.3f}) — multi-hop queries need "
                    "broad retrieval. Consider increasing top_k and checking the embedding model."
                )

        # Reranking hint
        if not cfg.use_reranking and metrics.mean_precision < 0.4:
            hints.append(
                "Reranking is disabled. Enabling it can improve precision by re-scoring "
                "candidates with a cross-encoder."
            )

        return hints

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        results: List[QueryResult],
        reward: Optional[float],
    ) -> RAGDebugObservation:
        cs = self._corpus_stats_dict
        corpus_stats = CorpusStats(
            domain=Domain(self._domain),
            n_documents=cs.get("n_documents", 0),
            n_chunks=cs.get("n_chunks", len(self._chunks)),
            avg_chunk_tokens=cs.get("avg_chunk_tokens", 0),
            has_near_duplicates=bool(cs.get("has_near_duplicates", False)),
            n_queries=cs.get("n_queries", 0),
            n_multi_hop_queries=cs.get("n_multi_hop_queries", 0),
        )
        metrics = self._compute_metrics(results)
        hints = self._generate_diagnostic_hints(metrics, results)

        obs = RAGDebugObservation(
            pipeline_config=self._config,
            query_results=results,
            metrics=metrics,
            corpus_stats=corpus_stats,
            steps_taken=self._state.step_count,
            max_steps=_MAX_STEPS,
            task_id=self._task_id,
            task_description=_TASK_DESCRIPTION[self._task_id],
            done=self._done,
            last_action_error=self._last_action_error,
            diagnostic_hints=hints,
            reward_components=self._last_reward_components,
            reward=reward,
        )
        return obs


# Backward-compat alias (server/__init__.py and app.py import RagDebugEnvironment)
RagDebugEnvironment = RAGDebugEnvironment
