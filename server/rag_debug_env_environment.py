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
from openenv.core.env_server.types import State

from server.constants import (
    _TASK_DOMAIN,
    _TASK_DESCRIPTION,
    _N_EPISODE_QUERIES,
    _MAX_STEPS,
    _SUCCESS_COVERAGE,
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

        # Route action
        reward_value = self._apply_action(action)

        # Recompute retrieval results and metrics
        new_results = self._simulate_retrieval()
        new_metrics = self._compute_metrics(new_results)

        # Compute reward (if not already set by SUBMIT handler)
        if reward_value is None:
            reward_value = self._compute_reward(prev_metrics, new_metrics, action)

        self._internal_state.action_history.append(action)
        self._internal_state.reward_history.append(reward_value)

        self._prev_metrics = new_metrics
        self._prev_action_type = action.action_type

        # Auto-terminate on max steps
        if self._state.step_count >= _MAX_STEPS and not self._done:
            self._done = True

        return self._build_observation(new_results, reward=reward_value)

    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    # Action routing
    # ------------------------------------------------------------------

    def _apply_action(self, action: RAGDebugAction) -> Optional[float]:
        """
        Apply action to config / overlays.
        Returns override reward if the action is SUBMIT, else None.
        """
        t = action.action_type
        p = action.params

        if t == ActionType.ADJUST_CHUNK_SIZE:
            value = int(p.get("value", self._config.chunk_size))
            try:
                self._config = self._config.model_copy(update={"chunk_size": value})
            except Exception:
                pass  # invalid param; no-op
            self._recompute_S_faulted()

        elif t == ActionType.ADJUST_CHUNK_OVERLAP:
            value = int(p.get("value", self._config.chunk_overlap))
            try:
                self._config = self._config.model_copy(update={"chunk_overlap": value})
            except Exception:
                pass
            # Overlap has no direct matrix effect; no recompute needed

        elif t == ActionType.ADJUST_THRESHOLD:
            value = float(p.get("value", self._config.similarity_threshold))
            try:
                self._config = self._config.model_copy(update={"similarity_threshold": value})
            except Exception:
                pass
            # Threshold is applied in retrieval simulation; no matrix recompute

        elif t == ActionType.ADJUST_TOP_K:
            value = int(p.get("value", self._config.top_k))
            try:
                self._config = self._config.model_copy(update={"top_k": value})
            except Exception:
                pass

        elif t == ActionType.ADJUST_CONTEXT_LIMIT:
            value = int(p.get("value", self._config.context_window_limit))
            try:
                self._config = self._config.model_copy(update={"context_window_limit": value})
            except Exception:
                pass
            self._recompute_S_faulted()

        elif t == ActionType.SWAP_EMBEDDING_MODEL:
            model_str = str(p.get("model", self._active_model.value))
            try:
                new_model = EmbeddingModel(model_str)
                self._active_model = new_model
                self._config = self._config.model_copy(update={"embedding_model": new_model})
            except Exception:
                pass
            self._recompute_S_faulted()

        elif t == ActionType.TOGGLE_RERANKING:
            enabled = bool(p.get("enabled", not self._config.use_reranking))
            self._config = self._config.model_copy(update={"use_reranking": enabled})
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
            # Compute final metrics (already computed above, passed as prev_metrics here)
            # We'll use self._prev_metrics because _apply_action is called before metrics update
            # But at SUBMIT time we want the *current* state, so do a retrieval now
            results = self._simulate_retrieval()
            metrics = self._compute_metrics(results)
            task_score = self._compute_task_score(metrics)
            threshold = _SUCCESS_COVERAGE[self._task_id]
            self._done = True

            # Check task-specific success condition
            success = self._check_success(metrics, task_score)
            if success:
                return 2.0  # terminal_bonus
            else:
                return -0.50  # premature_submit_penalty

        return None  # signal caller to compute reward normally

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
                self._chunk_tokens[self._chunk_ids.index(cid)]
                for cid in retrieved_ids
                if cid in self._chunk_ids
            )
            is_overflow = total_tokens > config.context_window_limit

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
                self._chunk_tokens[self._chunk_ids.index(cid)]
                for cid in r.retrieved_chunk_ids
                if cid in self._chunk_ids
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
    ) -> float:
        components: Dict[str, float] = {}

        if prev is not None:
            components["coverage_delta"] = (new.mean_coverage - prev.mean_coverage) * 0.6
            components["precision_delta"] = (new.mean_precision - prev.mean_precision) * 0.3
        else:
            components["coverage_delta"] = 0.0
            components["precision_delta"] = 0.0

        components["step_cost"] = -0.02

        # Redundancy penalty for repeating the same action type
        if self._prev_action_type is not None and action.action_type == self._prev_action_type:
            components["redundancy_penalty"] = -0.10
        else:
            components["redundancy_penalty"] = 0.0

        # Penalty for newly introduced empty retrievals
        if prev is not None:
            new_empties = max(0, new.n_empty_retrievals - prev.n_empty_retrievals)
            components["empty_retrieval_penalty"] = -0.15 * new_empties
        else:
            components["empty_retrieval_penalty"] = 0.0

        return sum(components.values())

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
        )
        obs.reward = reward
        return obs


# Backward-compat alias (server/__init__.py and app.py import RagDebugEnvironment)
RagDebugEnvironment = RAGDebugEnvironment
