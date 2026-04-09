"""
models.py
---------
Defines the typed Action and Observation for RAGDebugEnv, plus all
internal simulation models used by the environment logic.

Architecture
------------
Two tiers of models live here:

  Tier 1 — OpenEnv interface types (must inherit from framework bases)
    RAGDebugAction     inherits openenv.core.env_server.types.Action
    RAGDebugObservation inherits openenv.core.env_server.types.Observation

  Tier 2 — Internal simulation models (plain Pydantic BaseModel)
    PipelineConfig, QueryResult, QualityMetrics, CorpusStats, Reward,
    InternalState, EpisodeResult

  The OpenEnv-provided State class is used directly for episode
  metadata (episode_id, step_count). It is NOT subclassed — the
  framework owns that contract.

Import convention
-----------------
    from models import RAGDebugAction, RAGDebugObservation
  from openenv.core.env_server.types import State   # for episode state
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, field_validator, model_validator

# ── OpenEnv base types ────────────────────────────────────────────────────────
# These are the two types the framework requires us to subclass.
# Import path confirmed from official docs:
# https://meta-pytorch.org/OpenEnv/environment-builder/
from openenv.core.env_server.types import Action, Observation


# =============================================================================
# Enums shared across both tiers
# =============================================================================

class EmbeddingModel(str, Enum):
    """
    The four embedding models the pipeline can use.

    GENERAL  — sentence-transformers/all-MiniLM-L6-v2.
               Fast, general-purpose.
               Works well on everyday text but degrades on specialist domains.
    MEDICAL  — NeuML/pubmedbert-base-embeddings.
               Trained on biomedical retrieval tasks.
    LEGAL    — nlpaueb/legal-bert-base-uncased.  Trained on legal corpora.
    CODE     — sentence-transformers/multi-qa-mpnet-base-dot-v1.
               Retrieval-tuned contrast model (keeps historical "code" slot).
    """
    GENERAL = "general"
    MEDICAL = "medical"
    LEGAL   = "legal"
    CODE    = "code"


class Domain(str, Enum):
    """
    The corpus domain for each task difficulty.

    SOFTWARE — Python docs.  Clean prose, unambiguous vocabulary.  Task 1.
    CLIMATE  — IPCC reports.  Cross-disciplinary, more ambiguous.  Task 2.
    MEDICAL  — MedRAG textbooks.  Heavy domain terminology.  Task 3.
    """
    SOFTWARE = "software"
    CLIMATE  = "climate"
    MEDICAL  = "medical"


class ActionType(str, Enum):
    """
    Every action the agent can take against the pipeline.

    Config actions modify PipelineConfig in-place. The environment
    re-simulates retrieval on the updated config immediately.

    REWRITE_QUERY rewrites one query's text — simulated by
    perturbing its similarity scores toward the ground-truth set.

    SUBMIT declares the agent is done.  Triggers grading.
    Submitting before the success threshold incurs a penalty.
    """
    ADJUST_CHUNK_SIZE    = "adjust_chunk_size"
    ADJUST_CHUNK_OVERLAP = "adjust_chunk_overlap"
    ADJUST_THRESHOLD     = "adjust_threshold"
    ADJUST_TOP_K         = "adjust_top_k"
    SWAP_EMBEDDING_MODEL = "swap_embedding_model"
    TOGGLE_RERANKING     = "toggle_reranking"
    ADJUST_CONTEXT_LIMIT = "adjust_context_limit"
    REWRITE_QUERY        = "rewrite_query"
    SUBMIT               = "submit"


class FaultType(str, Enum):
    """
    Every fault that can be injected into the simulated pipeline.
    Stored in InternalState.  Never exposed in RAGDebugObservation.
    """
    CHUNK_TOO_LARGE       = "chunk_too_large"
    CHUNK_TOO_SMALL       = "chunk_too_small"
    THRESHOLD_TOO_LOW     = "threshold_too_low"
    THRESHOLD_TOO_HIGH    = "threshold_too_high"
    TOP_K_TOO_SMALL       = "top_k_too_small"
    CONTEXT_OVERFLOW      = "context_overflow"
    DUPLICATE_FLOODING    = "duplicate_flooding"
    WRONG_EMBEDDING_MODEL = "wrong_embedding_model"
    NO_RERANKING          = "no_reranking"


# =============================================================================
# Tier 1 — OpenEnv interface types
# =============================================================================

class RAGDebugAction(Action):
    """
    The action an agent takes against the RAG pipeline.

    Inherits from openenv.core.env_server.types.Action as required by
    the OpenEnv spec.  The framework uses this class for serialisation,
    deserialisation, and web-UI form generation.

    action_type selects the operation.  params carries its arguments.

    Parameter schemas by action_type
    ---------------------------------
    adjust_chunk_size     {"value": int}        64 ≤ value ≤ 2048
    adjust_chunk_overlap  {"value": int}        0 ≤ value ≤ 500
    adjust_threshold      {"value": float}      0.0 ≤ value ≤ 1.0
    adjust_top_k          {"value": int}        1 ≤ value ≤ 50
    swap_embedding_model  {"model": str}        EmbeddingModel enum value
    toggle_reranking      {"enabled": bool}
    adjust_context_limit  {"value": int}        512 ≤ value ≤ 16384
    rewrite_query         {"query_id": int,
                           "strategy": str}    currently only "rephrase" is supported
    submit                {}
    """
    action_type: ActionType = Field(
        ...,
        description="Which pipeline operation to perform.",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for the chosen action_type.",
    )

    @field_validator("params", mode="before")
    @classmethod
    def coerce_params_dict(cls, value: Any) -> Dict[str, Any]:
        """Accept dicts and JSON-stringified dicts from the web UI."""
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError("params must be a dictionary or valid JSON object string") from exc
            if not isinstance(parsed, dict):
                raise ValueError("params JSON must decode to an object")
            return parsed
        raise TypeError("params must be a dictionary or JSON object string")

    def __str__(self) -> str:
        if self.params:
            param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
            return f"{self.action_type.value}({param_str})"
        return f"{self.action_type.value}()"


class RAGDebugObservation(Observation):
    """
    Everything the agent is allowed to see after each step.

    Inherits from openenv.core.env_server.types.Observation as required
    by the OpenEnv spec.

    Intentional omissions
    ---------------------
    injected_faults is NOT here.  The agent must infer faults from
    metrics alone — that reasoning IS the task.  Faults are only
    revealed in InternalState (accessible via env.state(), used by
    graders and debuggers, not given to the agent).

    Fields
    ------
    pipeline_config   The current parameter set the agent may modify.
    query_results     Per-query retrieval results under current config.
    metrics           Aggregate quality metrics across all queries.
    corpus_stats      Static metadata about the corpus (domain, size).
    steps_taken       Actions taken so far this episode.
    max_steps         Budget before the episode force-terminates.
    task_id           1 = easy, 2 = medium, 3 = hard.
    task_description  Plain-language objective for the agent's prompt.
    done              True once the episode has ended.
    """
    pipeline_config:  PipelineConfig  = Field(
        ..., description="Current pipeline configuration the agent can modify."
    )
    query_results:    List[QueryResult] = Field(
        ..., description="Per-query retrieval results under the current config."
    )
    metrics:          QualityMetrics  = Field(
        ..., description="Aggregate retrieval quality metrics."
    )
    corpus_stats:     CorpusStats     = Field(
        ..., description="Static metadata about the corpus for this episode."
    )
    steps_taken:      int             = Field(
        ..., description="Number of actions taken so far this episode."
    )
    max_steps:        int             = Field(
        ..., description="Maximum actions allowed before episode force-terminates."
    )
    task_id:          int             = Field(
        ..., description="Task identifier: 1 = easy, 2 = medium, 3 = hard."
    )
    task_description: str             = Field(
        ..., description="Plain-language objective for the agent."
    )
    done:             bool            = Field(
        False, description="True once the episode has ended."
    )
    last_action_error: Optional[str]  = Field(
        None, description="Error message if the last action was invalid or failed."
    )
    diagnostic_hints:  List[str]      = Field(
        default_factory=list,
        description="Context-aware diagnostic hints based on current metric patterns.",
    )
    reward_components: Dict[str, float] = Field(
        default_factory=dict,
        description="Named breakdown of the reward signal for interpretability.",
    )


# =============================================================================
# Tier 2 — Internal simulation models  (plain Pydantic BaseModel)
# =============================================================================

# ── Pipeline Configuration ────────────────────────────────────────────────────

class PipelineConfig(BaseModel):
    """
    The complete set of parameters defining the RAG pipeline's behaviour.

    These are the knobs the agent turns.  Every RAGDebugAction ultimately
    modifies one field here (or switches the active embedding model, which
    swaps which S_true matrix is used in simulation).

    Bounds reflect real-world sensible ranges.  The validator enforces
    that overlap < chunk_size because an overlap equal to chunk_size
    would produce infinite identical chunks.
    """
    chunk_size:           int            = Field(512,  ge=64,   le=2048)
    chunk_overlap:        int            = Field(50,   ge=0,    le=500)
    similarity_threshold: float          = Field(0.3,  ge=0.0,  le=1.0)
    top_k:                int            = Field(10,   ge=1,    le=50)
    embedding_model:      EmbeddingModel = EmbeddingModel.GENERAL
    use_reranking:        bool           = False
    context_window_limit: int            = Field(4096, ge=512,  le=16384)

    @model_validator(mode="after")
    def overlap_less_than_chunk_size(self) -> "PipelineConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be "
                f"strictly less than chunk_size ({self.chunk_size})"
            )
        return self


# ── Per-Query Results ─────────────────────────────────────────────────────────

class QueryResult(BaseModel):
    """
    Retrieval outcome for a single query under the current config.

    retrieved_chunk_ids and retrieval_scores are parallel — index i of
    each list refers to the same chunk.

    coverage_score = |R_agent ∩ R*| / |R*|
      1.0 → all relevant chunks retrieved
      0.0 → no relevant chunks retrieved

    is_multi_hop flags queries that require two chunks to answer
    (relevant for Task 3 grading only).
    """
    query_id:            int
    query_text:          str
    retrieved_chunk_ids: List[int]
    retrieval_scores:    List[float]
    n_retrieved:         int
    coverage_score:      float = Field(ge=0.0, le=1.0)
    precision_score:     float = Field(ge=0.0, le=1.0)
    is_multi_hop:        bool  = False


# ── Aggregate Metrics ─────────────────────────────────────────────────────────

class QualityMetrics(BaseModel):
    """
    Aggregate retrieval quality across all queries in the episode.

    mean_coverage        Primary signal.  Mean of per-query coverage scores.
    mean_precision       Fraction of retrieved chunks that are relevant.
    mean_recall          Fraction of relevant chunks that were retrieved.
                         Numerically equals mean_coverage when R* is the
                         ground-truth set, but tracked separately for clarity.
    n_empty_retrievals   Queries where nothing passed the threshold filter.
    n_context_overflows  Queries where retrieved chunks exceeded limit.
    multi_hop_coverage   Mean coverage on multi-hop queries only.
                         None when no multi-hop queries exist (Tasks 1 & 2).
    """
    mean_coverage:      float          = Field(ge=0.0, le=1.0)
    mean_precision:     float          = Field(ge=0.0, le=1.0)
    mean_recall:        float          = Field(ge=0.0, le=1.0)
    n_empty_retrievals: int            = Field(ge=0)
    n_context_overflows: int           = Field(ge=0)
    multi_hop_coverage: Optional[float] = Field(None, ge=0.0, le=1.0)


# ── Corpus Metadata ───────────────────────────────────────────────────────────

class CorpusStats(BaseModel):
    """
    Static metadata about the corpus for this episode.
    Gives the agent context about the data it's working with.
    """
    domain:              Domain
    n_documents:         int
    n_chunks:            int
    avg_chunk_tokens:    int
    has_near_duplicates: bool
    n_queries:           int
    n_multi_hop_queries: int


# ── Reward ────────────────────────────────────────────────────────────────────

class Reward(BaseModel):
    """
    The reward signal produced by env.step().

    All rewards are in [0.0, 1.0].  Non-terminal step rewards span
    [0.0, ~0.89] based on absolute quality progress; terminal rewards
    occupy [0.7, 1.0] (success) or [0.0, 0.15] (failure).

    value is the scalar used by the RL algorithm.

    components is a labelled breakdown for interpretability.  The
    environment always populates this — it aids debugging and makes
    reward shaping decisions auditable.

    Non-terminal step components
    ----------------------------
    progress_reward         0.10 + 0.55 × progress → [0.10, 0.65]
                            progress = min(1, quality_score / quality_target)
                            Absolute quality level signal; ensures full reward
                            range is utilised across the episode.
    delta_bonus             clip(Δquality × 2.0, −0.15, +0.15)
                            Direction signal: distinguishes an improving step
                            from a no-op at the same quality level.
    empty_retrieval_signal  Bidirectional: rewards fixing empties, penalizes new ones, weight 0.06
    overflow_signal         Bidirectional: rewards fixing overflows, penalizes new ones, weight 0.04
    step_cost               Fixed -0.01 per step (efficiency pressure)
    redundancy_penalty      -0.04 if same action type taken twice consecutively
    invalid_action_penalty  -0.05 if the action had invalid parameters

    Terminal SUBMIT components
    --------------------------
    terminal_success        0.7 + 0.3 × task_score → [0.7, 1.0] on successful SUBMIT
    terminal_failure        0.2 × task_score → [0.0, 0.2] on premature SUBMIT
    """
    value:      float
    components: Dict[str, float] = Field(default_factory=dict)

    def __str__(self) -> str:
        parts = ", ".join(f"{k}={v:+.3f}" for k, v in self.components.items())
        return f"Reward(total={self.value:+.3f} | {parts})"


# ── Fault Config (internal, never sent to agent) ──────────────────────────────

class FaultConfig(BaseModel):
    """
    Parameters of a single injected fault.
    Stored in InternalState.  Never included in RAGDebugObservation.
    """
    fault_type:  FaultType
    params:      Dict[str, Any] = Field(default_factory=dict)
    description: str = ""


# ── Internal State (server-side only) ─────────────────────────────────────────

class InternalState(BaseModel):
    """
    Full server-side state of the environment.

    Returned by env.state() and used by graders and the
    RealPipelineBackend adapter.  NOT given to the agent during training.

    The OpenEnv framework's State class (with episode_id and step_count)
    is used alongside this for the parts the framework owns.  This class
    carries the domain-specific internal state.
    """
    injected_faults: List[FaultConfig]
    episode_seed:    int
    action_history:  List[RAGDebugAction] = Field(default_factory=list)
    reward_history:  List[float]          = Field(default_factory=list)

    @property
    def total_reward(self) -> float:
        return sum(self.reward_history)

    @property
    def fault_names(self) -> List[str]:
        return [f.fault_type.value for f in self.injected_faults]


# ── Episode Result (post-episode summary) ────────────────────────────────────

class EpisodeResult(BaseModel):
    """
    Summary returned by env.grade() after a completed episode.

    task_score     0.0–1.0 from the task's grader function.
    success        True if task_score >= the task's success_threshold.
    fault_names    Which faults were injected (revealed post-episode).
    """
    task_id:        int
    task_score:     float = Field(ge=0.0, le=1.0)
    success:        bool
    n_steps:        int
    total_reward:   float
    final_metrics:  QualityMetrics
    fault_names:    List[str]
    action_history: List[RAGDebugAction]


# =============================================================================
# Rebuild forward references
# =============================================================================
# RAGDebugObservation references PipelineConfig, QueryResult, QualityMetrics,
# and CorpusStats which are defined after it in the file.  model_rebuild()
# resolves those forward refs.
RAGDebugObservation.model_rebuild()
InternalState.model_rebuild()
EpisodeResult.model_rebuild()