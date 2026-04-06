# RAGDebugEnv — Claude Code Context

## What This Project Is

An OpenEnv-compliant RL environment for training agents to debug broken RAG (Retrieval-Augmented Generation) pipelines. Agents observe retrieval quality metrics and take corrective actions (adjust chunk size, similarity threshold, top-k, swap embedding model, etc.) to restore pipeline performance.

Submitted to the **AgentBeats OpenEnv Hackathon** hosted by MetaAI, HuggingFace, and Scaler School of Technology.

**The core value proposition:** Train agents in simulation, deploy against real production RAG pipelines. The observation schema is identical whether the backend is simulated (NumPy matrices) or real (Pinecone/Weaviate/Chroma).

---

## Current Build Status

See `docs/BUILD_STATUS.md` for full detail. Summary:

| Phase | Status | Files |
|---|---|---|
| Phase 0 — Skeleton | ✅ Complete | `pyproject.toml`, `openenv.yaml`, `Dockerfile` |
| Phase 1 — Models | ✅ Complete | `models.py`, `tests/test_models.py` |
| Phase 2a — Server + Environment | ✅ Complete | `server/rag_debug_env_environment.py`, `server/app.py` |
| Phase 2b — Client | ✅ Complete | `client.py`, `__init__.py` |
| Phase 3 — Corpus Build (Stages 1-5) | ✅ Complete | `corpora/stages/s1_load.py` ... `corpora/stages/s5_embed.py` |
| Phase 3 — Corpus Build (Stage 6) | ⏳ Pending | `corpora/stages/s6_grade.py` |
| Phase 4 — Tests | ⏳ Not started | `tests/test_environment.py`, `tests/test_faults.py` |
| Phase 5 — Baseline Script | ⏳ Not started | `baseline/run_baseline.py` |
| Phase 6 — HF Deployment | ⏳ Not started | |

---

## Project File Structure

```
rag_debug_env/
├── CLAUDE.md                      ← you are here
├── docs/
│   ├── BUILD_STATUS.md            ← detailed phase-by-phase status
│   ├── ARCHITECTURE.md            ← system design and key decisions
│   ├── MODELS_REFERENCE.md        ← every model field explained precisely
│   └── CORPUS_BUILD_PLAN.md       ← Stage 1-6 plan for build_corpus.py
├── models.py                      ← ALL Pydantic models (Tier 1 + Tier 2)
├── client.py                      ← RAGDebugEnv(HTTPEnvClient) 
├── __init__.py                    ← public surface: RAGDebugEnv, RAGDebugAction, RAGDebugObservation
├── server/
│   ├── __init__.py
│   ├── rag_debug_env_environment.py ← RAGDebugEnvironment(Environment) — core logic
│   └── app.py                     ← FastAPI app via create_env_app()
├── corpora/
│   ├── __init__.py
│   ├── build_corpus.py            ← orchestrator for Stage 1-5 (Stage 6 optional/pending)
│   └── stages/
│       ├── __init__.py
│       ├── s1_load.py             ← ✅ Stage 1: document loading (working)
│       ├── s2_chunk.py            ← ✅ Stage 2: chunking with tiktoken
│       ├── s3_queries.py          ← ✅ Stage 3: synthetic query generation via GPT-4o-mini
│       ├── s4_multihop.py         ← ✅ Stage 4: multi-hop query construction (medical)
│       ├── s5_embed.py            ← ✅ Stage 5: embedding + S_true matrix computation
│       ├── s6_grade.py            ← ⏳ Stage 6: cross-encoder R* labeling
│       └── verify.py              ← ⏳ verification / sanity checks
├── tests/
│   ├── __init__.py
│   ├── test_models.py             ← ✅ 35+ tests, all passing
│   ├── test_environment.py        ← written, not yet run against real env
│   └── test_faults.py             ← written, not yet run against real env
├── baseline/
│   └── run_baseline.py            ← ⏳ NOT YET WRITTEN
├── openenv.yaml                   ← OpenEnv manifest
├── pyproject.toml                 ← dependencies
└── Dockerfile                     ← container definition
```

---

## OpenEnv Spec — Critical Facts

**Package name:** `openenv-core` (install with `pip install openenv-core`)

**Correct import paths (confirmed from official docs):**
```python
from openenv.core.env_server.types import Action, Observation, State
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server import create_env_app
from openenv.core.http_env_client import HTTPEnvClient
from openenv.core.types import StepResult
```

**Our classes:**
- `RAGDebugAction(Action)` — inherits from OpenEnv Action ✅
- `RAGDebugObservation(Observation)` — inherits from OpenEnv Observation ✅
- `RAGDebugEnvironment(Environment)` — server-side logic ✅
- `RAGDebugEnv(HTTPEnvClient[RAGDebugAction, RAGDebugObservation])` — client ✅

**App creation pattern:**
```python
app = create_env_app(env, RAGDebugAction, RAGDebugObservation)
```

**State:** Use `openenv.core.env_server.types.State` directly for `episode_id` and `step_count`. Domain-specific state lives in our `InternalState(BaseModel)`.

**Required file structure per OpenEnv spec:**
```
env_name/
├── __init__.py        # exports Action, Observation, EnvClient
├── models.py          # Action + Observation subclasses
├── client.py          # HTTPEnvClient subclass
├── openenv.yaml       # manifest
├── pyproject.toml
└── server/
    ├── app.py         # create_env_app(...)
    ├── environment.py # Environment subclass
    ├── requirements.txt
    └── Dockerfile
```

---

## The Three Tasks

| Task | Domain | Faults | Max Steps | Success Threshold |
|---|---|---|---|---|
| 1 — SingleFaultFix | Software (Python docs) | 1 | 10 | mean_coverage > 0.80 |
| 2 — CompoundFaultFix | Climate (Wikipedia) | 2 interacting | 15 | mean_coverage > 0.75 |
| 3 — MultiHopDebug | Medical (textbooks) | 3 incl. wrong embedding | 20 | mean_coverage > 0.70 |

---

## The Simulation Architecture — Most Important Concept

**The environment never runs a real RAG pipeline during episodes.** All expensive work happens once in `build_corpus.py`. During episodes, everything is pure NumPy.

```
build_corpus.py (one-time):
  Documents → Chunks → Embed (4 models) → S_true_[model].npy
  Chunks + Queries → Cross-encoder → ground_truth.json

Episode runtime (milliseconds):
  S_faulted = apply_fault_math(S_true_general)
  R_agent = threshold_filter(top_k(S_faulted, config))
  coverage = |R_agent ∩ R*| / |R*|
  reward = delta(coverage) × weights
```

**S_true matrices:** shape (n_queries, n_chunks), one per embedding model. Stored as `.npy`.

**R* (ground truth):** `{query_id: [relevant_chunk_ids]}`. Computed once by cross-encoder. Model-independent — this is what the grader trusts.

**The key insight:** S_true captures what each embedding model *perceives* as similar. R* captures what is *actually* relevant. The gap between them IS the learning signal for the wrong_embedding_model fault.

---

## Corpus Build — Current State

Stages 1-5 are implemented and producing cached artifacts for all target domains
(`software`, `climate`, `medical`):

- `docs.json`, `chunks.json`, `queries.json`
- `S_true_general.npy`, `S_true_medical.npy`, `S_true_legal.npy`, `S_true_code.npy`

Stage 6 (`s6_grade.py`) and `verify.py` are the remaining corpus-build pieces.

---

## Key Design Decisions (Don't Change These)

1. **Procedural fault generation, not a fixed dataset.** `reset()` samples faults randomly each episode. Infinite variety, no memorization.

2. **Four S_true matrices, one per embedding model.** Swapping embedding model during episode = swapping which matrix is used. The `WRONG_EMBEDDING_MODEL` fault scrambles scores for domain-specific queries on the GENERAL matrix.

3. **Cross-encoder for R*, bi-encoder for S_true.** Cross-encoder is slow but accurate and model-agnostic. Bi-encoder is fast but model-dependent. Never use S_true to define R* — that would make the grader circular.

4. **Delta-based rewards.** Reward = Δcoverage × 0.6 + Δprecision × 0.3 − step_cost + terminal_bonus. Agent gets signal at every improvement, not just episode end.

5. **`InternalState` never sent to agent.** Agent must infer faults from metrics alone. This IS the task.

6. **Synthetic corpus fallback.** `RAGDebugEnvironment` generates synthetic NumPy data if corpus files don't exist. Allows tests to run before `build_corpus.py` completes.

---

## Environment Variables Required

```bash
OPENAI_API_KEY=sk-...        # Required for corpus build (Stages 3, 4)
CORPORA_DIR=corpora          # Optional, defaults to this path
ENABLE_WEB_INTERFACE=true    # Optional, enables OpenEnv web UI at /web
HF_TOKEN=hf_...              # Optional, avoids HF rate limits
```

---

## Running Things

```bash
# Install
pip install openenv-core
pip install -e .

# Run tests (work without corpus files — uses synthetic fallback)
pytest tests/ -v

# Build corpus (requires OPENAI_API_KEY)
python -m corpora.build_corpus --domain software
python -m corpora.build_corpus --domain climate
python -m corpora.build_corpus --domain medical

# Run server locally
uvicorn rag_debug_env.server.app:app --host 0.0.0.0 --port 8000

# Docker
docker build -t rag_debug_env .
docker run -p 8000:8000 rag_debug_env

# Validate OpenEnv compliance
openenv validate
```

---

## What To Work On Next

1. **Write `s6_grade.py`** — cross-encoder R* labeling
2. **Write `verify.py`** — corpus sanity checks and clean-pipeline coverage checks
3. **Run full corpus verification** for `software`, `climate`, `medical`
4. **Validate server environment against real corpus artifacts** (replace synthetic fallback path in tests)
5. **Write `baseline/run_baseline.py`** — GPT-4o agent with structured outputs

See `docs/CORPUS_BUILD_PLAN.md` for detailed Stage 2-6 specs.
