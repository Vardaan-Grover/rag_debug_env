# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Training Loop                        │
│  Agent observes → picks action → receives reward        │
└─────────────────┬───────────────────────────────────────┘
                  │ RAGDebugAction
                  ▼
┌─────────────────────────────────────────────────────────┐
│              RAGDebugEnv (client.py)                    │
│         HTTPEnvClient — WebSocket to server             │
└─────────────────┬───────────────────────────────────────┘
                  │ WebSocket /ws
                  ▼
┌─────────────────────────────────────────────────────────┐
│              FastAPI Server (server/app.py)             │
│         create_env_app(env, Action, Observation)        │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ RAGDebugEnvironment (server/rag_debug_env_environment.py) │
│                                                         │
│  reset() → load corpus → sample faults → S_faulted      │
│  step()  → apply action → re-simulate → compute reward  │
│  state   → OpenEnv State(episode_id, step_count)        │
│                                                         │
│  Internal:                                              │
│    S_true_[model].npy  ← loaded from disk at reset()    │
│    S_faulted           ← S_true + fault math            │
│    ground_truth.json   ← R* for grading                 │
│    InternalState       ← faults, history (not sent out) │
└─────────────────────────────────────────────────────────┘
```

---

## Simulation Strategy — Why It's Fast

Real RAG pipelines are too slow for RL training (5-10 seconds per step). The environment simulates pipeline behavior mathematically instead.

**Offline (build_corpus.py — runs once):**
1. Load documents for each domain
2. Chunk with canonical config (chunk_size=512, overlap=50)
3. Embed all chunks with 4 embedding models → `S_true_[model].npy`
4. Generate synthetic queries from chunks via GPT-4o-mini
5. Run cross-encoder on all (query, chunk) pairs → `ground_truth.json` (R*)

**Online (episodes — milliseconds per step):**
1. Load `.npy` files into memory at `reset()`
2. Apply fault math to S_true → S_faulted
3. Simulate retrieval: `threshold_filter(argsort(S_faulted)[-top_k:])`
4. Grade: `coverage = |R_agent ∩ R*| / |R*|`
5. Reward: `Δcoverage × 0.6 + Δprecision × 0.3 − step_cost`

Result: ~1ms per step, enabling millions of training episodes.

---

## The Two Matrices

### S_true (bi-encoder similarity)
- Shape: (n_queries, n_chunks), dtype float32
- Content: `cosine_similarity(query_embedding, chunk_embedding)` 
- One file per embedding model: `S_true_general.npy`, `S_true_medical.npy`, etc.
- **What it represents:** how the embedding model perceives similarity
- **Used for:** simulating retrieval behavior under different configs

### R* (cross-encoder ground truth)
- Shape: `{query_id: [chunk_id, chunk_id, ...]}`
- Content: chunk IDs that are genuinely relevant, determined by cross-encoder
- One file per domain: `ground_truth.json`
- **What it represents:** actual relevance, independent of any embedding model
- **Used for:** grading coverage, computing reward

**Critical distinction:** Never define R* using S_true. That would make the grader circular — you'd be grading how well the pipeline satisfies its own (potentially flawed) perception of relevance, not actual relevance.

---

## Fault Simulation Math

Each fault is a mathematical transformation of S_true → S_faulted.

| Fault | Transformation |
|---|---|
| CHUNK_TOO_LARGE | `scipy.ndimage.uniform_filter1d(S, size=scale, axis=1)` — averages neighboring scores, diluting relevance |
| CHUNK_TOO_SMALL | `S + normal(0, 0.15)` — noise causes fine-grained chunks to be retrieved inconsistently |
| THRESHOLD_TOO_LOW | `S + normal(0, 0.20)` — noise elevates irrelevant chunks above threshold |
| THRESHOLD_TOO_HIGH | `S × 0.55` — deflates all scores, almost nothing passes the high threshold |
| TOP_K_TOO_SMALL | `0.5 + (S - 0.5) × 0.3` — compresses score range, making ranking differences tiny |
| DUPLICATE_FLOODING | `S[:, dupe_ids] += 0.35` — artificially boosts random chunk subset |
| CONTEXT_OVERFLOW | `S[:, cutoff:] = 0.0` — zeroes out high-index chunks (simulates truncation) |
| NO_RERANKING | `S + normal(0, 0.10)` — moderate noise without second-pass cleanup |
| WRONG_EMBEDDING_MODEL | `permute_rows(S[domain_queries])` — scrambles scores for domain-specific queries |

**Compound faults:** Applied sequentially. Each fault's output becomes the next fault's input. The interactions produce emergent degradation — e.g. `CHUNK_TOO_LARGE` + `NO_RERANKING` compounds because large chunks produce noisy initial retrieval AND there's no second pass to clean it.

**Action effects on faults:**
- `SWAP_EMBEDDING_MODEL(MEDICAL)` → switch to `S_true_medical`, re-apply non-model faults
- `TOGGLE_RERANKING(True)` → if `NO_RERANKING` was a fault, re-apply faults minus that one
- Config actions (`ADJUST_THRESHOLD`, etc.) → change `self._config`, re-simulate retrieval on same `S_faulted`

---

## Reward Function

Dense reward across the full trajectory:

```python
coverage_delta   = Δmean_coverage × 0.6
precision_delta  = Δmean_precision × 0.3
step_cost        = -0.02  # every step
redundancy       = -0.10  # same action_type twice in a row
empty_retrieval  = -0.15 × new_empty_retrievals
terminal_bonus   = +2.0   # on successful SUBMIT
premature_submit = -0.50  # SUBMIT before threshold
```

**Why delta-based:** Agent gets signal at every improvement, not just episode end. Partial fixes are acknowledged. This enables learning in early training when the agent rarely completes tasks successfully.

**Why asymmetric:** Missing an attack (empty retrieval) costs more than over-retrieving (precision loss). This mirrors real-world priorities — a pipeline that retrieves nothing is worse than one that retrieves some noise.

---

## Real-World Deployment (Sim-to-Real)

After training, the same agent can be pointed at a real pipeline via `RealPipelineBackend`:

```
User's vector store (Pinecone/Weaviate/Chroma)
  → proxy metrics computed from real retrieval
  → same RAGDebugObservation schema
  → trained agent applies learned policy
  → recommends action sequence
  → user validates
```

The agent doesn't know whether it's talking to simulation or reality — it just sees the observation schema it trained on.

**Key challenge:** Real pipelines need proxy metrics instead of true coverage (no ground truth). Observable proxies:
- Score distribution statistics (mean, variance, percentiles)
- Empty retrieval rate
- Score drop-off from rank 1 to rank K
- Retrieved chunk length distribution

`RealPipelineBackend` is stubbed as an abstract class in `server/rag_debug_env_environment.py` to signal this future extension.

---

## Three Task Designs

### Task 1 — SingleFaultFix (Easy)
- One fault, uniform sampling from 9 fault types
- Software domain (Python docs + HF docs) — clean, unambiguous vocabulary
- 3 queries, max 10 steps
- Designed to be solvable by any reasonably capable LLM baseline

### Task 2 — CompoundFaultFix (Medium)
- Two interacting faults from curated pairs:
  - `(CHUNK_TOO_LARGE, NO_RERANKING)`
  - `(THRESHOLD_TOO_LOW, DUPLICATE_FLOODING)`
  - `(TOP_K_TOO_SMALL, CONTEXT_OVERFLOW)`
- Climate domain — cross-disciplinary vocabulary, more ambiguity
- 5 queries, max 15 steps
- Key mechanic: fixing fault A raises coverage from 0.2→0.5; only then does fault B become the bottleneck

### Task 3 — MultiHopDebug (Hard)
- Three faults: `WRONG_EMBEDDING_MODEL` + `CHUNK_TOO_LARGE` + `THRESHOLD_TOO_LOW`
- Medical domain — heavy domain-specific terminology (Harrison's, Robbins)
- 5 queries, 2 of which are multi-hop (require 2 chunks each)
- Key mechanic: config fixes alone plateau at ~0.50 coverage; only swapping to MEDICAL embedding unlocks further improvement
- `multi_hop_coverage` tracked separately in QualityMetrics and scored in grade()
