---
title: RAGDebugEnv
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
tags:
- openenv
---

# RAGDebugEnv

> **An OpenEnv-compatible reinforcement learning environment for training AI agents to diagnose and repair broken Retrieval-Augmented Generation pipelines.**

---

## Why This Environment Matters

Retrieval-Augmented Generation has become the architectural backbone of production AI — enterprise search, legal document analysis, clinical decision support, customer support automation, and code assistance all depend on it. But RAG pipelines break silently and often. Retrieval quality degrades without throwing exceptions. Engineers spend hours with no obvious place to look: Is the chunk size wrong? Is the embedding model mismatched to the domain? Is the similarity threshold too aggressive? Is the context window overflowing?

This is not a solved problem. There is no benchmark, no training signal, and no agent that can systematically diagnose and fix a broken RAG pipeline.

**RAGDebugEnv fills that gap.**

It simulates the exact failure modes that engineers encounter in production — wrong embedding models, misconfigured thresholds, duplicate flooding, context overflows — and wraps them in a structured RL environment where agents receive dense, per-step rewards for systematically diagnosing and fixing the pipeline. An agent trained here learns what a senior ML engineer learns over years: how retrieval quality signals map to root causes, and what interventions fix them.

The environment is designed for the research community as a reusable benchmark with:
- **Three task difficulties** spanning software, climate, and medical domains
- **Nine distinct fault types** with mathematically grounded injection mechanisms
- **~1ms per step** via precomputed similarity matrices — thousands of training episodes per minute
- **Full OpenEnv compatibility** for drop-in use with any compliant agent framework

---

## Architecture Overview

```
Agent (inference.py)
  │
  ├─► client.py  (OpenEnv EnvClient, HTTP/WebSocket)
  │
  └─► server/app.py  (FastAPI + OpenEnv HTTP server)
        │
        └─► RAGDebugEnvironment  (server/rag_debug_env_environment.py)
              │
              ├── Precomputed S_true matrices  (corpora/<domain>/S_true_*.npy)
              │     shape: (n_queries, n_chunks), dtype float32
              │     one matrix per embedding model × domain
              │
              ├── Fault injection math  (server/fault_math.py)
              │     apply_faults(S_true, config, faults) → S_faulted
              │
              └── Ground truth R* sets  (corpora/<domain>/ground_truth.json)
                    {query_id: [chunk_id, ...]}  built by cross-encoder in Stage 6
```

**Key design principle:** The environment never calls a live vector database. Every `step()` executes in ~1ms because fault injection is pure matrix arithmetic. This is what makes RL feasible: an agent can run tens of thousands of training episodes without waiting for real embedding inference.

---

## How the Corpora Were Built: The 6-Stage Pipeline

The environment's credibility rests on real documents, real embeddings, and real relevance labels. Building that took a six-stage pipeline.

### Stage 1 — Document Loading (`corpora/stages/s1_load.py`)

Raw documents are sourced from authoritative, domain-appropriate sources for each of the three task domains:

**Software domain (Task 1):**
- Python 3 official documentation (text archive, `tutorial/`, `library/`, `reference/`, `howto/` sections)
- HuggingFace documentation dataset (`m-ric/huggingface_doc`)
- Target: 50 documents, 300–5000 words each

**Climate domain (Task 2):**
- Wikipedia articles on climate topics via the Wikipedia REST API — 55 carefully chosen articles spanning core climate science (greenhouse gases, ocean acidification, permafrost), policy (Paris Agreement, carbon tax, emissions trading), energy (solar, wind, nuclear, carbon capture), and ecosystem impacts
- Wikipedia was chosen over alternatives like `climate_fever` (evidence passages of 1–3 sentences, too short) and arXiv papers (deprecated loading scripts)

**Medical domain (Task 3):**
- MedRAG/textbooks dataset — actual medical textbook chapters from Harrison's Principles, Robbins Pathology, Pathoma, Gray's Anatomy, Pharmacology by Katzung, and others. Ten consecutive passages (~1,300 words total) are aggregated per document to produce expository prose rather than isolated exam vignettes
- Wikipedia medical articles (50 articles) as a supplement — diseases, treatments, anatomy, pharmacology
- Medical textbooks were specifically chosen to expose the `WRONG_EMBEDDING_MODEL` fault: dense clinical vocabulary (receptor subtypes, metabolic pathways, cytokines) degrades severely under general-purpose embedding models, making the fault visually obvious from score distributions

Documents pass quality filters: minimum 300 words, maximum 5,000 words, at least 50% alphabetic characters.

### Stage 2 — Token-Level Chunking (`corpora/stages/s2_chunk.py`)

Documents are split using tiktoken's `cl100k_base` encoding — the same tokenizer used by most production embedding models. Token-level chunking (rather than word or character) ensures chunks never overflow a model's context window by accident.

```
chunk_size    = 512 tokens  (canonical — S_true is computed against this)
chunk_overlap = 50 tokens   (sliding window stride = 462 tokens)
min_chunk     = 100 tokens  (tail chunks shorter than this are dropped)
```

This produces the corpora used at runtime:
- **Software:** 271 chunks
- **Climate:** 612 chunks
- **Medical:** 359 chunks

The canonical chunk_size and overlap are fixed at corpus-build time. The environment's fault injection then *simulates* what happens when an agent changes these parameters — rather than re-chunking actual documents (which would take minutes per step).

### Stage 3 — Synthetic Query Generation (`corpora/stages/s3_queries.py`)

Queries are generated by GPT-4o-mini from seed chunks, then filtered by a cross-encoder. The process:

1. **Select 25 seed chunks** per domain — preferring chunks from diverse source documents, ending with a sentence boundary, and with high alphabetic density
2. **Generate 2 queries per chunk** via GPT-4o-mini:
   - `DIRECT` — a specific question the chunk alone completely answers (the answer is explicitly in the text)
   - `PARTIAL` — a question where the chunk provides essential but incomplete context
3. **Filter with cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) — only queries where the seed chunk scores ≥ 0.50 relevance are kept. This ensures every query genuinely retrieves something meaningful from its seed chunk, preventing spurious ground truth labels.

Queries are designed to sound natural ("How does insulin resistance develop?", not "According to the passage...").

### Stage 4 — Multi-Hop Query Construction (`corpora/stages/s4_multihop.py`)

Multi-hop queries are built for the medical domain only (Task 3). These are questions that *cannot* be answered by either chunk alone but *are* specifically answered when both are read together — exactly what tests cross-chunk reasoning.

The algorithm:
1. **Embed all medical chunks** with the medical embedding model in-memory
2. **Compute chunk-to-chunk cosine similarity** across the full chunk matrix
3. **Find candidate pairs** satisfying: similarity in [0.85, 0.97] (similar enough to be clinically related, but not so similar that one chunk already answers the other), from different source books, with a minimum index gap of 15, restricted to mechanism-dense books (Pathoma, Pharmacology/Katzung, Immunology/Janeway, Biochemistry/Lippincott, Physiology/Levy, Pathology/Robbins, Cell Biology/Alberts), sharing at least 1 mechanism term
4. **Prompt GPT-4o-mini** to generate a bridging question for each candidate pair, with explicit instructions that the question must be unanswerable from either chunk alone
5. **Validate with cross-encoder** — both chunks must score above threshold for the generated question
6. **Target: 5–8 multi-hop queries** per domain

Multi-hop queries use 30 specific mechanism terms as eligibility filters (receptor, cytokine, kinase, apoptosis, transporter, etc.) to ensure the selected pairs reflect genuine clinical relationships rather than surface vocabulary overlap.

### Stage 5 — Embedding & Similarity Matrix Computation (`corpora/stages/s5_embed.py`)

This is the stage that makes fast RL possible. For each domain, all chunks and queries are embedded with all four models, and cosine similarity is computed between every (query, chunk) pair:

| Model Key | HuggingFace Model | Domain Fit |
|-----------|-------------------|------------|
| `general` | `sentence-transformers/all-MiniLM-L6-v2` | Everyday text |
| `medical` | `NeuML/pubmedbert-base-embeddings` | Biomedical retrieval |
| `legal`   | `nlpaueb/legal-bert-base-uncased` | Legal corpora |
| `code`    | `sentence-transformers/multi-qa-mpnet-base-dot-v1` | Retrieval-tuned |

Each domain gets four `.npy` files:
```
corpora/<domain>/S_true_general.npy   shape: (n_queries, n_chunks), float32
corpora/<domain>/S_true_medical.npy
corpora/<domain>/S_true_legal.npy
corpora/<domain>/S_true_code.npy
```

These are the canonical ground-truth similarity matrices. Every episode loads a subset of rows (the 5 sampled queries) from the appropriate matrix. Fault injection then transforms this subset mathematically — no re-embedding ever happens at runtime.

### Stage 6 — Cross-Encoder Ground Truth Labeling (`corpora/stages/s6_grade.py`)

The ground-truth relevance set R* for each query is built using `BAAI/bge-reranker-v2-m3`, a high-quality cross-encoder reranker that scores every (query, chunk) pair. A chunk is considered relevant if its cross-encoder score exceeds 0.70.

Seed chunks are always included in R* regardless of cross-encoder score (they were explicitly chosen as the answer source for each query). For multi-hop queries, both seed chunks are always included.

After labeling, calibration stats are printed:
- Mean R* size 1–4 for direct queries (target)
- Mean R* size 2–5 for multi-hop queries (target)
- Thresholds can be adjusted and the stage re-run if R* sets are too sparse or too large

The output, `ground_truth.json`, is the oracle against which every agent action is scored.

---

## How Fast Real-Time RAG Simulation Works

The core innovation of this environment is the **precomputed similarity matrix + fault injection** approach. Here is the full reasoning chain:

**The problem with simulating real RAG for RL:**
A real RAG pipeline change — say, increasing chunk size — requires: re-chunking documents, re-embedding all chunks, rebuilding the vector index, running retrieval queries. That takes 30–120 seconds per configuration change. An RL agent that trains for 100,000 steps would need months of wall-clock time.

**The insight:**
The similarity score between a query and a chunk is a continuous function of the embedding model and chunking parameters. Rather than recomputing it exactly for every configuration, we can:
1. Precompute exact similarity scores for one canonical configuration (S_true)
2. Simulate the *effect* of configuration changes as mathematical transformations on those scores

This is `S_faulted = apply_faults(S_true, config, active_faults)`.

Each step executes as:
1. Agent takes an action (e.g., `adjust_threshold(value=0.15)`)
2. `_apply_action()` updates `PipelineConfig`
3. If the action affects similarity scores (chunk_size, chunk_overlap, context_limit, reranking, model swap), `_recompute_S_faulted()` calls `apply_faults()` — pure numpy, ~0.1ms
4. `_simulate_retrieval()` applies the threshold filter and top-k selection to S_faulted — another ~0.1ms
5. Metrics are computed, reward is calculated, observation is returned

Total per-step time: **~1ms**. This enables thousands of training episodes per minute.

The noise arrays (`noise[FaultType.CHUNK_TOO_SMALL]`, etc.) are **pre-generated at `reset()` time** and reused across all recomputations within the same episode. This guarantees that the stochastic elements of fault math are deterministic within an episode — the agent sees a consistent world as it tunes the pipeline — while being seeded differently per episode for training diversity.

---

## The Three Task Types

| | Task 1 | Task 2 | Task 3 |
|---|---|---|---|
| **Domain** | Software (Python docs + HF docs) | Climate (Wikipedia) | Medical (textbooks + Wikipedia) |
| **Difficulty** | Easy | Medium | Hard |
| **Fault Complexity** | 1–2 config faults | Compound config faults | Wrong embedding model + config faults |
| **Episode Queries** | 5 (all direct) | 5 (all direct) | 5 (3 direct + 2 multi-hop) |
| **Success Condition** | `task_score ≥ 0.75` | `task_score ≥ 0.75` | `task_score ≥ 0.70` AND `multi_hop_coverage > 0.60` |
| **Possible Faults** | CHUNK_TOO_LARGE + NO_RERANKING, THRESHOLD_TOO_HIGH, TOP_K_TOO_SMALL, CHUNK_TOO_LARGE | THRESHOLD_TOO_LOW + DUPLICATE_FLOODING, TOP_K_TOO_SMALL + CONTEXT_OVERFLOW, DUPLICATE_FLOODING, CONTEXT_OVERFLOW | WRONG_EMBEDDING_MODEL + CHUNK_TOO_LARGE + THRESHOLD_TOO_HIGH (always) |

**Task Score Formula:**

Tasks 1 & 2 reward efficiency (completing the fix in fewer steps):
```
task_score = 0.60 × mean_coverage + 0.25 × mean_precision + 0.15 × efficiency
efficiency = 1.0 − (steps_taken / max_steps)
```

Task 3 drops the efficiency bonus and adds multi-hop coverage as a first-class signal:
```
task_score = 0.55 × mean_coverage + 0.25 × mean_precision + 0.20 × multi_hop_coverage
```

Task 3 explicitly does not reward efficiency because finding the wrong-embedding-model fault often requires multiple diagnostic steps (checking if the score distribution is compressed, trying a model swap and observing the change).

**Baseline Scores:**

| Task | Random Actions | Zero-Shot Heuristic | Zero-Shot LLM (Qwen2.5-72B) | Target RL Score |
|------|---------------|---------------------|------------------------------|-----------------|
| 1 — Software | ~0.15 | ~0.50 | ~0.72 | **> 0.85** |
| 2 — Climate  | ~0.10 | ~0.45 | ~0.65 | **> 0.80** |
| 3 — Medical  | ~0.05 | ~0.35 | ~0.55 | **> 0.75** |

Task 3's low zero-shot score reflects the difficulty of identifying the WRONG_EMBEDDING_MODEL fault without systematic experimentation — exactly the kind of structured exploration that RL can learn to do reliably.

---

## The Nine Fault Types

Every fault is a mathematically grounded transformation of the S_true matrix. Faults are injected at `reset()` time and remain active throughout the episode — the agent must diagnose and compensate for them through configuration changes.

### 1. `CHUNK_TOO_LARGE` — Score Smearing via Box Filter

**What it simulates:** When chunk_size is too large, a single chunk spans multiple concepts. The embedding averages over all of them and becomes a blurred representation of the document region rather than a focused semantic unit. Retrieval scores get smeared — a chunk that is highly relevant gets diluted by the irrelevant content around it.

**Injection math:**
```python
filter_size = max(1, round(4 × config_chunk_size / 512))
S = uniform_filter1d(S, size=filter_size, axis=1, mode="nearest")
```
A 1D box filter along the chunk axis smears scores toward neighbors. Larger chunk_size → larger filter → more smearing. The agent can undo this by reducing chunk_size.

**Fix:** `adjust_chunk_size` to a smaller value (128–256 range typically works)

---

### 2. `CHUNK_TOO_SMALL` — Gaussian Noise from Unstable Embeddings

**What it simulates:** When chunk_size is very small (64–128 tokens), chunks often end mid-sentence. Embedding models trained on full sentences produce unreliable representations for sentence fragments — high variance in the embedding space manifests as score noise.

**Injection math:**
```python
overlap_reduction = min(0.5, config_chunk_overlap / 1000.0)
sigma = 0.15 × min(1.0, 512.0 / max(config_chunk_size, 64)) × (1.0 − overlap_reduction)
S = S + sigma × noise[CHUNK_TOO_SMALL]
```
Additive Gaussian noise with magnitude inversely proportional to chunk_size. Crucially, higher chunk_overlap *reduces* the noise — more overlap means more context at chunk boundaries, stabilizing the embeddings. The overlap's effect is computed dynamically every time `_recompute_S_faulted()` is called (this was a deliberate design decision requiring careful implementation).

**Fix:** `adjust_chunk_size` upward, and `adjust_chunk_overlap` to reduce boundary instability

---

### 3. `THRESHOLD_TOO_HIGH` — Score Deflation

**What it simulates:** When the corpus uses a domain-specific vocabulary that the embedding model handles poorly (e.g., medical terminology with a general model), all cosine similarity scores are systematically compressed into a low range. Relevant chunks that would normally score 0.7–0.8 now score 0.35–0.45, and any reasonable similarity threshold filters them all out.

**Injection math:**
```python
S = S × 0.55
```
Multiplicative deflation of all scores by 55%. After deflation, no chunk exceeds ~0.55 similarity, so a default threshold of 0.3 starts cutting into relevant chunks. The agent observes empty or near-empty retrievals and must lower the threshold to compensate.

**Fix:** `adjust_threshold` to a much lower value (0.05–0.15 range)

---

### 4. `THRESHOLD_TOO_LOW` — Noise Cluttering Retrieval

**What it simulates:** When the similarity threshold is too permissive, irrelevant chunks score high by chance and flood the retrieved set. This tanks precision without necessarily affecting coverage.

**Injection math:**
```python
S = S + 0.10 × noise[THRESHOLD_TOO_LOW]
```
Additive Gaussian noise lifts irrelevant chunks into the retrieval band. The agent observes high recall but poor precision and many retrieved chunks, and must raise the threshold.

**Fix:** `adjust_threshold` upward, `toggle_reranking` to filter noise

---

### 5. `TOP_K_TOO_SMALL` — Score Compression

**What it simulates:** Some retrieval backends only return a small number of candidates (top-k=2 or 3), and when the relevant chunks are not all in that tiny set, coverage collapses. The score compression simulates the effect of having so few candidates that the ranking becomes unreliable — relative differences between scores are small, making threshold-based filtering less effective.

**Injection math:**
```python
compress = 0.24 if not config_use_reranking else 0.65
S = 0.5 + (S − 0.5) × compress
```
Score range is compressed toward 0.5. Without reranking, compression is severe (24% of original spread); with reranking, a cross-encoder partially restores the true ranking signal (65% of original spread). The agent starts with `top_k=2–3` when this fault is active, so relevant chunks are simply missed.

**Fix:** `adjust_top_k` to a larger value (15–25), `toggle_reranking`

---

### 6. `DUPLICATE_FLOODING` — Boosted Duplicate Chunks

**What it simulates:** Near-duplicate documents in the corpus (e.g., the same article published in multiple places) crowd the top-k retrieved set. Relevant chunks are displaced by high-scoring duplicates that provide no additional information.

**Injection math:**
```python
boost = 0.08 if config_use_reranking else 0.20
S[:, dupe_ids] = minimum(S[:, dupe_ids] + boost, 1.0)
```
A random 14% of chunks are designated as duplicates (selected at `reset()` time). Their scores are boosted by 0.20 (or 0.08 with reranking active). The agent observes high top-k utilization but low precision — many retrieved chunks are near-duplicates of each other.

**Fix:** `toggle_reranking` (cross-encoder sharply reduces the boost effect), `adjust_top_k`

---

### 7. `CONTEXT_OVERFLOW` — Tail Column Zeroing

**What it simulates:** When the context window limit is too small, chunks beyond the cutoff cannot be included in the LLM context window even if retrieved. The environment simulates this by zeroing out similarity scores for chunks whose index falls beyond the context cutoff.

**Injection math:**
```python
cutoff = max(1, int(n_chunks × config_context_limit / 16384))
S[:, cutoff:] = 0.0
```
Chunks beyond the cutoff position are effectively invisible to retrieval — their scores are zeroed. The cutoff scales proportionally to `context_window_limit`, so increasing the context limit moves the cutoff further out, restoring access to more chunks.

**Fix:** `adjust_context_limit` to a larger value (8192–16384 range)

---

### 8. `NO_RERANKING` — Additive Noise Without Cross-Encoder

**What it simulates:** Without a cross-encoder reranker, the retrieval relies solely on approximate vector similarity, which is inherently noisier than the full query-document attention a cross-encoder uses. Minor score perturbations can cause relevant chunks to fall just below the threshold.

**Injection math:**
```python
if not config_use_reranking:
    S = S + 0.10 × noise[NO_RERANKING]
```
Mild additive Gaussian noise is applied only when reranking is disabled. Enabling reranking skips this fault entirely and additionally blends faulted scores back toward S_true (a 35% blending toward the pre-fault matrix, simulating the cross-encoder's ability to recover true relevance).

**Fix:** `toggle_reranking(enabled=true)`

---

### 9. `WRONG_EMBEDDING_MODEL` — Fundamentally Wrong Score Distribution

**What it simulates:** Using a general-purpose embedding model on a specialized domain (e.g., `all-MiniLM-L6-v2` on medical textbook content) produces systematically poor retrieval. The model's vocabulary and training distribution do not align with the domain — clinical terms like "receptor", "cytokine", and "metabolic pathway" are underrepresented in its training data, resulting in compressed, unreliable similarity scores.

**Injection mechanism:** Unlike other faults, WRONG_EMBEDDING_MODEL is **implicit** — Task 3 starts with `embedding_model=LEGAL` (the LEGAL model for medical data, the most mismatched combination). The `_recompute_S_faulted()` method selects which S_true matrix to load:

```python
if WRONG_EMBEDDING_MODEL in active_faults:
    model_key = _MODEL_FILE[self._active_model]  # uses whatever model is currently active
else:
    model_key = "general"  # locked to general on tasks 1 & 2
```

The LEGAL model's S_true matrix on medical text has a very different score distribution (compressed range, lower mean relevance for truly relevant chunks) compared to the MEDICAL model. The agent observes this as compressed retrieval scores with low variance (`std < 0.05`) across per-query results — a diagnostic hint the system explicitly surfaces.

**Fix:** `swap_embedding_model(model="medical")` or `swap_embedding_model(model="general")`

---

### Cross-Encoder Reranking Simulation

When `use_reranking=True`, `apply_faults()` blends the faulted matrix back toward the pre-fault scores:

```python
if config_use_reranking:
    rerank_alpha = 0.35
    S = (1.0 − rerank_alpha) × S + rerank_alpha × S_clean
```

This is a principled simulation of a cross-encoder's effect: it operates on the full query-document pair (seeing both together rather than separately encoded), which partially undoes noise-based corruption and restores compressed score spreads.

---

## What the Agent Observes

Every `step()` returns a `RAGDebugObservation` with:

### `pipeline_config` — Current Configuration
The full set of tunable parameters:
```
chunk_size           int    64–2048      current: 512
chunk_overlap        int    0–500        current: 50
similarity_threshold float  0.0–1.0     current: 0.30
top_k                int    1–50         current: 10
embedding_model      enum   general|medical|legal|code
use_reranking        bool               current: false
context_window_limit int    512–16384   current: 4096
```

### `query_results` — Per-Query Retrieval Results
For each of the 5 episode queries:
```
query_id             int
query_text           str
retrieved_chunk_ids  List[int]
retrieval_scores     List[float]
n_retrieved          int
coverage_score       float  |R_agent ∩ R*| / |R*|
precision_score      float  |R_agent ∩ R*| / |R_agent|
is_multi_hop         bool
```

### `metrics` — Aggregate Quality
```
mean_coverage       float   mean of per-query coverage_score
mean_precision      float   mean of per-query precision_score
mean_recall         float   numerically equal to mean_coverage (tracked separately for clarity)
n_empty_retrievals  int     queries where n_retrieved == 0
n_context_overflows int     queries where token sum exceeds context_window_limit
multi_hop_coverage  float?  mean coverage on multi-hop queries only (None for Tasks 1 & 2)
```

### `diagnostic_hints` — Context-Aware Hints
The environment generates up to 3 natural-language hints based on the current metric pattern:
- "N queries have empty retrievals — lower threshold or increase top_k"
- "Score variance is low (std < 0.05) — possible wrong embedding model"
- "Context overflow detected — increase context_window_limit"
- "Coverage low but precision decent — top_k may be too small"

### `reward_components` — Named Reward Breakdown
The full component decomposition of the last step's reward — useful for debugging agent behavior and understanding which aspects of the pipeline the reward function is tracking.

### `last_action_error` — Invalid Action Feedback
If the agent attempted an invalid configuration (e.g., `chunk_overlap >= chunk_size`, out-of-range values, unknown embedding model name), this field contains the validation error message. The agent is expected to learn to avoid such errors.

### Intentional omissions
The `injected_faults` list is **never exposed** in the observation. The agent must infer the fault type purely from the metric signatures — that inference IS the task.

---

## The Agent's Action Space

| Action | Parameters | Valid Range | Effect |
|--------|-----------|-------------|--------|
| `adjust_chunk_size` | `{"value": int}` | 64–2048 | Changes chunk size; modulates CHUNK_TOO_LARGE smearing severity |
| `adjust_chunk_overlap` | `{"value": int}` | 0–500 | Changes overlap; reduces CHUNK_TOO_SMALL noise at boundaries |
| `adjust_threshold` | `{"value": float}` | 0.0–1.0 | Threshold filter for retrieved chunks |
| `adjust_top_k` | `{"value": int}` | 1–50 | Number of candidates to retrieve per query |
| `swap_embedding_model` | `{"model": str}` | general/medical/legal/code | Switches the active embedding model |
| `toggle_reranking` | `{"enabled": bool}` | — | Enables/disables cross-encoder reranking simulation |
| `adjust_context_limit` | `{"value": int}` | 512–16384 | Shifts the context overflow cutoff |
| `rewrite_query` | `{"query_id": int, "strategy": str}` | strategy: "rephrase" | Boosts a specific query's scores toward R* by +0.20 |
| `submit` | `{}` | — | Ends the episode and triggers grading |

**Important constraint:** `chunk_overlap` must be strictly less than `chunk_size`. The environment validates this via Pydantic's model validator and returns a `last_action_error` if violated — the config is not updated in that case.

**REWRITE_QUERY mechanics:** Internally, the environment adds a persistent +0.20 boost overlay to the similarity scores of R* chunks for the specified query. This simulates query expansion or reformulation improving recall for a specific question.

---

## How the Reward Function Works

All rewards are bounded to **[0.0, 1.0]**. The reward design deliberately avoids sparse terminal-only rewards — every step provides a learning signal tied to the current state of the pipeline.

### Non-Terminal Step Reward Components

**`progress_reward` — Absolute Quality Level Signal (range: [0.10, 0.65])**

```python
quality_target = 0.75  # (0.70 for task 3)
current_quality = quality_score(new_metrics)
progress = min(1.0, current_quality / quality_target)
progress_reward = 0.10 + 0.55 × progress
```

This is the primary signal. Even a terrible state (quality ≈ 0) receives 0.10 rather than 0.00 — ensuring a gradient everywhere. At the success threshold, progress_reward reaches 0.65. This ensures the full reward range is utilized across the episode rather than being concentrated in the last few steps.

**`delta_bonus` — Direction Signal (range: [-0.15, +0.15])**

```python
q_delta = current_quality − prev_quality
delta_bonus = clip(q_delta × 2.0, −0.15, +0.15)
```

Distinguishes an improving step from a no-op at the same quality level. A large positive improvement gives +0.15; a large regression gives -0.15. The ×2.0 amplification and ±0.15 cap ensure individual steps cannot dominate the reward signal.

**`empty_retrieval_signal` — Bidirectional Empty Retrieval Feedback (range: [-0.06, +0.06])**

```python
empty_change = prev.n_empty_retrievals − new.n_empty_retrievals
empty_retrieval_signal = clip(empty_change / n_queries, −1.0, +1.0) × 0.06
```

Positive when empty retrievals decrease (queries are now returning results). Negative when empty retrievals increase. Normalized by the total number of queries.

**`overflow_signal` — Bidirectional Overflow Feedback (range: [-0.04, +0.04])**

Identical structure to `empty_retrieval_signal`, tracking context overflows instead.

**`step_cost` — Efficiency Pressure (fixed: -0.01)**

A small fixed cost per step that encourages the agent to solve tasks efficiently rather than taking unnecessary actions.

**`redundancy_penalty` (−0.04 if triggered)**

Applied when the agent takes the same action type consecutively. Discourages uninformative repeat actions like adjusting threshold twice in a row.

**`invalid_action_penalty` (−0.05 if triggered)**

Applied when an action violates configuration constraints (out of range, chunk_overlap ≥ chunk_size, etc.).

### Reward Combination

```python
raw = sum(all_components.values())
value = clip(raw, 0.0, 1.0)
```

Typical step reward ranges:
- Terrible state, no improvement: **≈ 0.09**
- Mid quality, no change: **≈ 0.42**
- At success threshold: **≈ 0.64**
- Large improvement step: **up to 0.89**
- Large regression with penalties: **clipped to 0.00**

### Terminal Reward (SUBMIT)

When the agent calls `submit`, the environment computes the final task score and returns a terminal reward in its own zone, clearly separable from non-terminal rewards:

```python
if check_success(metrics, task_score):
    terminal_value = clip(0.7 + 0.3 × task_score, 0.7, 1.0)   # SUCCESS: [0.7, 1.0]
else:
    terminal_value = clip(0.2 × task_score, 0.0, 0.2)           # FAILURE: [0.0, 0.2]
```

Success and failure zones are disjoint — an agent cannot confuse a strong terminal failure with a weak terminal success. Submitting immediately on an unimproved episode typically yields a terminal reward of ~0.04–0.08.

---

## How `reset()` Ensures a Clean Episode

`reset(seed, task_id)` is a comprehensive initialization that makes every episode reproducible and independent:

```
1. Reset bookkeeping
   _done = False, _prev_action_type = None, _last_action_error = None
   _state = State(episode_id=new_uuid, step_count=0)

2. Seed the RNG
   rng = np.random.default_rng(seed)
   All randomness in the episode flows from this single seed.

3. Load domain corpus
   chunks, queries, ground_truth, corpus_stats, s_true matrices

4. Sample 5 episode queries
   Task 3: 3 direct + 2 multi-hop
   Tasks 1 & 2: 5 direct queries

5. Slice S_true to episode query rows
   _s_true_episode[model_name] = s_true_full[ep_query_row_indices, :]

6. Pre-generate noise arrays (unit normal, deterministic from seed)
   _noise[CHUNK_TOO_SMALL]   shape: (n_queries, n_chunks)
   _noise[THRESHOLD_TOO_LOW] shape: (n_queries, n_chunks)
   _noise[NO_RERANKING]      shape: (n_queries, n_chunks)
   _dupe_ids = random 14% of chunk indices (for DUPLICATE_FLOODING)

7. Initialize config
   PipelineConfig() with defaults
   Task 3: set embedding_model=LEGAL (the wrong model for medical data)
   Randomize top_k (5–8) and threshold (0.34–0.48) from seed

8. Fault-specific config nudges
   TOP_K_TOO_SMALL active → top_k = rng.integers(2, 4)
   DUPLICATE_FLOODING active → top_k = rng.integers(4, 8)

9. Sample and inject faults
   Task 1: one of 4 possible fault configurations (randomly selected)
   Task 2: one of 4 possible compound fault configurations
   Task 3: always [WRONG_EMBEDDING_MODEL, CHUNK_TOO_LARGE, THRESHOLD_TOO_HIGH]

10. Calibrate initial difficulty
    Compute initial metrics; if coverage already exceeds task cap,
    nudge threshold upward and reduce top_k to ensure the episode
    starts in an improvable state.

11. Initial S_faulted computation
    _recompute_S_faulted() → _S_faulted = apply_faults(S_true, config, faults)

12. Return initial observation
    Computes query results, metrics, diagnostic hints, and returns
    the full RAGDebugObservation for the agent's first step.
```

The noise pre-generation in step 6 is particularly important: stochastic fault components are stable within an episode (the agent sees a consistent world) but vary across episodes (the agent must generalize). The `seed` parameter makes any specific episode fully reproducible for debugging.

---

## GRPO Training Scaffold (`outputs/train_grpo.py`)

The repository includes a working Group Relative Policy Optimization scaffold for training agents using the OpenAI-compatible API as the policy.

GRPO is well-suited to this environment because:
- Episodes provide a dense scalar reward at every step (not just terminal)
- The environment runs at ~1ms/step, enabling many rollouts per batch
- GRPO requires no value network — it normalizes rewards within the group

The core normalization:
```python
def grpo_normalize(rollouts):
    rewards = [r.total_reward for r in rollouts]
    mean_r = sum(rewards) / len(rewards)
    variance = sum((r - mean_r)**2 for r in rewards) / len(rewards)
    std_r = variance ** 0.5
    for rollout in rollouts:
        rollout.normalized_reward = (rollout.total_reward - mean_r) / (std_r + 1e-8)
```

Rollouts better than the group average get positive normalized rewards; those worse get negative. This relative baseline is what makes GRPO work without a learned value function.

Training data is saved to `outputs/grpo_data.jsonl` in a format directly compatible with TRL's `GRPOTrainer` for gradient-based fine-tuning of smaller models (e.g., Qwen2.5-1.5B).

```bash
python outputs/train_grpo.py --task 1 --batches 3 --group-size 4
python outputs/train_grpo.py --task all --batches 2 --group-size 4
```

---

## Setup & Running

### Environment Variables

```bash
export API_BASE_URL="https://router.huggingface.co/v1"   # LLM endpoint
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"            # Model identifier
export HF_TOKEN="your-hugging-face-token"                 # API key
```

### Install & Run

```bash
# Install dependencies
uv sync

# Start the environment server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run inference (in a separate terminal)
python inference.py

# Run GRPO training scaffold
python outputs/train_grpo.py --task 1 --batches 3 --group-size 4

# Run the test suite
uv run python -m pytest tests/ -v
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

### Rebuild the Corpus (optional)

The prebuilt corpus artifacts are included in the repository. To rebuild from scratch:

```bash
export OPENAI_API_KEY="your-openai-key"   # required for Stage 3 (query gen) and Stage 4 (multi-hop)
python -m corpora.build_corpus --domain all
```

This runs all 6 stages for all domains sequentially. Individual domains can be rebuilt with `--domain software`, `--domain climate`, or `--domain medical`. Each stage caches its output, so stages can be re-run incrementally with `--force-reload`.

---

## Repository Layout

```
rag_debug_env/
│
├── inference.py                    # Competition inference script ([START]/[STEP]/[END] logging)
├── client.py                       # OpenEnv client (WebSocket-based)
├── models.py                       # Typed Pydantic models (Action, Observation, Config, etc.)
├── openenv.yaml                    # OpenEnv manifest
├── pyproject.toml                  # Package metadata and uv dependencies
├── Dockerfile                      # Multi-stage Docker build for HF Spaces
│
├── server/
│   ├── app.py                      # FastAPI application (create_app with OpenEnv)
│   ├── rag_debug_env_environment.py # Core environment: reset(), step(), reward, fault routing
│   ├── constants.py                # Task definitions, fault sets, thresholds
│   ├── fault_math.py               # Pure fault injection math: apply_faults(S_true) → S_faulted
│   └── corpus.py                   # Corpus loader with synthetic fallback
│
├── corpora/
│   ├── build_corpus.py             # Orchestrates all 6 stages
│   ├── stages/
│   │   ├── s1_load.py              # Stage 1: Load raw documents
│   │   ├── s2_chunk.py             # Stage 2: Token-level chunking
│   │   ├── s3_queries.py           # Stage 3: GPT-4o-mini query generation + cross-encoder filter
│   │   ├── s4_multihop.py          # Stage 4: Multi-hop query construction (medical only)
│   │   ├── s5_embed.py             # Stage 5: Embed all 4 models, save S_true matrices
│   │   └── s6_grade.py             # Stage 6: Cross-encoder R* labeling → ground_truth.json
│   ├── software/
│   │   ├── chunks.json             # 271 chunks
│   │   ├── queries.json            # 48 queries
│   │   ├── ground_truth.json       # R* sets for all queries
│   │   ├── S_true_general.npy      # (48, 271) float32
│   │   ├── S_true_medical.npy
│   │   ├── S_true_legal.npy
│   │   └── S_true_code.npy
│   ├── climate/                    # 612 chunks, 44 queries
│   └── medical/                    # 359 chunks, 44 queries (6 multi-hop)
│
├── outputs/
│   ├── eval_agent.py               # GPT-4o-mini zero-shot evaluation agent
│   └── train_grpo.py               # GRPO training scaffold (full rollout + normalization)
│
├── tests/
│   ├── test_fault_math.py          # 13 tests: per-fault transformation correctness
│   ├── test_reward.py              # Reward bounds, terminal rewards, component correctness
│   ├── test_environment.py         # Episode lifecycle, action routing, bug-fix verification
│   └── test_stdout_format.py       # [START]/[STEP]/[END] format compliance
│
└── docs/
    ├── ARCHITECTURE.md             # Detailed architecture documentation
    └── MODELS_REFERENCE.md         # Embedding model details
```

---

## Example Agent Interaction

```
[START] task=task_1 env=rag_debug_env model=Qwen/Qwen2.5-72B-Instruct

Observation:
  Config:  chunk_size=512 overlap=50 threshold=0.42 top_k=6 model=general reranking=false
  Metrics: coverage=0.340 precision=0.280 empty=2 overflow=0
  Hints:   "2 queries have empty retrievals — lower threshold or increase top_k"

Step 1: adjust_threshold(value=0.15)
  Metrics: coverage=0.620 precision=0.450 empty=0 overflow=0  reward=0.52

Step 2: toggle_reranking(enabled=true)
  Metrics: coverage=0.720 precision=0.580 empty=0 overflow=0  reward=0.60

Step 3: adjust_top_k(value=15)
  Metrics: coverage=0.840 precision=0.610 empty=0 overflow=0  reward=0.66

Step 4: submit()
  task_score = 0.60×0.840 + 0.25×0.610 + 0.15×(1 - 4/10) = 0.504 + 0.153 + 0.090 = 0.747
  SUCCESS ✓  terminal_reward = 0.7 + 0.3×0.747 = 0.924

[END] success=true steps=4 score=0.747 rewards=0.52,0.60,0.66,0.92
```

---

## Test Coverage

122 tests across 4 files verify correctness of every layer:

| File | Tests | What it covers |
|------|-------|----------------|
| `test_fault_math.py` | 13 | Per-fault transformation correctness, bounds, non-mutation |
| `test_reward.py` | ~50 | Reward bounds (fuzz tested), terminal ranges, component logic, task score formulas |
| `test_environment.py` | ~45 | Episode lifecycle, all 9 action types, ADJUST_CHUNK_OVERLAP bug fix verification |
| `test_stdout_format.py` | ~25 | [START]/[STEP]/[END] field names, ordering, numeric precision |

```bash
uv run python -m pytest tests/ -v
# 122 passed in 1.96s
```
