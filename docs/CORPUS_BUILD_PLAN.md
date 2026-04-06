# Corpus Build Plan

## Overview

`build_corpus.py` runs once, offline, before any training. It produces for each domain:

```
corpora/
  software/
    chunks.json           ← all chunks with metadata
    queries.json          ← all queries (direct + multi-hop)
    ground_truth.json     ← {query_id: [relevant_chunk_ids]}
    S_true_general.npy    ← (n_queries × n_chunks) float32
    S_true_medical.npy
    S_true_legal.npy
    S_true_code.npy
    corpus_stats.json     ← CorpusStats fields
  climate/
    [same structure]
  medical/
    [same structure]
```

---

## Stage 1 — Document Loading ✅ DONE

`corpora/stages/s1_load.py` — working for software and medical, climate needs verification.

**Target:** 40-60 documents per domain, 300-5000 words each.

**Sources:**
- Software: Python 3.12 docs zip + m-ric/huggingface_doc
- Climate: Wikipedia articles via REST API (55 article titles)
- Medical: MedRAG/textbooks (Harrison's, Robbins, etc., 8 passages aggregated per doc)

**Stage 1 output:** list of `{"text": str, "source": str, "domain": str}`

---

## Stage 2 — Chunking ⏳

**File:** `corpora/stages/s2_chunk.py`

**Why tokens not words:** Embedding models have a `max_tokens` limit, not a max_words limit. One token ≈ 0.75 words on average, but varies widely — "ChatGPT" is 1 token, "supercalifragilistic" is 4. Using word count would produce chunks that randomly overflow the model's context window. Tiktoken gives exact token counts.

**Config:** chunk_size=512 tokens, chunk_overlap=50 tokens (canonical config — S_true is computed against this)

**Algorithm:**
```python
def chunk_document(text: str, doc_meta: dict) -> list[dict]:
    enc = tiktoken.get_encoding("cl100k_base")  # same as GPT-4
    tokens = enc.encode(text)
    
    chunks = []
    stride = chunk_size - chunk_overlap  # = 462
    
    for i in range(0, len(tokens), stride):
        chunk_tokens = tokens[i : i + chunk_size]
        if len(chunk_tokens) < MIN_CHUNK_TOKENS:  # = 100, skip stub chunks
            break
        chunk_text = enc.decode(chunk_tokens)
        chunks.append({
            "chunk_id":    global_chunk_counter,
            "text":        chunk_text,
            "n_tokens":    len(chunk_tokens),
            "source_doc":  doc_meta["source"],
            "domain":      doc_meta["domain"],
            "token_start": i,
            "token_end":   i + len(chunk_tokens),
        })
        global_chunk_counter += 1
    
    return chunks
```

**Output:** `chunks.json` — list of chunk dicts

**Verification:** After chunking, print:
- Total chunks per domain (target: 200-400)
- Average tokens per chunk (should be ~480-512)
- First chunk text — read it, verify it's coherent prose not truncated mid-sentence

---

## Stage 3 — Synthetic Query Generation ⏳

**File:** `corpora/stages/s3_queries.py`

**Why reverse-engineer queries from chunks:** Guarantees at least one relevant chunk exists for every query (the seed chunk). Avoids annotation entirely.

**Chunk selection heuristic:**
- Skip chunks with < 150 tokens (too sparse)
- Skip chunks with < 50% alphabetic characters (code blocks, tables)
- Prefer chunks with sentence-ending punctuation (complete thoughts)
- Select ~20-30 chunks per domain

**GPT-4o-mini prompt:**
```python
QUERY_GEN_PROMPT = """
You are building a retrieval benchmark.

Given this text chunk, generate exactly 2 questions:
1. DIRECT: A specific question that this chunk alone completely answers.
   The answer must be explicitly stated in the chunk, not inferred.
2. PARTIAL: A question where this chunk provides essential but incomplete information.
   The full answer requires reading this chunk plus at least one related chunk.

Domain: {domain}
Chunk text:
{chunk_text}

Rules:
- Questions must sound natural, like a real user would ask
- Questions must NOT be answerable from general knowledge alone
- Questions must NOT ask for a list of items (too easy to satisfy partially)
- Do NOT reference "the text" or "the passage" — phrase as standalone questions

Respond in JSON only, no preamble:
{{"direct": "...", "partial": "..."}}
"""
```

**Output per chunk:**
```python
{
    "query_id":       int,
    "text":           str,
    "type":           "direct" | "partial",
    "seed_chunk_id":  int,         # chunk this was generated from
    "is_multi_hop":   False,       # True only after Stage 4
    "domain":         str,
    "difficulty":     "easy" | "medium"
}
```

**Quality filter:** After generation, run each (query, seed_chunk) pair through the cross-encoder. If score < 0.5, the query doesn't actually match its seed chunk — discard it.

**Output:** `queries.json` (partial — multi-hop added in Stage 4)

---

## Stage 4 — Multi-Hop Query Construction ⏳

**File:** `corpora/stages/s4_multihop.py`

**Only needed for medical domain (Task 3).** Software and climate get 0 multi-hop queries.

**Algorithm:**
1. Embed all chunks with the GENERAL model (already done in Stage 5 — can reuse)
2. Compute chunk-to-chunk cosine similarity: `S_chunks = cosine_similarity(chunk_vecs, chunk_vecs)`
3. Find candidate pairs: `0.35 < S_chunks[i][j] < 0.60` AND from different source documents
   - Too similar (> 0.60): chunks are probably redundant, one alone likely answers any question
   - Too dissimilar (< 0.35): no natural bridge question exists
4. For each candidate pair, prompt GPT-4o-mini to generate a bridging question
5. Keep pairs where the model succeeds (doesn't return "SKIP")

**GPT-4o-mini prompt:**
```python
MULTIHOP_PROMPT = """
You are building a multi-hop retrieval benchmark.

Chunk A:
{chunk_a}

Chunk B:
{chunk_b}

Generate ONE question that:
- CANNOT be fully answered by Chunk A alone
- CANNOT be fully answered by Chunk B alone
- IS completely and specifically answered when BOTH chunks are read together
- Sounds like a natural question a medical professional might ask

If no such question naturally exists between these two chunks, respond with exactly: SKIP

Otherwise respond with just the question, no preamble.
"""
```

**Output per multi-hop query:**
```python
{
    "query_id":        int,
    "text":            str,
    "type":            "multi_hop",
    "seed_chunk_ids":  [chunk_id_a, chunk_id_b],
    "is_multi_hop":    True,
    "domain":          "medical",
    "difficulty":      "hard"
}
```

Target: 5-8 multi-hop queries for the medical domain.

---

## Stage 5 — Embedding + S_true Computation ⏳

**File:** `corpora/stages/s5_embed.py`

**The four embedding models:**
```python
EMBEDDING_MODELS = {
    "general": "sentence-transformers/all-MiniLM-L6-v2",    # 384-dim, fast
    "medical": "pritamdeka/S-PubMedBert-MS-MARCO",          # biomedical
    "legal":   "nlpaueb/legal-bert-base-uncased",            # legal
    "code":    "microsoft/codebert-base",                    # code + docstrings
}
```

**Note on model names:** Verify these model names exist on HuggingFace before running. The exact model string matters for `SentenceTransformer()`.

**Algorithm:**
```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

for model_name, model_path in EMBEDDING_MODELS.items():
    model = SentenceTransformer(model_path)
    
    chunk_texts = [c["text"] for c in chunks]
    query_texts = [q["text"] for q in queries]
    
    # Encode in batches to avoid OOM
    chunk_vecs = model.encode(chunk_texts, batch_size=32, show_progress_bar=True)
    query_vecs = model.encode(query_texts, batch_size=32, show_progress_bar=True)
    
    # (n_queries, n_chunks) float32
    S_true = cosine_similarity(query_vecs, chunk_vecs).astype(np.float32)
    
    np.save(out_dir / f"S_true_{model_name}.npy", S_true)
```

**Verification:** For each model, pick a test query and print top-5 retrieved chunk texts. Read them. For the GENERAL model on medical text, the top chunks should look somewhat relevant but miss nuanced domain-specific matches. For the MEDICAL model on the same query, the top chunks should be more precisely relevant.

**Expected time:** 5-15 minutes per domain depending on hardware. Run once, never again.

---

## Stage 6 — Cross-Encoder R* Labeling ⏳

**File:** `corpora/stages/s6_grade.py`

**Model:** `BAAI/bge-reranker-large` — trained on multi-domain corpora including scientific and medical text. Consistently outperforms MS-MARCO-only MiniLM models on BEIR benchmarks (the standard for passage relevance). R* is computed once and never changes, so accuracy here matters more than speed.

CPU fallback: `BAAI/bge-reranker-base` (~278M params) if the large model is too slow. Minimum acceptable: `cross-encoder/ms-marco-MiniLM-L-12-v2` if hardware is very constrained, but avoid MiniLM-L-6 — it was trained only on web search passages and underperforms on medical/scientific text.

**Algorithm:**
```python
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("BAAI/bge-reranker-large")
ground_truth = {}

for query in queries:
    # Score every chunk against this query
    pairs = [(query["text"], chunk["text"]) for chunk in chunks]
    scores = cross_encoder.predict(pairs, batch_size=32, show_progress_bar=False)

    # Collect relevant chunks above threshold
    relevant_ids = [
        chunks[i]["chunk_id"]
        for i, score in enumerate(scores)
        if score > RELEVANCE_THRESHOLD  # = 0.70
    ]

    # Always include seed chunk(s) regardless of cross-encoder score
    seed_ids = query.get("seed_chunk_ids") or [query["seed_chunk_id"]]
    relevant_ids = list(set(relevant_ids + seed_ids))

    ground_truth[query["query_id"]] = relevant_ids

# Calibration check
mean_r_star_size = sum(len(v) for v in ground_truth.values()) / len(ground_truth)
print(f"Mean R* size: {mean_r_star_size:.2f}")
# Target: 1-4 for direct queries, 2-5 for multi-hop
# If < 0.8: threshold too strict, lower RELEVANCE_THRESHOLD to 0.60
# If > 8:   threshold too loose, raise RELEVANCE_THRESHOLD to 0.80
```

**RELEVANCE_THRESHOLD = 0.70** is the default. Calibrate by checking mean R* size after the first run.

**Expected time:** For 50 queries × 300 chunks = 15,000 pairs: ~5 min on GPU, ~20–30 min on CPU (bge-reranker-large). Run once, never again.

---

## Verification Step ⏳

**File:** `corpora/stages/verify.py`

After all stages, run these checks:

**1. Clean pipeline coverage check:**
```python
# Load S_true_general and ground_truth
# Simulate retrieval with default PipelineConfig
# Compute coverage across all queries
# MUST be >= 0.75 for software, >= 0.65 for climate/medical
# If lower: cross-encoder threshold was too strict (R* too small) 
#           OR embedding model is severely misaligned
```

**2. Fault degradation check:**
```python
# For each fault type, apply it and check coverage drops
# MUST see meaningful degradation (> 0.15 drop) for all fault types
# If a fault doesn't degrade coverage, it won't teach the agent anything
```

**3. Manual spot check:**
```python
# Print 5 random queries with their R* chunks
# Read them — does R* make intuitive sense?
# Are there obviously relevant chunks missing from R*?
# Are there clearly irrelevant chunks included in R*?
```

**4. Stats summary:**
```python
# Print for each domain:
# - n_documents, n_chunks, n_queries, n_multi_hop_queries
# - Mean R* size
# - S_true shape
# - Clean coverage score
```

---

## After Corpus Build

Once all domains pass verification, the environment's synthetic fallback is no longer needed. The environment will load real data automatically from `CORPORA_DIR`.

Run the full test suite:
```bash
pytest tests/ -v
```

Then write the baseline script (`baseline/run_baseline.py`).
