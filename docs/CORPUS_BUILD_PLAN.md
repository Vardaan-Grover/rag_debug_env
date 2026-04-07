# Corpus Build Plan

This plan reflects the current implementation in `corpora/build_corpus.py` and `corpora/stages/*`.

## Pipeline Overview

`python -m corpora.build_corpus --domain <software|climate|medical|all>` executes:

1. Stage 1: document loading
2. Stage 2: chunking
3. Stage 3: synthetic query generation
4. Stage 4: multi-hop query generation (medical only)
5. Stage 5: embedding and similarity matrix build
6. Stage 6: cross-encoder relevance labeling
7. Verification: corpus sanity checks

Environment requirement:

- `OPENAI_API_KEY` is required (Stages 3 and 4)

Primary output directory:

- `corpora/<domain>/`

## Output Artifacts Per Domain

- `docs.json`
- `chunks.json`
- `queries.json`
- `ground_truth.json`
- `S_true_general.npy`
- `S_true_medical.npy`
- `S_true_legal.npy`
- `S_true_code.npy`
- `corpus_stats.json`

## Stage 1 - Load Documents (`s1_load.py`)

Implemented behavior:

- Caches to `corpora/<domain>/docs.json`
- Returns list items with `text`, `source`, `domain`

Domain-specific loaders:

- Software:
  - Python docs text archive (`python-3.14-docs-text.zip`)
  - `m-ric/huggingface_doc`
  - target count: `TARGET_DOCS = 50`
- Climate:
  - Wikipedia API article pulls (`CLIMATE_ARTICLES`)
  - target count: `TARGET_DOCS = 50`
- Medical:
  - `MedRAG/textbooks` aggregated passages
  - Wikipedia medical supplement list
  - target count: `MEDICAL_TARGET_DOCS = 90` in stage logic

Filtering constraints:

- `MIN_WORDS = 300`
- `MAX_WORDS = 5000`
- minimum alphabetic ratio gate

## Stage 2 - Chunk Documents (`s2_chunk.py`)

Implemented behavior:

- Tokenizer: `tiktoken` `cl100k_base`
- `CHUNK_SIZE = 512`
- `CHUNK_OVERLAP = 50`
- `MIN_CHUNK_TOKENS = 100`
- Caches to `corpora/<domain>/chunks.json`

Chunk schema includes:

- `chunk_id`, `text`, `n_tokens`, `source_doc`, `domain`, `token_start`, `token_end`

## Stage 3 - Generate Queries (`s3_queries.py`)

Implemented behavior:

- Uses OpenAI `gpt-4o-mini`
- Selects seed chunks (`SEED_CHUNKS_PER_DOMAIN = 25`)
- Generates `direct` and `partial` questions
- Filters generated query quality using CE model:
  - `cross-encoder/ms-marco-MiniLM-L-6-v2`
  - threshold `CE_FILTER_THRESHOLD = 0.50`
- Caches to `corpora/<domain>/queries.json`

Generated schema includes:

- `query_id`, `text`, `type`, `seed_chunk_id`, `is_multi_hop`, `domain`, `difficulty`

## Stage 4 - Multi-Hop Queries (`s4_multihop.py`)

Implemented behavior:

- No-op for non-medical domains
- For medical domain:
  - embeds chunks with `NeuML/pubmedbert-base-embeddings`
  - finds cross-document candidate pairs in similarity window:
    - `SIM_LOW = 0.85`
    - `SIM_HIGH = 0.97`
  - uses mechanism-term and chunk-quality filters
  - prompts OpenAI `gpt-4o-mini` for bridge questions
  - validates both chunks with CE model (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
  - appends to `queries.json`

Targets:

- min multi-hop count `TARGET_MIN = 5`
- max multi-hop count `TARGET_MAX = 8`

## Stage 5 - Embed and Build Similarity Matrices (`s5_embed.py`)

Implemented behavior:

- Encodes all chunks and queries with four model slots
- Computes cosine similarity matrix `(n_queries, n_chunks)`
- Writes one `.npy` per slot

Configured model map:

- `general`: `sentence-transformers/all-MiniLM-L6-v2`
- `medical`: `NeuML/pubmedbert-base-embeddings`
- `legal`: `nlpaueb/legal-bert-base-uncased`
- `code`: `sentence-transformers/multi-qa-mpnet-base-dot-v1`

Notes:

- Stage skips recompute unless files are missing or `force_reload=True`

## Stage 6 - Label Ground Truth (`s6_grade.py`)

Implemented behavior:

- Default CE model: `BAAI/bge-reranker-v2-m3`
- Scores every `(query, chunk)` pair
- Relevance threshold: `RELEVANCE_THRESHOLD = 0.70`
- Always injects seed chunk(s) into final relevant set
- Writes `ground_truth.json`
- Prints calibration stats (mean/min/max R* sizes)

CLI supports:

- `--domain <software|climate|medical|all>`
- `--force`
- `--model <override_model>`

## Verification (`verify.py`)

`verify_corpus(out_dir, domain)` runs after Stage 6 in `build_corpus.py`.

Hard checks:

- clean pipeline coverage threshold:
  - software >= 0.75
  - climate >= 0.65
  - medical >= 0.65

Soft diagnostics:

- per-fault degradation target drop >= 0.15
- spot-check prints sample query to R* chunk previews
- matrix/shape/stats summary

Special handling:

- degradation tests use lower test `top_k` for `top_k_too_small` and `duplicate_flooding` to reflect runtime episode behavior

## Recommended Rebuild Workflow

```bash
# Full rebuild
python -m corpora.build_corpus --domain all

# Single-domain rebuild
python -m corpora.build_corpus --domain medical

# Optional explicit Stage 6 rerun with model override
python -m corpora.stages.s6_grade --domain medical --force --model BAAI/bge-reranker-v2-m3

# Optional explicit verify pass
python -m corpora.stages.verify --domain all
```

## Operational Note

If corpus files are absent or incomplete, `server/corpus.py` will use synthetic fallback data. That fallback is useful for smoke tests but not for meaningful training or benchmark reporting.
