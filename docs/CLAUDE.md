# RAGDebugEnv Reference Context

This file is a fast orientation guide for contributors and coding agents working in this repository.

## What This Repository Implements

A simulated OpenEnv environment for debugging RAG retrieval pipelines by RL-style interaction.

- Server app: `server/app.py`
- Environment logic: `server/rag_debug_env_environment.py`
- Action/observation models: `models.py`
- Client: `client.py`
- Inference script: `inference.py` (competition-ready, uses OpenAI client)
- Corpus build pipeline: `corpora/build_corpus.py` and `corpora/stages/*`

## Project Layout (Current)

```text
rag_debug_env/
  corpora/
    build_corpus.py
    software/
    climate/
    medical/
    stages/
      s1_load.py
      s2_chunk.py
      s3_queries.py
      s4_multihop.py
      s5_embed.py
      s6_grade.py
      verify.py
      playground.py
  docs/
    ARCHITECTURE.md
    BUILD_STATUS.md
    CLAUDE.md
    CORPUS_BUILD_PLAN.md
    MODELS_REFERENCE.md
  outputs/
    eval_agent.py
    train_grpo.py
  server/
    app.py
    constants.py
    corpus.py
    fault_math.py
    rag_debug_env_environment.py
  client.py
  inference.py
  models.py
  pyproject.toml
  openenv.yaml
  Dockerfile
```

## Runtime Facts To Keep In Mind

- All tasks currently use `max_steps=10`
- `PipelineConfig.similarity_threshold` default is `0.3` (not `0.7`)
- Task 3 starts with `embedding_model=legal` intentionally
- Task success is based on task score thresholds in `_check_success`, not raw coverage alone
- Synthetic corpus fallback exists in `server/corpus.py` for missing artifacts
- HF Spaces port is `7860` (set in README frontmatter, Dockerfile, and openenv.yaml)

## Corpus Build Facts

`corpora/build_corpus.py` runs all six stages and calls `verify_corpus`.

Outputs per domain:

- `docs.json`
- `chunks.json`
- `queries.json`
- `ground_truth.json`
- `S_true_general.npy`
- `S_true_medical.npy`
- `S_true_legal.npy`
- `S_true_code.npy`
- `corpus_stats.json`

## Scripts

- `outputs/eval_agent.py` — GPT-4o-mini zero-shot eval agent (actively usable).
- `outputs/train_grpo.py` — GRPO training scaffold (stub with TODOs).
- `inference.py` — Competition inference script with [START]/[STEP]/[END] logging.

## Commands

```bash
# Build corpus for one domain
python -m corpora.build_corpus --domain software

# Build all domains
python -m corpora.build_corpus --domain all

# Run server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run baseline evaluator
python outputs/eval_agent.py --task 1 --episodes 3

# Run inference script
python inference.py

# Validate OpenEnv integration
openenv validate
```
