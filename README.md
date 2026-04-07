---
title: RAGDebugEnv
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# RAGDebugEnv

RAGDebugEnv is an OpenEnv-compatible environment for training and evaluating agents that debug broken retrieval pipelines.

The environment simulates retrieval behavior with precomputed similarity matrices so each step is fast, while still exposing realistic debugging actions such as threshold tuning, top-k tuning, embedding model swaps, reranking toggles, and query rewrites.

## Current Status

This repository currently includes:

- A working OpenEnv server app in server/app.py
- A working environment implementation in server/rag_debug_env_environment.py
- Typed action and observation contracts in models.py
- A working client in client.py
- A full corpus build pipeline across stages s1 to s6 plus verification
- A working baseline evaluator in baseline/eval_agent.py
- A GRPO training scaffold in baseline/train_grpo.py (not fully implemented yet)

## Tasks

The environment defines three tasks:

- Task 1: software domain, sampled config faults
- Task 2: climate domain, sampled compound faults
- Task 3: medical domain, fixed fault set including wrong embedding model and multi-hop emphasis

Runtime constants currently used by the environment:

- Episode queries per task: 5
- Max steps per episode: 10
- Task success is based on task score thresholds in environment logic, not raw coverage alone

## Architecture (Short)

Episode-time simulation flow:

1. Load corpus artifacts for one domain
2. Sample episode queries and injected faults
3. Build faulted similarity matrix from S_true matrices and fault math
4. Simulate top-k plus threshold retrieval
5. Compute coverage and precision against ground truth R*
6. Return dense reward for iterative debugging

For full details, see docs/ARCHITECTURE.md.

## Repository Layout

- baseline/: eval and training scripts
- corpora/: corpus artifacts, builder, and stages
- docs/: architecture and reference docs
- server/: app entrypoint, environment, constants, corpus loader, fault math
- client.py: OpenEnv client implementation
- models.py: Action, Observation, and internal models
- openenv.yaml: OpenEnv manifest
- pyproject.toml: package metadata and base dependencies

## Setup

Recommended setup uses uv lock resolution:

```bash
uv sync
```

Alternative setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Note: corpus building and baseline evaluation require additional ML/data dependencies (for example sentence-transformers, datasets, scikit-learn, tiktoken, torch, openai). If you do not use uv sync, install those manually.

## Required Environment Variables

For corpus build and baseline eval:

- OPENAI_API_KEY

Optional runtime variables used by parts of the stack:

- HF_TOKEN
- API_BASE_URL
- MODEL_NAME

## Build Corpus

Build one domain:

```bash
python -m corpora.build_corpus --domain software
python -m corpora.build_corpus --domain climate
python -m corpora.build_corpus --domain medical
```

Build all domains:

```bash
python -m corpora.build_corpus --domain all
```

Outputs are written under corpora/<domain>/ including:

- docs.json
- chunks.json
- queries.json
- ground_truth.json
- S_true_general.npy
- S_true_medical.npy
- S_true_legal.npy
- S_true_code.npy
- corpus_stats.json

For stage details, see docs/CORPUS_BUILD_PLAN.md.

## Run Server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Run Baseline Evaluator

```bash
python baseline/eval_agent.py --task 1 --episodes 3
python baseline/eval_agent.py --task all --episodes 2
```

## Validate OpenEnv Wiring

```bash
openenv validate
```

## Known Gaps

- baseline/train_grpo.py is a scaffold with TODO sections
- inference.py is currently a template for a different environment and is not wired to RAGDebugEnv
- If corpus artifacts are missing, server/corpus.py falls back to synthetic data for smoke tests

## Additional Documentation

- docs/ARCHITECTURE.md
- docs/BUILD_STATUS.md
- docs/CLAUDE.md
- docs/CORPUS_BUILD_PLAN.md
- docs/MODELS_REFERENCE.md
