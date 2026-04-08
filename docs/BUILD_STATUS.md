# Build Status

Last synchronized: 2026-04-08

## Current Snapshot

The project is runnable and OpenEnv-compatible with an implemented simulation environment, generated corpus artifacts, and a baseline evaluation agent.

## Core Environment

- Packaging and manifest are present: `pyproject.toml`, `openenv.yaml`, `Dockerfile`
- OpenEnv app wiring is implemented in `server/app.py` via `create_app(...)`
- Environment logic is implemented in `server/rag_debug_env_environment.py`
- Action/observation/internal models are implemented in `models.py`
- Client is implemented in `client.py` (`RAGDebugEnv` over `openenv.core.EnvClient`)

## Corpus Pipeline

All six stages are implemented and wired by `corpora/build_corpus.py`:

1. `s1_load.py` document loading
2. `s2_chunk.py` token chunking
3. `s3_queries.py` query generation + CE filtering
4. `s4_multihop.py` medical multi-hop construction
5. `s5_embed.py` `S_true_*.npy` matrix generation
6. `s6_grade.py` cross-encoder R* labeling

Verification logic is implemented in `corpora/stages/verify.py` and called at the end of `build_corpus.py`.

## Corpus Artifacts Present

Current artifacts on disk under `corpora/`:

- `software`: 50 docs, 271 chunks, 48 queries, 0 multi-hop
- `climate`: 50 docs, 612 chunks, 44 queries, 0 multi-hop
- `medical`: 18 docs, 359 chunks, 44 queries, 6 multi-hop

For each domain, all four matrices are present:

- `S_true_general.npy`
- `S_true_medical.npy`
- `S_true_legal.npy`
- `S_true_code.npy`

## Baselines

- `outputs/eval_agent.py`: implemented zero-shot evaluator using OpenAI structured outputs
- `outputs/train_grpo.py`: scaffold/stub only (training loop TODOs remain)

## Testing Status

- No `tests/` package is currently present in the repository snapshot.
- Validation is currently operationally centered on:
  - corpus verification (`corpora/stages/verify.py`)
  - baseline agent rollouts (`outputs/eval_agent.py`)
  - OpenEnv validation command (`openenv validate`)

## Deployment and Runtime

- Local server run: `uvicorn server.app:app --host 0.0.0.0 --port 7860`
- Script entry point exists: `server = "server.app:main"` in `pyproject.toml`
- Docker build is configured with `uv sync` lock-aware install flow

## Known Gaps

- GRPO training is not fully implemented yet (`outputs/train_grpo.py`)
- Baseline inference is implemented in `inference.py` and emits structured `[START]/[STEP]/[END]` logs.
- Some documentation outside `docs/` may still lag recent tuning changes and should be reconciled periodically.
