# Hackathon Judgement: RAGDebugEnv

**Project:** RAGDebugEnv - An OpenEnv environment for training and evaluating AI agents that diagnose and fix broken Retrieval-Augmented Generation (RAG) pipelines.

**Judge:** Claude Opus 4.6 (Automated Review)
**Date:** 2026-04-08

---

## Table of Contents

1. [Validation Stage (Pass/Fail)](#1-validation-stage-passfail)
2. [Evaluation Scoring (out of 100)](#2-evaluation-scoring)
3. [Recommended Improvements](#3-recommended-improvements)

---

## 1. Validation Stage (Pass/Fail)

### Functional Requirements

| # | Requirement | Status | Evidence |
|---|------------|--------|----------|
| 1 | **Real-world task simulation** | PASS | RAG pipeline debugging is a genuine industry task. Production RAG systems break silently and root-cause diagnosis is an expert-level skill. This is not a game or toy problem. |
| 2 | **OpenEnv spec compliance** | PASS | See detailed breakdown below. |
| 3 | **Minimum 3 tasks with agent graders** | PASS | 3 tasks (Easy/Medium/Hard) with programmatic graders producing scores in [0.0, 1.0]. |
| 4 | **Meaningful reward function** | PASS | Dense, multi-component reward over full trajectory. Rewards partial progress, penalizes undesirable behavior. All rewards bounded in [0.0, 1.0]. |
| 5 | **Baseline inference script** | PASS | `inference.py` in root directory, uses OpenAI client, reads env vars, produces baseline scores. |

#### OpenEnv Spec Compliance Detail

| Element | Status | Location |
|---------|--------|----------|
| Typed `Action` (Pydantic, inherits framework base) | PASS | `models.py:124` - `RAGDebugAction(Action)` |
| Typed `Observation` (Pydantic, inherits framework base) | PASS | `models.py:184` - `RAGDebugObservation(Observation)` |
| Typed `Reward` model | PASS | `models.py:354` - `Reward(BaseModel)` with value + components |
| `step(action)` returns observation, reward, done | PASS | `server/rag_debug_env_environment.py:248-285` |
| `reset()` returns initial observation | PASS | `server/rag_debug_env_environment.py:126-246` |
| `state` property returns current state | PASS | `server/rag_debug_env_environment.py:287-289` |
| `openenv.yaml` with metadata | PASS | Root `openenv.yaml` with spec_version, name, type, runtime, app, port |
| Inherits from `Environment` base class | PASS | `server/rag_debug_env_environment.py:63` - `RAGDebugEnvironment(Environment)` |

#### Task & Grader Detail

| Task | Domain | Difficulty | Faults | Grader Score Range | Success Criteria |
|------|--------|-----------|--------|--------------------|------------------|
| Task 1 | Software (Python docs) | Easy | 1-2 config faults (randomly sampled from 4 sets) | [0.0, 1.0] | `task_score >= 0.75` |
| Task 2 | Climate (IPCC reports) | Medium | Compound config faults (randomly sampled from 4 sets) | [0.0, 1.0] | `task_score >= 0.75` |
| Task 3 | Medical (MedRAG textbooks) | Hard | Wrong embedding model + chunk_too_large + threshold_too_high (fixed set) | [0.0, 1.0] | `task_score >= 0.70` AND `multi_hop_coverage > 0.60` |

Grader formulas:
- Tasks 1 & 2: `0.60 * coverage + 0.25 * precision + 0.15 * efficiency`
- Task 3: `0.55 * coverage + 0.25 * precision + 0.20 * multi_hop_coverage`

All graders are deterministic, programmatic, and produce scores strictly in [0.0, 1.0]. Difficulty progression is clear: Task 1 has simple single/double faults on clean prose, Task 2 introduces compound interacting faults on cross-disciplinary text, Task 3 requires diagnosing a fundamental model mismatch plus config faults on heavily specialized medical text with multi-hop query constraints.

### Non-Functional Requirements

| # | Requirement | Status | Evidence |
|---|------------|--------|----------|
| 1 | **HF Space deployment** | PASS | README frontmatter configured (`sdk: docker`, `app_port: 7860`). Dockerfile targets port 7860. `openenv.yaml` specifies `port: 7860`. |
| 2 | **Containerized execution** | PASS | Multi-stage `Dockerfile` using `openenv-base` image. Installs deps via `uv sync`, exposes port 7860, health check included. |
| 3 | **Documentation** | PASS | README contains: environment description/motivation, action/observation space tables, task descriptions with difficulty, setup instructions (local + Docker), baseline scores table. Additionally: `docs/ARCHITECTURE.md` with detailed internals. |

### Pre-Submission Checklist

| # | Requirement | Status | Evidence |
|---|------------|--------|----------|
| 1 | **Baseline reproduces** | PASS | `inference.py` has proper error handling, connects to server, runs episodes, and produces [START]/[STEP]/[END] output. Graceful fallback to heuristic agent on LLM failure. |
| 2 | **3+ tasks with graders** | PASS | 3 tasks with `_compute_task_score()` and `_check_success()` producing scores in [0.0, 1.0]. |
| 3a | **API_BASE_URL defined** | PASS | `inference.py:53` - `os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"` |
| 3b | **MODEL_NAME defined** | PASS | `inference.py:54` - `os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"` |
| 3c | **HF_TOKEN defined** | PASS | `inference.py:52` - `os.getenv("HF_TOKEN") or os.getenv("API_KEY")` |
| 3d | **Script named `inference.py` in root** | PASS | Located at project root: `/inference.py` |
| 3e | **Uses OpenAI Client** | PASS | `inference.py:46` imports `from openai import OpenAI`, creates client at line 669. |
| 3f | **Stdout log format compliance** | PASS | See detailed analysis below. |
| 4 | **Infra restrictions** (runtime < 20min, vcpu=2, 8GB RAM) | PASS | Environment uses precomputed similarity matrices (~1ms/step). No heavy computation during inference. 10 steps max per task. |

#### Stdout Format Compliance Analysis

The project's logging functions (`log_start`, `log_step`, `log_end` at lines 318-337) are compared against the mandatory template:

**[START] line:**
- Template: `[START] task={task} env={env} model={model}`
- Project: `[START] task={_one_line(task)} env={_one_line(env)} model={_one_line(model)}`
- Field names: `task`, `env`, `model` - MATCH
- Ordering: MATCH
- Extra: `_one_line()` strips internal newlines for safety - good practice, no format violation.

**[STEP] line:**
- Template: `[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}`
- Project: `[STEP] step={step} action={action_val} reward={reward:.2f} done={done_val} error={error_val}`
- Field names: `step`, `action`, `reward`, `done`, `error` - MATCH
- Ordering: MATCH
- Reward format: `.2f` (2 decimal places) - MATCH
- `done`: lowercase boolean via `str(done).lower()` - MATCH
- `error`: string or `"null"` - MATCH

**[END] line:**
- Template: `[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}`
- Project: `[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}`
- Field names: `success`, `steps`, `score`, `rewards` - MATCH
- Ordering: MATCH
- `success`: lowercase boolean - MATCH
- `score`: `.3f` format - matches template code exactly
- `rewards`: comma-separated `.2f` values - MATCH

**Critical design decision:** All rich visual output (banners, bars, metrics display) is sent to `stderr` via the `_rich()` function (`print(text, file=sys.stderr)`), ensuring `stdout` contains ONLY the `[START]`/`[STEP]`/`[END]` lines. This is excellent compliance engineering.

### Verdict: PASS

**The project passes all functional requirements, non-functional requirements, and pre-submission checklist items.** All mandatory elements are present and correctly implemented. The stdout logging format matches the template specification exactly in field names, ordering, and formatting.

---

## 2. Evaluation Scoring

### Overall Score: 87 / 100

---

### 2.1 Real-World Utility — 27 / 30 (Weight: 30%)

**What it models:** The environment simulates the diagnosis and repair of broken RAG (Retrieval-Augmented Generation) pipelines -- one of the most common failure modes in production AI applications.

**Why this is genuinely useful:**

- **Real industry pain point.** RAG pipelines degrade silently. Retrieval quality drops without obvious errors, and root-cause diagnosis requires understanding the interplay between embedding models, chunking strategies, similarity thresholds, and reranking. This is a task engineers spend real hours on.
- **Transferable skills.** An agent trained on this environment would learn systematic debugging strategies (diagnose via metrics, form hypotheses, test fixes, verify) that map directly to real RAG debugging workflows.
- **No existing alternatives.** There are no other training environments targeting this specific skill for AI agents. This fills a genuine gap.
- **Practical architecture.** The precomputed similarity matrix approach means the environment runs at ~1ms/step, making it tractable for RL training while maintaining mathematical fidelity to real retrieval dynamics.

**Minor deductions (-3):**
- The environment simulates retrieval via matrices rather than running actual vector databases. While this is a sound engineering decision for speed, it means some failure modes (e.g., index corruption, embedding service latency, version mismatches) are not covered.
- The action space, while comprehensive (9 actions), doesn't cover some real-world debugging actions like inspecting specific chunks, running A/B retrievals, or examining embedding distributions directly.

---

### 2.2 Task & Grader Quality — 22 / 25 (Weight: 25%)

**Task design:**

| Aspect | Assessment |
|--------|-----------|
| Clear objectives | Each task has an explicit textual description and quantitative success criteria. |
| Difficulty progression | Easy (1-2 faults, clean text) -> Medium (compound faults, cross-disciplinary) -> Hard (model mismatch + faults + multi-hop). Well-calibrated. |
| Domain variety | Software docs, climate reports, medical textbooks -- three distinct domains with genuinely different retrieval characteristics. |
| Fault variety | 9 fault types covering the full spectrum of RAG failure modes: chunking, thresholding, model selection, reranking, context limits, duplicates. |
| Baseline scores provided | Yes, with a clear table showing Random / Heuristic / Zero-Shot LLM / Target RL scores. |

**Grader quality:**

- Graders are fully programmatic and deterministic given the same seed.
- Task scores use weighted combinations of coverage, precision, and efficiency (or multi-hop coverage for Task 3), which reflects real retrieval quality.
- Success thresholds are differentiated per task (0.75, 0.75, 0.70+multi-hop), adding appropriate challenge.
- The inclusion of multi-hop queries in Task 3 as a separate grading dimension is a thoughtful design choice that tests a deeper capability.

**Minor deductions (-3):**
- Fault sampling is stochastic within tasks (Task 1 samples from 4 fault sets, Task 2 from 4 fault sets). While this adds variety, it means two runs of "Task 1" can have meaningfully different difficulty. For evaluation consistency, fixed fault sets per task might be slightly preferable.
- The initial config randomization (`top_k` from 5-8, `threshold` from 0.34-0.48) plus the `_calibrate_initial_difficulty()` nudge adds variability that could affect score reproducibility across runs.
- Task 2 could use a sharper differentiation from Task 1 beyond just different fault sets -- perhaps more queries, a tighter step budget, or additional constraints.

---

### 2.3 Environment Design — 18 / 20 (Weight: 20%)

**State management:**
- Clean separation between internal state (`InternalState` with faults, seeds, action/reward history) and observable state (`RAGDebugObservation`). The agent never sees injected faults -- it must infer them from metrics, which IS the task.
- Deterministic noise arrays pre-generated at `reset()` ensure reproducibility within an episode.
- Proper episode lifecycle: `reset()` fully reinitializes, `step()` increments cleanly, auto-terminate at `max_steps`.

**Action/Observation spaces:**
- 9 well-typed actions with parameter validation via Pydantic models.
- Rich observation space includes: pipeline config, per-query results (with scores, coverage, precision), aggregate metrics, corpus stats, diagnostic hints, and reward component breakdown.
- The observation intentionally withholds fault information, creating a proper partial-observability debugging challenge.

**Reward shaping (outstanding):**
This is one of the project's strongest aspects. The reward function is multi-component and well-justified:

| Component | Range | Purpose |
|-----------|-------|---------|
| `progress_reward` | [0.10, 0.65] | Absolute quality level -- ensures full reward range used across episode |
| `delta_bonus` | [-0.15, +0.15] | Direction signal -- distinguishes improving from static |
| `empty_retrieval_signal` | [-0.06, +0.06] | Bidirectional feedback on empty retrievals |
| `overflow_signal` | [-0.04, +0.04] | Bidirectional feedback on context overflows |
| `step_cost` | -0.01 | Efficiency pressure |
| `redundancy_penalty` | -0.04 | Discourages repetitive actions |
| `invalid_action_penalty` | -0.05 | Feedback on invalid parameters |
| Terminal success | [0.70, 1.00] | Clear success zone |
| Terminal failure | [0.00, 0.20] | Clear failure zone with partial credit |

The separation of non-terminal rewards ([0.0, ~0.89]) from terminal rewards (success: [0.7, 1.0], failure: [0.0, 0.2]) is clean and prevents reward signal confusion. The `progress_reward` acting as a "base" that maps absolute quality level to reward range is a sophisticated design that avoids the common RL problem of reward sparsity.

**Episode boundaries:**
- Max 10 steps per episode (appropriate for the task complexity).
- `submit` action explicitly ends the episode.
- Auto-terminate at max steps if agent doesn't submit.

**Minor deductions (-2):**
- The `ADJUST_CHUNK_OVERLAP` action triggers no matrix recomputation (line 336: "Overlap has no direct matrix effect; no recompute needed"), yet the fault math for `CHUNK_TOO_SMALL` does use `config_chunk_overlap` to reduce noise sigma. This means the overlap effect is only visible after a DIFFERENT action triggers recomputation. This is a subtle bug/design inconsistency.
- The `rewrite_query` action's boost of +0.20 to ground-truth columns is somewhat artificial and doesn't vary by query difficulty or strategy type (the `strategy` parameter is accepted but not used differententially).

---

### 2.4 Code Quality & Spec Compliance — 12 / 15 (Weight: 15%)

**Strengths:**
- Clean project structure with logical separation: `models.py` (types), `server/` (environment logic), `client.py` (client), `inference.py` (competition script), `corpora/` (data pipeline).
- Fully typed Pydantic models with field validators, docstrings, and clear type hierarchies.
- Proper OpenEnv spec compliance: inherits from framework base classes (`Action`, `Observation`, `Environment`), uses `create_app`, has `openenv.yaml`.
- Multi-stage Dockerfile with health check, proper `ENV` setup, and HF Spaces compatibility.
- Comprehensive docstrings throughout (especially in `models.py` and `server/rag_debug_env_environment.py`).
- Corpus build pipeline is well-structured (6 stages with verification).
- `.gitignore` and `.dockerignore` properly configured.
- Rich stderr output in inference.py is well-separated from stdout compliance logs.

**Deductions (-3):**
- **No test suite.** `pytest` is listed in optional dependencies (`pyproject.toml:26`) and `.pytest_cache/` exists, but no test files were found anywhere in the project. For a project of this complexity (fault math, reward computation, metric calculation, action routing), unit tests would significantly increase confidence in correctness.
- **`train_grpo.py` is a stub.** The training script raises `NotImplementedError` in `collect_rollout()` and has all critical code commented out. While not strictly required by the hackathon, its presence as a stub signals incomplete work.
- **`.env` file in repository.** The `.env` file is in `.gitignore` but appears in `git status` as modified (`M README.md`... actually the status shows `.env` is not staged but present). This is a minor concern -- secrets should never be in the repo directory at all for a deployable project.

---

### 2.5 Creativity & Novelty — 9 / 10 (Weight: 10%)

**Novel elements:**

1. **RAG debugging as RL environment.** This is an original problem domain for RL -- no prior work targets this specific skill.
2. **Precomputed similarity matrices.** Instead of running actual vector databases, the environment uses `S_true` matrices and applies fault transformations mathematically. This is a clever engineering decision that enables ~1ms steps while maintaining mathematical rigor.
3. **Fault injection as mathematical transformations.** Each fault type corresponds to a specific matrix transformation (box filter for chunk smearing, Gaussian noise for unstable embeddings, score deflation for high threshold, etc.). These are grounded in the actual signal-processing effects that RAG faults produce.
4. **Cross-encoder reranking simulation.** Simulated by blending faulted scores back toward pre-fault scores (`alpha=0.35`). This is non-trivial -- it correctly models how a cross-encoder partially recovers rank order for noise-based faults and restores score spread for compression faults.
5. **Multi-embedding-model simulation.** Four different precomputed S_true matrices per domain allow the agent to swap models without running actual inference. The wrong-model fault is implicit in the score distribution differences between matrices.
6. **Multi-hop queries.** Task 3 includes queries requiring multiple chunks, with separate grading for multi-hop coverage. This adds a meaningful dimension beyond single-chunk retrieval.
7. **Diagnostic hints system.** The environment generates context-aware hints based on metric patterns, aiding agent interpretability without giving away fault identities.
8. **Reward component decomposition.** Every reward value comes with a named breakdown, making reward shaping decisions auditable and debuggable.

**Minor deduction (-1):**
- The core concept of "tune config parameters to fix a system" has been explored in other domains (network tuning, database config optimization). The novelty here is specifically in applying it to RAG pipelines with realistic fault mechanics.

---

### Score Summary

| Criterion | Weight | Score | Weighted |
|-----------|--------|-------|----------|
| Real-World Utility | 30% | 27/30 | 27.0 |
| Task & Grader Quality | 25% | 22/25 | 22.0 |
| Environment Design | 20% | 18/20 | 18.0 |
| Code Quality & Spec Compliance | 15% | 12/15 | 12.0 |
| Creativity & Novelty | 10% | 9/10 | 9.0 |
| **TOTAL** | **100%** | | **88.0 / 100** |

---

## 3. Recommended Improvements

The project is mature and well-executed. The following are improvements that could push the score higher:

### Essential Fixes

1. **`ADJUST_CHUNK_OVERLAP` should trigger `_recompute_S_faulted()`.**
   In `server/rag_debug_env_environment.py:330-336`, the chunk overlap action does NOT call `_recompute_S_faulted()`, despite `apply_faults()` in `fault_math.py:88-90` using `config_chunk_overlap` to modulate `CHUNK_TOO_SMALL` noise severity. This means the overlap parameter's effect is only visible after a *different* action triggers recomputation. This is a functional bug that could confuse agents and produce incorrect reward signals.

   **Fix:** Add `self._recompute_S_faulted()` after the config update in the `ADJUST_CHUNK_OVERLAP` handler, similar to how `ADJUST_CHUNK_SIZE` does it.

2. **Add a test suite.**
   The project has no tests despite `pytest` being listed as a dev dependency. At minimum, add tests for:
   - Reward computation bounds (verify all rewards stay in [0.0, 1.0])
   - Fault math correctness (each fault type degrades metrics in the expected direction)
   - Task score formula correctness
   - Action routing (each action type modifies the expected config field)
   - Episode lifecycle (reset clears state, step increments, auto-terminate works)
   - Stdout format compliance (parse output and verify field names/ordering)

3. **Complete or remove `train_grpo.py`.**
   The stub with `NotImplementedError` and commented-out code sends the wrong signal. Either implement a minimal working training loop or remove it entirely and reference it as "future work" in the README.
