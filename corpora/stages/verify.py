"""
corpora/stages/verify.py
------------------------
Corpus sanity checks. Called by build_corpus.py after all 6 stages complete.

Four checks:
  1. Clean-pipeline coverage  — default PipelineConfig, no faults, must hit threshold
  2. Fault degradation        — each fault individually must drop coverage >= 0.15
  3. Spot check               — 5 sample (query, R* chunks) pairs printed for human review
  4. Stats summary            — shapes, R* sizes, score distributions

Usage:
    # Via build_corpus.py (automatic)
    # Or standalone:
    python -m corpora.stages.verify --domain software
    python -m corpora.stages.verify --domain all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from models import FaultType
from server.fault_math import apply_faults, make_noise

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum clean coverage per domain (no faults, default config)
_CLEAN_COV_THRESHOLD: dict[str, float] = {
    "software": 0.75,
    "climate":  0.65,
    "medical":  0.65,
}

# Every active fault must drop coverage by at least this much
_MIN_DEGRADATION = 0.15

# Default PipelineConfig values (must match models.py defaults)
_DEFAULT_CHUNK_SIZE    = 512
_DEFAULT_CONTEXT_LIMIT = 4096
_DEFAULT_THRESHOLD     = 0.3
_DEFAULT_TOP_K         = 10

# Fault-specific top_k overrides used in Check 2.
# These mirror the initial-config adjustments in RAGDebugEnvironment.reset():
#   TOP_K_TOO_SMALL   → episode starts at top_k=2-4 (mid-range test value: 3)
#   DUPLICATE_FLOODING → episode starts at top_k=4-7 (mid-range test value: 5)
# Testing at default top_k=10 understates degradation for both faults because:
#   - TOP_K_TOO_SMALL score compression preserves rank order at top_k=10
#   - DUPLICATE_FLOODING can't crowd out high-scoring relevant chunks at top_k=10
_FAULT_TEST_TOP_K: dict = {
    FaultType.TOP_K_TOO_SMALL:    3,
    FaultType.DUPLICATE_FLOODING: 5,
}

# Faults to test in the degradation check (all matrix-transformable ones)
_MATRIX_FAULTS = [
    FaultType.CHUNK_TOO_LARGE,
    FaultType.CHUNK_TOO_SMALL,
    FaultType.THRESHOLD_TOO_LOW,
    FaultType.THRESHOLD_TOO_HIGH,
    FaultType.TOP_K_TOO_SMALL,
    FaultType.DUPLICATE_FLOODING,
    FaultType.CONTEXT_OVERFLOW,
    FaultType.NO_RERANKING,
]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def verify_corpus(out_dir: Path, domain: str) -> bool:
    """
    Run all sanity checks on a built corpus.

    Parameters
    ----------
    out_dir : Path
        Directory containing chunks.json, queries.json, ground_truth.json,
        S_true_general.npy, etc.
    domain : str
        "software" | "climate" | "medical"

    Returns
    -------
    True if all checks pass.

    Raises
    ------
    AssertionError if a hard check fails (clean coverage or missing files).
    Prints warnings (but does not raise) for soft failures (fault degradation).
    """
    print(f"\n{'='*60}")
    print(f"  verify_corpus: {domain.upper()}")
    print(f"{'='*60}")

    # Load artifacts
    chunks, queries, ground_truth, s_true = _load_artifacts(out_dir)
    n_q = len(queries)
    n_c = len(chunks)
    chunk_ids  = [c["chunk_id"] for c in chunks]
    query_ids  = [q["query_id"] for q in queries]
    chunk_by_id = {c["chunk_id"]: c for c in chunks}

    # -----------------------------------------------------------------------
    # Check 4 (stats) — always runs first so info is visible even on failure
    # -----------------------------------------------------------------------
    _print_stats(domain, chunks, queries, ground_truth, s_true)

    # -----------------------------------------------------------------------
    # Check 1 — Clean pipeline coverage
    # -----------------------------------------------------------------------
    S_gen = s_true["general"]
    clean_cov = _simulate_coverage(S_gen, ground_truth, query_ids, chunk_ids)
    threshold = _CLEAN_COV_THRESHOLD[domain]

    print(f"\n[Check 1] Clean coverage (no faults, default config)")
    print(f"  coverage = {clean_cov:.3f}  (threshold: >= {threshold})")

    if clean_cov < threshold:
        raise AssertionError(
            f"[verify] FAIL: clean coverage {clean_cov:.3f} < {threshold} for {domain}.\n"
            f"  This usually means the cross-encoder threshold in s6_grade.py was too strict\n"
            f"  (R* too small), OR the general embedding model is severely misaligned.\n"
            f"  Try lowering RELEVANCE_THRESHOLD in s6_grade.py and re-running Stage 6."
        )
    print(f"  PASS")

    # -----------------------------------------------------------------------
    # Check 2 — Fault degradation
    # -----------------------------------------------------------------------
    print(f"\n[Check 2] Fault degradation (each fault must drop coverage >= {_MIN_DEGRADATION})")

    rng = np.random.default_rng(42)
    shape = (n_q, n_c)
    noise = make_noise(rng, shape)
    dupe_ids = rng.choice(n_c, size=max(1, n_c // 7), replace=False)

    all_degraded = True
    rows = []
    for fault in _MATRIX_FAULTS:
        S_faulted = apply_faults(
            S=S_gen,
            fault_types={fault},
            config_chunk_size=_DEFAULT_CHUNK_SIZE,
            config_context_limit=_DEFAULT_CONTEXT_LIMIT,
            config_use_reranking=False,
            noise=noise,
            dupe_ids=dupe_ids,
        )
        test_top_k = _FAULT_TEST_TOP_K.get(fault, _DEFAULT_TOP_K)
        faulted_cov = _simulate_coverage(
            S_faulted, ground_truth, query_ids, chunk_ids, top_k=test_top_k
        )
        drop = clean_cov - faulted_cov
        passed = drop >= _MIN_DEGRADATION
        if not passed:
            all_degraded = False
        status = "PASS" if passed else "WARN"
        label = fault.value
        if test_top_k != _DEFAULT_TOP_K:
            label += f" (top_k={test_top_k})"
        rows.append((label, faulted_cov, drop, status))

    # Special case: WRONG_EMBEDDING_MODEL (implicit via model swap)
    if "legal" in s_true and domain == "medical":
        S_legal = s_true["legal"]
        legal_cov = _simulate_coverage(S_legal, ground_truth, query_ids, chunk_ids)
        drop = clean_cov - legal_cov
        passed = drop >= _MIN_DEGRADATION
        if not passed:
            all_degraded = False
        status = "PASS" if passed else "WARN"
        rows.append(("wrong_embedding_model (legal vs general)", legal_cov, drop, status))

    # Print table
    col_w = max(len(r[0]) for r in rows) + 2
    print(f"  {'Fault':<{col_w}} {'Faulted cov':>12}  {'Drop':>6}  Status")
    print(f"  {'-'*col_w} {'-'*12}  {'-'*6}  ------")
    for fault_name, cov, drop, status in rows:
        print(f"  {fault_name:<{col_w}} {cov:>12.3f}  {drop:>+6.3f}  {status}")

    if not all_degraded:
        print(
            f"\n  WARNING: Some faults produce insufficient degradation. "
            f"These faults will provide a weak training signal.\n"
            f"  Consider increasing fault magnitude constants in fault_math.py."
        )
    else:
        print(f"\n  All faults degrade coverage sufficiently. PASS")

    # -----------------------------------------------------------------------
    # Check 3 — Spot check
    # -----------------------------------------------------------------------
    print(f"\n[Check 3] Spot check — 5 sample (query → R* chunks)")
    _spot_check(queries, ground_truth, chunk_by_id, rng=np.random.default_rng(0), n=5)

    print(f"\n{'='*60}")
    print(f"  verify_corpus {domain.upper()}: COMPLETE")
    print(f"{'='*60}\n")
    return True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_artifacts(out_dir: Path) -> tuple:
    """Load and return (chunks, queries, ground_truth, s_true dict)."""
    required = ["chunks.json", "queries.json", "ground_truth.json"]
    for fname in required:
        if not (out_dir / fname).exists():
            raise FileNotFoundError(f"[verify] Missing {out_dir / fname}")

    with open(out_dir / "chunks.json") as f:
        chunks = json.load(f)
    with open(out_dir / "queries.json") as f:
        queries = json.load(f)
    with open(out_dir / "ground_truth.json") as f:
        ground_truth = json.load(f)

    s_true: dict[str, np.ndarray] = {}
    for model in ["general", "medical", "legal", "code"]:
        path = out_dir / f"S_true_{model}.npy"
        if path.exists():
            s_true[model] = np.load(path, mmap_mode="r").astype(np.float32)

    if "general" not in s_true:
        raise FileNotFoundError(f"[verify] Missing S_true_general.npy in {out_dir}")

    return chunks, queries, ground_truth, s_true


def _simulate_coverage(
    S: np.ndarray,
    ground_truth: dict,
    query_ids: list[int],
    chunk_ids: list[int],
    threshold: float = _DEFAULT_THRESHOLD,
    top_k: int = _DEFAULT_TOP_K,
) -> float:
    """Simulate retrieval and return mean coverage across queries."""
    coverages = []
    n_c = len(chunk_ids)
    for i, qid in enumerate(query_ids):
        scores = S[i]
        k = min(top_k, n_c)
        top_idx = np.argsort(scores)[::-1][:k]
        retrieved = {chunk_ids[j] for j in top_idx if scores[j] >= threshold}
        r_star = set(ground_truth.get(str(qid), []))
        cov = len(retrieved & r_star) / len(r_star) if r_star else 0.0
        coverages.append(cov)
    return float(np.mean(coverages)) if coverages else 0.0


def _print_stats(
    domain: str,
    chunks: list[dict],
    queries: list[dict],
    ground_truth: dict,
    s_true: dict[str, np.ndarray],
) -> None:
    """Print summary statistics."""
    n_mh = sum(1 for q in queries if q.get("is_multi_hop"))
    r_star_sizes = [len(v) for v in ground_truth.values()]

    direct_sizes = [
        len(ground_truth[str(q["query_id"])])
        for q in queries if not q.get("is_multi_hop") and str(q["query_id"]) in ground_truth
    ]
    mh_sizes = [
        len(ground_truth[str(q["query_id"])])
        for q in queries if q.get("is_multi_hop") and str(q["query_id"]) in ground_truth
    ]

    print(f"\n[Stats] {domain}")
    print(f"  chunks      : {len(chunks)}")
    print(f"  queries     : {len(queries)} total  ({n_mh} multi-hop)")
    print(f"  R* size     : mean={np.mean(r_star_sizes):.2f}  "
          f"min={min(r_star_sizes)}  max={max(r_star_sizes)}")
    if direct_sizes:
        print(f"  R* direct   : mean={np.mean(direct_sizes):.2f}  (target 1-4)")
    if mh_sizes:
        print(f"  R* multi-hop: mean={np.mean(mh_sizes):.2f}  (target 2-5)")

    print(f"\n  S_true matrices:")
    for model, S in s_true.items():
        print(f"    {model:<10} shape={S.shape}  "
              f"min={S.min():.3f}  max={S.max():.3f}  "
              f"mean={S.mean():.3f}  std={S.std():.3f}")


def _spot_check(
    queries: list[dict],
    ground_truth: dict,
    chunk_by_id: dict[int, dict],
    rng: np.random.Generator,
    n: int = 5,
) -> None:
    """Print n random (query, R* chunk previews) for human inspection."""
    indices = rng.choice(len(queries), size=min(n, len(queries)), replace=False)
    for idx in indices:
        q = queries[int(idx)]
        qid = q["query_id"]
        r_star = ground_truth.get(str(qid), [])
        mh_tag = " [multi-hop]" if q.get("is_multi_hop") else ""
        print(f"\n  Q{qid}{mh_tag}: {q['text']}")
        if not r_star:
            print(f"    R* = (empty — check calibration!)")
        else:
            for cid in r_star[:3]:  # show up to 3 relevant chunks
                chunk = chunk_by_id.get(cid)
                if chunk:
                    preview = chunk["text"][:100].replace("\n", " ").strip()
                    print(f"    chunk {cid}: \"{preview}...\"")
            if len(r_star) > 3:
                print(f"    ... and {len(r_star) - 3} more")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify a built corpus")
    parser.add_argument(
        "--domain",
        choices=["software", "climate", "medical", "all"],
        default="all",
    )
    args = parser.parse_args()

    _corpora_dir = Path(__file__).parent.parent
    domains = ["software", "climate", "medical"] if args.domain == "all" else [args.domain]

    all_passed = True
    for domain in domains:
        out_dir = _corpora_dir / domain
        try:
            verify_corpus(out_dir, domain)
        except (AssertionError, FileNotFoundError) as e:
            print(f"\nFAIL [{domain}]: {e}", file=sys.stderr)
            all_passed = False

    sys.exit(0 if all_passed else 1)
