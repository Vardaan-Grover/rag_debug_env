"""
eval_embedding_models.py
------------------------
Evaluate embedding models for suitability in the RAGDebugEnv RL environment.

Usage:
    # Evaluate a single model on all domains
    python eval_embedding_models.py --model "sentence-transformers/all-MiniLM-L6-v2"

    # Evaluate multiple models and compare
    python eval_embedding_models.py \
        --model "BAAI/bge-small-en-v1.5" \
        --model "sentence-transformers/all-mpnet-base-v2" \
        --model "NeuML/pubmedbert-base-embeddings"

    # Evaluate only specific domains
    python eval_embedding_models.py \
        --model "BAAI/bge-small-en-v1.5" \
        --domain medical --domain software

    # Save results to JSON
    python eval_embedding_models.py --model "BAAI/bge-small-en-v1.5" --save results.json

What this script measures
--------------------------
For the RL environment to produce meaningful learning signals, each embedding
model needs to have "discriminability" — the ability to rank truly relevant
chunks (R*) above irrelevant ones.

Metrics explained:
  separation         Mean R* score minus mean non-R* score. Higher = better
                     discrimination. Below 0.20 is too weak for RL.
  rstar_mean         Average similarity score for ground-truth relevant chunks.
                     Should be in 0.55–0.85 range for threshold tuning to work.
  nonrstar_mean      Average score for all other chunks. Should be low (< 0.25)
                     so raising threshold can filter them out.
  coverage@10        % of R* chunks found in top-10 results, no threshold filter.
                     This is the best-case retrieval quality.
  coverage@0.30      % of R* chunks retrieved with top-k=10 and threshold=0.30.
                     This is what the clean pipeline achieves with default config.
  pct_above_0.30     % of all chunks with score >= 0.30. Should be < 15% for
                     threshold to be a useful lever. At 100%, threshold is useless.
  threshold_slope    How much coverage drops per 0.1 increase in threshold
                     (from 0.1 to 0.5). Higher slope = threshold is a better action.
  rank_stability     Std dev of R* chunk ranks across different random noise draws.
                     Lower = more stable retrieval = better RL signal.
  rl_score           Composite score 0–100 combining all metrics. Use this to
                     compare models at a glance. Target: > 60 for correct models.

What you're looking for:
  - "Correct" domain model: high rl_score (>60), high separation (>0.35),
    good coverage@0.30 (>0.75), low pct_above_0.30 (<15%)
  - "Wrong" domain model: lower coverage on OTHER domains, creating a clear
    fault signal when WRONG_EMBEDDING_MODEL is injected
  - General model: moderate rl_score across ALL domains (not specialized)
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Corpus paths ───────────────────────────────────────────────────────────────
_CORPORA_DIR = Path(__file__).parent / "corpora"
_DOMAINS = ["software", "climate", "medical"]


# ── Corpus loading ─────────────────────────────────────────────────────────────

def load_corpus(domain: str) -> Tuple[List[dict], List[dict], Dict[str, List[int]]]:
    """Load chunks, queries, and ground truth for a domain."""
    d = _CORPORA_DIR / domain
    chunks = json.loads((d / "chunks.json").read_text())
    queries = json.loads((d / "queries.json").read_text())
    ground_truth = json.loads((d / "ground_truth.json").read_text())
    return chunks, queries, ground_truth


# ── Embedding ──────────────────────────────────────────────────────────────────

def embed_texts(model_name: str, texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Embed a list of texts using a SentenceTransformer model.
    Returns float32 array of shape (n_texts, dim).
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed. Run: pip install sentence-transformers")
        sys.exit(1)

    import torch
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"    Loading model on {device.upper()}...", end=" ", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
    print("done")

    print(f"    Embedding {len(texts)} texts in batches of {batch_size}...", end=" ", flush=True)
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,   # cosine similarity = dot product after L2 norm
        convert_to_numpy=True,
    )
    print("done")
    return vecs.astype(np.float32)


# ── Similarity computation ─────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute (n_queries x n_chunks) cosine similarity matrix."""
    # Already normalized in embed_texts → dot product suffices
    return (a @ b.T).astype(np.float32)


# ── Metric computation ─────────────────────────────────────────────────────────

def compute_metrics(
    S: np.ndarray,
    queries: List[dict],
    ground_truth: Dict[str, List[int]],
    top_k: int = 10,
) -> Dict:
    """
    Compute all RL-suitability metrics for a similarity matrix S.

    S: (n_queries, n_chunks) float32
    """
    n_q, n_c = S.shape

    # ── Collect R* and non-R* scores ──────────────────────────────────────────
    rstar_scores = []
    nonrstar_scores = []

    for i, q in enumerate(queries):
        qid = str(q["query_id"])
        r_star = set(ground_truth.get(qid, []))
        row = S[i]
        for j in range(n_c):
            if j in r_star:
                rstar_scores.append(float(row[j]))
            else:
                nonrstar_scores.append(float(row[j]))

    rstar_scores = np.array(rstar_scores)
    nonrstar_scores = np.array(nonrstar_scores)

    rstar_mean = float(np.mean(rstar_scores))
    rstar_p10 = float(np.percentile(rstar_scores, 10))
    rstar_p50 = float(np.percentile(rstar_scores, 50))
    rstar_p90 = float(np.percentile(rstar_scores, 90))

    nonrstar_mean = float(np.mean(nonrstar_scores))
    nonrstar_p90 = float(np.percentile(nonrstar_scores, 90))  # worst-case competition

    separation = rstar_mean - nonrstar_mean

    # ── Coverage at various thresholds ────────────────────────────────────────
    def mean_coverage(threshold: float) -> float:
        covs = []
        for i, q in enumerate(queries):
            qid = str(q["query_id"])
            r_star = set(ground_truth.get(qid, []))
            if not r_star:
                continue
            scores = S[i]
            top_idx = np.argsort(scores)[::-1][:top_k]
            retrieved = {int(j) for j in top_idx if scores[j] >= threshold}
            covs.append(len(retrieved & r_star) / len(r_star))
        return float(np.mean(covs)) if covs else 0.0

    cov_nothresh = mean_coverage(0.0)   # pure top-K, no threshold
    cov_020 = mean_coverage(0.20)
    cov_030 = mean_coverage(0.30)
    cov_040 = mean_coverage(0.40)
    cov_050 = mean_coverage(0.50)

    # Threshold slope: how much coverage changes per 0.1 step (0.1 → 0.5)
    # High slope = threshold is a meaningful tuning lever for the agent
    threshold_slope = (cov_020 - cov_050) / 3.0  # 3 × 0.1 steps from 0.2 to 0.5

    # ── % chunks above various thresholds ─────────────────────────────────────
    pct_above_020 = float((S >= 0.20).mean())
    pct_above_030 = float((S >= 0.30).mean())
    pct_above_050 = float((S >= 0.50).mean())

    # ── Empty retrieval rate at default config ────────────────────────────────
    n_empty = sum(
        1 for i, q in enumerate(queries)
        if len([j for j in np.argsort(S[i])[::-1][:top_k] if S[i][j] >= 0.30]) == 0
    )
    empty_rate = n_empty / len(queries)

    # ── R* rank statistics ────────────────────────────────────────────────────
    rstar_ranks = []
    for i, q in enumerate(queries):
        qid = str(q["query_id"])
        r_star = list(ground_truth.get(qid, []))
        if not r_star:
            continue
        scores = S[i]
        sorted_idx = np.argsort(scores)[::-1]
        rank_map = {int(j): pos for pos, j in enumerate(sorted_idx)}
        ranks = [rank_map.get(c, n_c) for c in r_star if c < n_c]
        if ranks:
            rstar_ranks.append(float(np.min(ranks)))  # best-rank R* chunk

    rstar_rank_mean = float(np.mean(rstar_ranks)) if rstar_ranks else n_c
    rstar_rank_p90 = float(np.percentile(rstar_ranks, 90)) if rstar_ranks else n_c
    # % of queries where at least one R* chunk ranks in top-10
    rstar_in_top10 = float(np.mean([r < top_k for r in rstar_ranks])) if rstar_ranks else 0.0

    # ── Multi-hop specific ────────────────────────────────────────────────────
    mh_queries = [q for q in queries if q.get("is_multi_hop")]
    if mh_queries:
        mh_covs = []
        for q in mh_queries:
            qid = str(q["query_id"])
            qrow = queries.index(q)
            r_star = set(ground_truth.get(qid, []))
            scores = S[qrow]
            top_idx = np.argsort(scores)[::-1][:top_k]
            retrieved = {int(j) for j in top_idx if scores[j] >= 0.30}
            mh_covs.append(len(retrieved & r_star) / len(r_star) if r_star else 0.0)
        mh_coverage = float(np.mean(mh_covs))
    else:
        mh_coverage = None

    # ── RL suitability score (0–100) ──────────────────────────────────────────
    # Combines the most important metrics into a single number for quick comparison.
    # Each component is weighted by how much it matters for RL learning quality.
    s_sep    = min(1.0, separation / 0.50) * 35       # separation weight: 35pts
    s_cov    = cov_030 * 25                            # clean coverage: 25pts
    s_discr  = max(0, 1.0 - pct_above_030 / 0.20) * 20  # discrimination: 20pts
                                                         # (reward full pts if <2% of chunks pass)
    s_slope  = min(1.0, threshold_slope / 0.30) * 10  # threshold sensitivity: 10pts
    s_rank   = rstar_in_top10 * 10                     # top-10 hit rate: 10pts
    rl_score = s_sep + s_cov + s_discr + s_slope + s_rank

    return {
        # Score distribution
        "rstar_mean": rstar_mean,
        "rstar_p10": rstar_p10,
        "rstar_p50": rstar_p50,
        "rstar_p90": rstar_p90,
        "nonrstar_mean": nonrstar_mean,
        "nonrstar_p90": nonrstar_p90,
        "separation": separation,
        # Coverage
        "coverage_top10_nothresh": cov_nothresh,
        "coverage_030": cov_030,
        "coverage_020": cov_020,
        "coverage_040": cov_040,
        "coverage_050": cov_050,
        "threshold_slope": threshold_slope,
        # Density
        "pct_above_020": pct_above_020,
        "pct_above_030": pct_above_030,
        "pct_above_050": pct_above_050,
        # Rank
        "rstar_rank_mean": rstar_rank_mean,
        "rstar_rank_p90": rstar_rank_p90,
        "rstar_in_top10": rstar_in_top10,
        # Multi-hop
        "mh_coverage_030": mh_coverage,
        # Empty retrievals
        "empty_rate_030": empty_rate,
        # Summary
        "rl_score": rl_score,
    }


# ── Report printing ────────────────────────────────────────────────────────────

def grade(value: float, thresholds: Tuple, labels: Tuple = ("✓ GOOD", "~ OK", "✗ POOR")) -> str:
    """Return a grade label based on value thresholds (high is good)."""
    if value >= thresholds[0]:
        return labels[0]
    elif value >= thresholds[1]:
        return labels[1]
    else:
        return labels[2]


def print_report(model_name: str, domain: str, metrics: Dict) -> None:
    print(f"\n  {'─'*62}")
    print(f"  Model : {model_name}")
    print(f"  Domain: {domain}")
    print(f"  {'─'*62}")

    sep = metrics["separation"]
    cov = metrics["coverage_030"]
    pct = metrics["pct_above_030"]
    slope = metrics["threshold_slope"]
    top10 = metrics["rstar_in_top10"]
    rl = metrics["rl_score"]

    print(f"  RL Suitability Score : {rl:.1f}/100  {grade(rl, (60, 40))}")
    print()
    print(f"  Score distribution")
    print(f"    R* chunks   : mean={metrics['rstar_mean']:.3f}  p10={metrics['rstar_p10']:.3f}  p50={metrics['rstar_p50']:.3f}  p90={metrics['rstar_p90']:.3f}")
    print(f"    non-R* chunks: mean={metrics['nonrstar_mean']:.3f}  p90={metrics['nonrstar_p90']:.3f}")
    print(f"    Separation  : {sep:.3f}  {grade(sep, (0.35, 0.20))}")
    print()
    print(f"  Coverage (top-k=10)")
    print(f"    No threshold: {metrics['coverage_top10_nothresh']:.3f}")
    print(f"    thresh=0.20 : {metrics['coverage_020']:.3f}")
    print(f"    thresh=0.30 : {cov:.3f}  {grade(cov, (0.75, 0.55))}")
    print(f"    thresh=0.40 : {metrics['coverage_040']:.3f}")
    print(f"    thresh=0.50 : {metrics['coverage_050']:.3f}")
    print(f"    Slope (0.2→0.5 per 0.1): {slope:.3f}  {grade(slope, (0.15, 0.08))}")
    print()
    print(f"  Threshold density")
    print(f"    Chunks ≥0.20: {metrics['pct_above_020']*100:.1f}%")
    print(f"    Chunks ≥0.30: {pct*100:.1f}%  {grade(1-pct, (0.85, 0.70), ('✓ GOOD (<15%)', '~ OK (<30%)', '✗ POOR (>30%)'))}")
    print(f"    Chunks ≥0.50: {metrics['pct_above_050']*100:.1f}%")
    print()
    print(f"  Retrieval rank")
    print(f"    R* rank mean   : {metrics['rstar_rank_mean']:.1f}  (lower is better)")
    print(f"    R* rank p90    : {metrics['rstar_rank_p90']:.1f}")
    print(f"    R* in top-10   : {top10*100:.1f}%  {grade(top10, (0.80, 0.65))}")
    print(f"    Empty retrievals: {metrics['empty_rate_030']*100:.1f}%  (at thresh=0.30)")

    if metrics["mh_coverage_030"] is not None:
        print(f"\n  Multi-hop coverage@0.30: {metrics['mh_coverage_030']:.3f}")

    print()


def print_cross_domain_summary(results: Dict) -> None:
    """
    Print a summary showing each model's coverage across all domains.
    This helps identify which models are "right" vs "wrong" for each domain
    — a high contrast between domains is what makes WRONG_EMBEDDING_MODEL work.
    """
    print("\n" + "═"*80)
    print("CROSS-DOMAIN COVERAGE SUMMARY (coverage@0.30, top-k=10)")
    print("═"*80)

    all_models = sorted({m for (m, _) in results})
    all_domains = _DOMAINS

    # Header
    header = f"  {'Model':<45}"
    for d in all_domains:
        header += f"  {d:<10}"
    header += "  RL_score"
    print(header)
    print("  " + "─"*75)

    for model in all_models:
        row = f"  {model:<45}"
        rl_scores = []
        for domain in all_domains:
            key = (model, domain)
            if key in results:
                cov = results[key]["coverage_030"]
                rl = results[key]["rl_score"]
                rl_scores.append(rl)
                row += f"  {cov:.3f}     "
            else:
                row += f"  {'—':<10}"
        if rl_scores:
            row += f"  {np.mean(rl_scores):.1f}"
        print(row)

    print()
    print("Interpretation:")
    print("  High coverage on native domain + low on others → good 'correct/wrong' model pair")
    print("  Consistent moderate coverage across all domains → good 'general' model")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate embedding models for RAGDebugEnv RL environment suitability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", "-m",
        action="append",
        dest="models",
        required=True,
        help="HuggingFace model name (can be repeated for multiple models)",
    )
    parser.add_argument(
        "--domain", "-d",
        action="append",
        dest="domains",
        choices=_DOMAINS,
        help="Domain(s) to evaluate (default: all). Can be repeated.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of chunks to retrieve (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size (default: 64, reduce if OOM)",
    )
    parser.add_argument(
        "--save",
        metavar="FILE",
        help="Save full results to JSON file",
    )
    args = parser.parse_args()

    domains_to_eval = args.domains or _DOMAINS
    all_results: Dict = {}

    for model_name in args.models:
        print(f"\n{'='*70}")
        print(f"Model: {model_name}")
        print(f"{'='*70}")

        for domain in domains_to_eval:
            print(f"\n  Evaluating on '{domain}' corpus...")

            try:
                chunks, queries, ground_truth = load_corpus(domain)
            except FileNotFoundError as e:
                print(f"  SKIP: {e}")
                continue

            chunk_texts = [c["text"] for c in chunks]
            query_texts = [q["text"] for q in queries]

            try:
                chunk_vecs = embed_texts(model_name, chunk_texts, batch_size=args.batch_size)
                query_vecs = embed_texts(model_name, query_texts, batch_size=args.batch_size)
            except Exception as e:
                print(f"  ERROR embedding: {e}")
                continue

            # Check embedding dimension and warn about context limits
            print(f"    Embedding dim: {chunk_vecs.shape[1]}")

            # Compute similarity matrix
            print(f"    Computing similarity matrix ({len(queries)} × {len(chunks)})...", end=" ", flush=True)
            S = cosine_similarity(query_vecs, chunk_vecs)
            print("done")

            metrics = compute_metrics(S, queries, ground_truth, top_k=args.top_k)
            all_results[(model_name, domain)] = metrics
            print_report(model_name, domain, metrics)

    if len(all_results) > 0:
        print_cross_domain_summary(all_results)

        # Print recommendations
        print("RECOMMENDED ROLES (based on rl_score and cross-domain contrast):")
        print()
        domain_best: Dict[str, Tuple[str, float]] = {}
        for (model, domain), metrics in all_results.items():
            rl = metrics["rl_score"]
            if domain not in domain_best or rl > domain_best[domain][1]:
                domain_best[domain] = (model, rl)
        for domain, (model, score) in sorted(domain_best.items()):
            role = "correct domain model" if score > 60 else "possible domain model (weak)"
            print(f"  {domain:10s}: {model}  (rl_score={score:.1f}) → {role}")
        print()

    if args.save:
        # Serialize for JSON
        serializable = {
            f"{m}|{d}": v
            for (m, d), v in all_results.items()
        }
        Path(args.save).write_text(json.dumps(serializable, indent=2))
        print(f"Results saved to {args.save}")


if __name__ == "__main__":
    main()
