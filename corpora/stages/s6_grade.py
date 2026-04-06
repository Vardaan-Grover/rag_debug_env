"""
stages/s6_grade.py
------------------
Stage 6: Cross-encoder R* labeling.

For each domain, scores every (query, chunk) pair with a cross-encoder
reranker to build the ground-truth relevance set R* used for grading
coverage during RL episodes.

Model: BAAI/bge-reranker-large
  Fallback: BAAI/bge-reranker-base (if large is too slow)
  Minimum:  cross-encoder/ms-marco-MiniLM-L-12-v2 (avoid on medical/legal)

Algorithm:
  For each query:
    1. Score every (query_text, chunk_text) pair with the cross-encoder
    2. Collect chunk_ids where score > RELEVANCE_THRESHOLD (default 0.70)
    3. Always include seed chunk(s) regardless of cross-encoder score
  Save: corpora/{domain}/ground_truth.json  →  {str(query_id): [chunk_id, ...]}

Calibration:
  After running, check mean R* size:
    Target 1-4 for direct queries, 2-5 for multi-hop.
    If mean < 0.8  → lower RELEVANCE_THRESHOLD to 0.60 and re-run
    If mean > 8    → raise RELEVANCE_THRESHOLD to 0.80 and re-run
"""

import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder


def _get_device() -> str:
    """Return the best available device: mps (Apple GPU) > cuda > cpu."""
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"      [s6] Embedding device: {device.upper()}")
    return device

_CORPORA_DIR = Path(__file__).parent.parent

RELEVANCE_THRESHOLD = 0.70
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-v2-m3"
BATCH_SIZE = 32


def grade_domain(domain: str, force_reload: bool = False) -> dict:
    """
    Run cross-encoder grading for one domain and write ground_truth.json.

    Args:
        domain:       "software" | "climate" | "medical"
        force_reload: If True, recompute even if ground_truth.json exists.

    Returns:
        ground_truth dict: {str(query_id): [chunk_id, ...]}
    """
    domain_dir = _CORPORA_DIR / domain
    gt_path = domain_dir / "ground_truth.json"

    if not force_reload and gt_path.exists():
        print(f"      [s6] ground_truth.json already exists for {domain}, skipping")
        with open(gt_path) as f:
            return json.load(f)

    chunks_path = domain_dir / "chunks.json"
    queries_path = domain_dir / "queries.json"

    if not chunks_path.exists():
        raise FileNotFoundError(f"      [s6] Missing chunks.json for {domain}")
    if not queries_path.exists():
        raise FileNotFoundError(f"      [s6] Missing queries.json for {domain}")

    with open(chunks_path) as f:
        chunks = json.load(f)
    with open(queries_path) as f:
        queries = json.load(f)

    print(f"      [s6] Loading cross-encoder: {CROSS_ENCODER_MODEL}")
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device=_get_device())

    chunk_texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    ground_truth: dict[str, list[int]] = {}
    n_queries = len(queries)

    for i, query in enumerate(queries):
        query_text = query["text"]
        query_id = query["query_id"]

        pairs = [(query_text, ct) for ct in chunk_texts]
        scores = cross_encoder.predict(pairs, batch_size=BATCH_SIZE, show_progress_bar=False)

        relevant_ids = [
            chunk_ids[j]
            for j, score in enumerate(scores)
            if score > RELEVANCE_THRESHOLD
        ]

        # Always include seed chunk(s) regardless of cross-encoder score
        if query.get("is_multi_hop") and "seed_chunk_ids" in query:
            seed_ids = query["seed_chunk_ids"]
        else:
            seed_ids = [query["seed_chunk_id"]]

        relevant_ids = list(set(relevant_ids + seed_ids))
        ground_truth[str(query_id)] = relevant_ids

        if (i + 1) % 10 == 0 or (i + 1) == n_queries:
            print(f"      [s6] {domain}: {i + 1}/{n_queries} queries graded")

    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    _print_calibration_stats(ground_truth, queries, domain)

    print(f"      [s6] Wrote {gt_path}")
    return ground_truth


def _print_calibration_stats(
    ground_truth: dict,
    queries: list[dict],
    domain: str,
) -> None:
    """Print R* size stats to guide threshold calibration."""
    sizes = [len(v) for v in ground_truth.values()]
    mean_size = np.mean(sizes)
    min_size = np.min(sizes)
    max_size = np.max(sizes)

    direct_sizes = [
        len(ground_truth[str(q["query_id"])])
        for q in queries
        if not q.get("is_multi_hop", False)
    ]
    multihop_sizes = [
        len(ground_truth[str(q["query_id"])])
        for q in queries
        if q.get("is_multi_hop", False)
    ]

    print(f"\n      [s6] === Calibration stats for {domain} ===")
    print(f"      [s6]   Threshold used : {RELEVANCE_THRESHOLD}")
    print(f"      [s6]   Mean R* size   : {mean_size:.2f}  (target: 1-4 direct, 2-5 multi-hop)")
    print(f"      [s6]   Min / Max      : {min_size} / {max_size}")
    if direct_sizes:
        print(f"      [s6]   Direct mean   : {np.mean(direct_sizes):.2f}")
    if multihop_sizes:
        print(f"      [s6]   Multi-hop mean: {np.mean(multihop_sizes):.2f}")

    if mean_size < 0.8:
        print(f"      [s6] WARNING: Mean R* too small — lower RELEVANCE_THRESHOLD to 0.60")
    elif mean_size > 8:
        print(f"      [s6] WARNING: Mean R* too large — raise RELEVANCE_THRESHOLD to 0.80")
    else:
        print(f"      [s6] Threshold looks good.")
    print()


def label_ground_truth(
    chunks: list[dict],
    queries: list[dict],
    out_dir: Path,
) -> dict:
    """
    Convenience wrapper matching the build_corpus.py call signature.

    Scores all (query, chunk) pairs with the cross-encoder, writes
    ground_truth.json to out_dir, and returns the dict.

    Args:
        chunks:  List of chunk dicts (must have "chunk_id" and "text").
        queries: List of query dicts (must have "query_id", "text",
                 "seed_chunk_id" or "seed_chunk_ids", "is_multi_hop").
        out_dir: Directory where ground_truth.json will be written.

    Returns:
        ground_truth: {str(query_id): [chunk_id, ...]}
    """
    gt_path = out_dir / "ground_truth.json"

    print(f"      [s6] Loading cross-encoder: {CROSS_ENCODER_MODEL}")
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device=_get_device())

    chunk_texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]
    n_queries = len(queries)

    ground_truth: dict[str, list[int]] = {}

    for i, query in enumerate(queries):
        query_text = query["text"]
        query_id = query["query_id"]

        pairs = [(query_text, ct) for ct in chunk_texts]
        scores = cross_encoder.predict(pairs, batch_size=BATCH_SIZE, show_progress_bar=False)

        relevant_ids = [
            chunk_ids[j]
            for j, score in enumerate(scores)
            if score > RELEVANCE_THRESHOLD
        ]

        if query.get("is_multi_hop") and "seed_chunk_ids" in query:
            seed_ids = query["seed_chunk_ids"]
        else:
            seed_ids = [query["seed_chunk_id"]]

        relevant_ids = list(set(relevant_ids + seed_ids))
        ground_truth[str(query_id)] = relevant_ids

        if (i + 1) % 10 == 0 or (i + 1) == n_queries:
            print(f"      [s6] {i + 1}/{n_queries} queries graded")

    # Derive domain name from out_dir for calibration printout
    domain = out_dir.name
    _print_calibration_stats(ground_truth, queries, domain)

    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"      [s6] Wrote {gt_path}")
    return ground_truth


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 6: Cross-encoder R* labeling")
    parser.add_argument(
        "--domain",
        choices=["software", "climate", "medical", "all"],
        default="all",
        help="Domain to grade (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if ground_truth.json already exists",
    )
    parser.add_argument(
        "--model",
        default=CROSS_ENCODER_MODEL,
        help=f"Cross-encoder model to use (default: {CROSS_ENCODER_MODEL})",
    )
    args = parser.parse_args()

    # Allow model override from CLI
    if args.model != CROSS_ENCODER_MODEL:
        CROSS_ENCODER_MODEL = args.model
        print(f"[s6] Using model override: {CROSS_ENCODER_MODEL}")

    domains = ["software", "climate", "medical"] if args.domain == "all" else [args.domain]

    for domain in domains:
        print(f"\n[s6] Grading domain: {domain}")
        grade_domain(domain, force_reload=args.force)

    print("[s6] Done.")
