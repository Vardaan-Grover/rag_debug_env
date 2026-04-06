import json
import sys
import warnings
from typing import Dict
from pathlib import Path
import numpy as np

from src.server.constants import _CORPORA_DIR

_corpus_cache: Dict[str, Dict] = {}

def _load_corpus(domain: str) -> Dict:
    """Load and cache corpus data for a domain. Falls back to synthetic data."""
    if domain in _corpus_cache:
        return _corpus_cache[domain]

    domain_dir = _CORPORA_DIR / domain
    try:
        chunks = json.loads((domain_dir / "chunks.json").read_text())
        queries = json.loads((domain_dir / "queries.json").read_text())
        ground_truth = json.loads((domain_dir / "ground_truth.json").read_text())
        corpus_stats = json.loads((domain_dir / "corpus_stats.json").read_text())

        s_true: Dict[str, np.ndarray] = {}
        for model_name in ["general", "medical", "legal", "code"]:
            path = domain_dir / f"S_true_{model_name}.npy"
            if path.exists():
                s_true[model_name] = np.load(path, mmap_mode="r").astype(np.float32)

        data = {
            "chunks": chunks,
            "queries": queries,
            "ground_truth": ground_truth,
            "corpus_stats": corpus_stats,
            "s_true": s_true,
        }
    except Exception as exc:
        _REQUIRED_FILES = ["chunks.json", "queries.json", "ground_truth.json", "corpus_stats.json"]
        missing = [f for f in _REQUIRED_FILES if not (domain_dir / f).exists()]
        msg = (
            f"\n{'!'*60}\n"
            f"[RAGDebugEnv] WARNING: Real corpus unavailable for domain '{domain}'.\n"
            f"  Reason : {exc}\n"
            + (f"  Missing : {', '.join(missing)}\n" if missing else "")
            + f"  Fix    : run `python -m corpora.build_corpus --domain {domain}`\n"
            f"  Action : falling back to SYNTHETIC corpus — do NOT use for training.\n"
            f"{'!'*60}\n"
        )
        print(msg, file=sys.stderr, flush=True)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        data = _make_synthetic_corpus(domain)

    _corpus_cache[domain] = data
    return data

def _make_synthetic_corpus(domain: str) -> Dict:
    """Generate a minimal synthetic corpus for smoke-testing without real files."""
    rng = np.random.default_rng(0)
    n_chunks = 50
    n_queries = 10

    chunks = [
        {
            "chunk_id": i,
            "text": f"Synthetic chunk {i} for domain {domain}.",
            "n_tokens": 100,
            "source_doc": f"doc_{i // 5}",
            "domain": domain,
        }
        for i in range(n_chunks)
    ]
    queries = [
        {
            "query_id": i,
            "text": f"Synthetic query {i}?",
            "type": "direct",
            "seed_chunk_id": i * 5,
            "is_multi_hop": False,
            "domain": domain,
            "difficulty": "easy",
        }
        for i in range(n_queries)
    ]
    if domain == "medical":
        # Add a few multi-hop queries
        for j in range(3):
            queries.append(
                {
                    "query_id": n_queries + j,
                    "text": f"Synthetic multi-hop query {j}?",
                    "type": "multi_hop",
                    "seed_chunk_ids": [j * 4, j * 4 + 2],
                    "is_multi_hop": True,
                    "domain": domain,
                    "difficulty": "hard",
                }
            )

    ground_truth = {
        str(q["query_id"]): (
            q.get("seed_chunk_ids") or [q["seed_chunk_id"]]
        )
        for q in queries
    }

    s_true_general = rng.uniform(0.2, 0.6, (len(queries), n_chunks)).astype(np.float32)
    # Spike the actual relevant chunks
    for q in queries:
        qidx = q["query_id"]
        for cid in ground_truth[str(qidx)]:
            s_true_general[qidx, cid] = rng.uniform(0.75, 0.95)

    corpus_stats = {
        "domain": domain,
        "n_documents": 10,
        "n_chunks": n_chunks,
        "avg_chunk_tokens": 100,
        "has_near_duplicates": False,
        "n_queries": len(queries),
        "n_multi_hop_queries": sum(1 for q in queries if q.get("is_multi_hop")),
    }

    return {
        "chunks": chunks,
        "queries": queries,
        "ground_truth": ground_truth,
        "corpus_stats": corpus_stats,
        "s_true": {m: s_true_general.copy() for m in ["general", "medical", "legal", "code"]},
    }
