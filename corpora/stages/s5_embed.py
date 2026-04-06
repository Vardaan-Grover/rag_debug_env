"""
stages/s5_embed.py
------------------
Stage 5: Embed chunks + queries with 4 models and compute similarity matrices.

For each domain, writes:
  S_true_general.npy
  S_true_medical.npy
  S_true_legal.npy
  S_true_code.npy

Each matrix has shape (n_queries, n_chunks), dtype float32, and stores:
  cosine_similarity(query_embedding, chunk_embedding)
"""

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

_CORPORA_DIR = Path(__file__).parent.parent

EMBEDDING_MODELS = {
    "general": "sentence-transformers/all-MiniLM-L6-v2",
    "medical": "NeuML/pubmedbert-base-embeddings",
    "legal": "nlpaueb/legal-bert-base-uncased",
    # Keep the historical "code" slot key for API/stability, but use a
    # sentence-transformer-native retriever to avoid degenerate behavior on
    # non-code corpora.
    "code": "sentence-transformers/multi-qa-mpnet-base-dot-v1",
}


def embed_and_compute_similarity(
    chunks: list[dict],
    queries: list[dict],
    domain: str,
    force_reload: bool = False,
) -> None:
    """
    Build and cache S_true matrices for a domain.

    Args:
        chunks:       Output of s2_chunk.chunk_documents(domain).
        queries:      Output of s3_queries.generate_queries(domain) + s4 for medical.
        domain:       "software" | "climate" | "medical"
        force_reload: If True, recompute all matrices even if cached.
    """
    if not chunks:
        raise ValueError("[s5] Cannot embed: chunks is empty")
    if not queries:
        raise ValueError("[s5] Cannot embed: queries is empty")

    out_dir = _CORPORA_DIR / domain
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix_paths = {
        model_key: out_dir / f"S_true_{model_key}.npy"
        for model_key in EMBEDDING_MODELS
    }

    if not force_reload and all(path.exists() for path in matrix_paths.values()):
        print(f"      [s5] All S_true matrices already cached for {domain}, skipping")
        return

    to_compute = [
        key for key, path in matrix_paths.items()
        if force_reload or not path.exists()
    ]

    print(
        f"      [s5] Building {len(to_compute)} matrix/matrices for {domain} "
        f"({len(queries)} queries x {len(chunks)} chunks)..."
    )

    chunk_texts = [c["text"] for c in chunks]
    query_texts = [q["text"] for q in queries]

    for model_key in to_compute:
        model_name = EMBEDDING_MODELS[model_key]
        out_path = matrix_paths[model_key]

        print(f"      [s5] Loading {model_key} embedding model: {model_name}")
        model = SentenceTransformer(model_name)

        print(f"      [s5] Encoding chunks with {model_key} model...")
        chunk_vecs = model.encode(chunk_texts, batch_size=32, show_progress_bar=False)

        print(f"      [s5] Encoding queries with {model_key} model...")
        query_vecs = model.encode(query_texts, batch_size=32, show_progress_bar=False)

        S_true = cosine_similarity(query_vecs, chunk_vecs).astype(np.float32)
        np.save(out_path, S_true)

        print(
            f"      [s5] Saved {out_path.name} shape={S_true.shape} "
            f"range=[{S_true.min():.3f}, {S_true.max():.3f}]"
        )

    print(f"      [s5] Stage 5 complete for {domain}")
