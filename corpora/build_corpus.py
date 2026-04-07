"""
build_corpus.py
---------------
One-time setup script. Run this before training anything.

Produces for each domain (software, climate, medical):
  chunks.json          — all text chunks with metadata
  queries.json         — synthetic queries with type tags
  ground_truth.json    — {query_id: [relevant_chunk_ids]}
  S_true_[model].npy   — (n_queries x n_chunks) similarity matrices x 4
  corpus_stats.json    — CorpusStats fields for the environment

Usage:
    python -m corpora.build_corpus --domain software
    python -m corpora.build_corpus --domain climate
    python -m corpora.build_corpus --domain medical
    python -m corpora.build_corpus --all
"""

import argparse
import os
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from corpora.stages.s1_load import load_documents
from corpora.stages.s2_chunk import chunk_documents
from corpora.stages.s3_queries import generate_queries
from corpora.stages.s4_multihop import build_multihop_queries
from corpora.stages.s5_embed import embed_and_compute_similarity
from corpora.stages.s6_grade import label_ground_truth
from corpora.stages.verify import verify_corpus

from models import CorpusStats, Domain

DOMAINS = ["software", "climate", "medical"]

def build_domain(domain: str, output_dir: Path):
    print(f"\n{'='*60}")
    print(f"    Building Corpus: {domain.upper()}")
    print(f"{'='*60}\n")

    out = output_dir / domain
    out.mkdir(parents=True, exist_ok=True)

    # Stage 1: Load
    print("[1/6] Loading documents...")
    documents = load_documents(domain)
    print(f"    Loaded {len(documents)} documents.")

    # Stage 2: Chunk
    print("[2/6] Chunking documents...")
    chunks = chunk_documents(documents, domain)
    (out / "chunks.json").write_text(json.dumps(chunks, indent=2))
    print(f"    Created {len(chunks)} chunks.")

    # Stage 3: Generate Queries
    print("[3/6] Generating queries...")
    queries = generate_queries(chunks, domain)
    print(f"    Generated {len(queries)} queries.")

    # Stage 4: Build Multihop Queries
    print("[4/6] Building multihop queries...")
    queries = build_multihop_queries(chunks, queries, domain)
    (out / "queries.json").write_text(json.dumps(queries, indent=2))
    n_multihop = sum(1 for q in queries if q["is_multi_hop"])
    print(f"    Total queries: {len(queries)} ({n_multihop} multihop)")

    # Stage 5: Embed + compute S_true matrices
    print("[5/6] Embedding chunks and queries (this takes a few minutes)...")
    embed_and_compute_similarity(chunks, queries, domain)

    # Stage 6: Cross-encoder ground truth labeling
    print("[6/6] Labeling ground truth with cross-encoder...")
    label_ground_truth(chunks, queries, out)  # writes ground_truth.json internally
    
    # Corpus Stats
    stats = CorpusStats(
        domain=Domain(domain),
        n_documents=len(documents),
        n_chunks=len(chunks),
        avg_chunk_tokens=int(sum(c["n_tokens"] for c in chunks) / len(chunks)),
        has_near_duplicates=(domain == "climate"), # IPCC reports have duplicates
        n_queries=len(queries),
        n_multi_hop_queries=n_multihop
    )
    (out / "corpus_stats.json").write_text(stats.model_dump_json(indent=2))

    # Verify
    print("\n[✓] Verifying corpus...")
    verify_corpus(out, domain)
    print(f"[✓] {domain.upper()} corpus complete.\n")

def main():
    parser = argparse.ArgumentParser(description="Build RAGDebugEnv corpora")
    parser.add_argument("--domain", choices=DOMAINS + ["all"], default="all")
    parser.add_argument("--output-dir", default="corpora")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY", ""):
        raise EnvironmentError("OPENAI_API_KEY not found in environment. Stages 3 and 4 require it.")

    output_dir = Path(args.output_dir)
    domains_to_build = DOMAINS if args.domain == "all" else [args.domain]

    for domain in domains_to_build:
        build_domain(domain, output_dir)
    
    print("\n✅  All corpora built successfully.")
    print(f"    Output: {output_dir.resolve()}")

if __name__ == "__main__":
    main()

