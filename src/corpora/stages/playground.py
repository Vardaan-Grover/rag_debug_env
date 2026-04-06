from pathlib import Path

import numpy as np

from src.corpora.stages.s1_load import load_documents
from src.corpora.stages.s2_chunk import chunk_documents
from src.corpora.stages.s3_queries import generate_queries
from src.corpora.stages.s4_multihop import build_multihop_queries
from src.corpora.stages.s5_embed import embed_and_compute_similarity
from src.corpora.stages.s6_grade import grade_domain

for domain in ["software", "climate", "medical"]:
    print(f"\n{'='*50}")
    try:
        docs = load_documents(domain)
        chunks = chunk_documents(docs, domain)

        token_counts = [c["n_tokens"] for c in chunks]
        print(f"{domain}: {len(docs)} docs → {len(chunks)} chunks (avg {int(sum(token_counts)/len(token_counts))} tokens)")

        queries = generate_queries(chunks, domain)
        queries = build_multihop_queries(chunks, queries, domain)

        # Stage 5: build/load S_true matrices for this domain
        embed_and_compute_similarity(chunks, queries, domain)

        # Stage 6: cross-encoder R* labeling
        ground_truth = grade_domain(domain)

        domain_dir = Path(__file__).parent.parent / domain
        matrix_summaries = []
        for model in ["general", "medical", "legal", "code"]:
            matrix_path = domain_dir / f"S_true_{model}.npy"
            if matrix_path.exists():
                matrix = np.load(matrix_path, mmap_mode="r")
                matrix_summaries.append(
                    f"{model}:{matrix.shape[0]}x{matrix.shape[1]}"
                )
            else:
                matrix_summaries.append(f"{model}:missing")

        print(f"  S_true matrices: {', '.join(matrix_summaries)}")

        r_star_sizes = [len(v) for v in ground_truth.values()]
        print(f"  ground_truth: {len(ground_truth)} entries, mean R* size = {sum(r_star_sizes)/len(r_star_sizes):.2f}")

        by_type: dict[str, list] = {}
        for q in queries:
            by_type.setdefault(q["type"], []).append(q)

        print(f"  Queries: {len(queries)} total")
        for q_type, qs in sorted(by_type.items()):
            print(f"    {q_type}: {len(qs)}")

        chunk_by_id = {c["chunk_id"]: c for c in chunks}
        seen_chunks: set[int] = set()
        sample_count = 0
        print(f"\n  Sample queries (chunk → query):")
        for q in queries:
            if sample_count >= 5:
                break
            # multi-hop queries use seed_chunk_ids (list), others use seed_chunk_id
            if q.get("is_multi_hop"):
                cids = q["seed_chunk_ids"]
                key = tuple(cids)
                if key in seen_chunks:
                    continue
                seen_chunks.add(key)
                previews = " + ".join(
                    f"chunk {cid}: \"{chunk_by_id.get(cid, {}).get('text', '')[:60].replace(chr(10), ' ').strip()}...\""
                    for cid in cids
                )
                print(f"\n  {previews}")
                print(f"    [multi_hop] {q['text']}")
            else:
                cid = q["seed_chunk_id"]
                if cid in seen_chunks:
                    continue
                seen_chunks.add(cid)
                chunk_text = chunk_by_id.get(cid, {}).get("text", "")
                chunk_preview = chunk_text[:120].replace("\n", " ").strip()
                print(f"\n  chunk {cid}: \"{chunk_preview}...\"")
                for q2 in queries:
                    if not q2.get("is_multi_hop") and q2.get("seed_chunk_id") == cid:
                        print(f"    [{q2['type']:7s}] {q2['text']}")
            sample_count += 1

    except Exception as e:
        import traceback
        print(f"{domain}: FAILED — {e}")
        traceback.print_exc()
