"""
stages/s3_queries.py
--------------------
Stage 3: Generate synthetic queries from chunks via GPT-4o-mini.

For each domain, select ~25 seed chunks, generate one DIRECT and one PARTIAL
question per chunk, then filter with a cross-encoder to ensure each query
actually matches its seed chunk.

Requires: OPENAI_API_KEY environment variable.

Output per query:
  {
    "query_id":      int,
    "text":          str,
    "type":          "direct" | "partial",
    "seed_chunk_id": int,
    "is_multi_hop":  False,
    "domain":        str,
    "difficulty":    "easy" | "medium",
  }

Writes: corpora/{domain}/queries.json
"""

import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import CrossEncoder

load_dotenv()

SEED_CHUNKS_PER_DOMAIN = 25
MIN_SEED_TOKENS = 150
MIN_ALPHA_RATIO = 0.50
CE_FILTER_THRESHOLD = 0.50

_CORPORA_DIR = Path(__file__).parent.parent

QUERY_GEN_PROMPT = """\
You are building a retrieval benchmark.

Given this text chunk, generate exactly 2 questions:
1. DIRECT: A specific question that this chunk alone completely answers.
   The answer must be explicitly stated in the chunk, not inferred.
2. PARTIAL: A question where this chunk provides essential but incomplete information.
   The full answer requires reading this chunk plus at least one related chunk.

Domain: {domain}
Chunk text:
{chunk_text}

Rules:
- Questions must sound natural, like a real user would ask
- Questions must NOT be answerable from general knowledge alone
- Questions must NOT ask for a list of items (too easy to satisfy partially)
- Do NOT reference "the text" or "the passage" — phrase as standalone questions

Respond in JSON only, no preamble:
{{"direct": "...", "partial": "..."}}
"""

def generate_queries(
    chunks: list[dict],
    domain: str,
    force_reload: bool = False,
) -> list[dict]:
    """
    Generate synthetic queries for a domain from its chunks.

    Args:
        chunks:       Output of s2_chunk.chunk_documents(domain).
        domain:       "software" | "climate" | "medical"
        force_reload: If True, ignore cache and regenerate.
    """
    cache_path = _CORPORA_DIR / domain / "queries.json"
    
    if not force_reload and cache_path.exists():
        print(f"      [s3] Loading {domain} queries from cache ({cache_path})...")
        queries = json.loads(cache_path.read_text())
        print(f"      [s3] Loaded {len(queries)} queries from cache")
        return queries
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=api_key)
    
    seed_chunks = _select_seed_chunks(chunks, SEED_CHUNKS_PER_DOMAIN)
    print(f"      [s3] Selected {len(seed_chunks)} seed chunks for {domain}")

    print(f"      [s3] Loading cross-encoder for quality filtering...")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    queries: list[dict] = []
    query_id = 0
    skipped = 0
    
    for i, chunk in enumerate(seed_chunks):
        print(f"      [s3] Generating queries for chunk {i+1}/{len(seed_chunks)} (id={chunk['chunk_id']})...")
        prompt = QUERY_GEN_PROMPT.format(
            domain=domain,
            chunk_text=chunk["text"],
        )
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.7,
                max_tokens=256,
            )
            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)
        except Exception as e:
            print(f"        [s3] ERROR generating queries for chunk {chunk['chunk_id']}: {e}")
            skipped += 2
            continue
        
        for q_type, difficulty in [("direct", "easy"), ("partial", "medium")]:
            text = parsed.get(q_type, "").strip()
            if not text:
                skipped += 1
                continue
            
            ce_score = cross_encoder.predict([(text, chunk["text"])])[0]
            if ce_score < CE_FILTER_THRESHOLD:
                print(
                    f"      [s3] Filtered {q_type} for chunk {chunk['chunk_id']} "
                    f"(CE={ce_score:.2f})"
                )
                skipped += 1
                continue
            
            queries.append({
                "query_id":      query_id,
                "text":          text,
                "type":          q_type,
                "seed_chunk_id": chunk["chunk_id"],
                "is_multi_hop":  False,
                "domain":        domain,
                "difficulty":    difficulty,
            })
            query_id += 1
            
    print(f"      [s3] Generated {len(queries)} queries ({skipped} skipped/filtered)")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(queries, ensure_ascii=False, indent=2))
    print(f"      [s3] Cached {len(queries)} queries → {cache_path}")

    return queries


def _select_seed_chunks(chunks: list[dict], target: int) -> list[dict]:
    candidates = [
        c for c in chunks
        if c["n_tokens"] >= MIN_SEED_TOKENS
        and _alpha_ratio(c["text"]) >= MIN_ALPHA_RATIO
    ]

    preferred = [c for c in candidates if c["text"].rstrip()[-1] in ".!?"]
    pool = preferred if len(preferred) >= target else candidates

    by_source: dict[str, list[dict]] = {}
    for c in pool:
        by_source.setdefault(c["source_doc"], []).append(c)

    selected: list[dict] = []
    sources = list(by_source.keys())
    random.shuffle(sources)

    i = 0
    while len(selected) < target and i < target * 3:
        source = sources[i % len(sources)]
        available = [c for c in by_source[source] if c not in selected]
        if available:
            selected.append(random.choice(available))
        i += 1

    return selected[:target]


def _alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if c.isalpha()) / len(text)