"""
stages/s2_chunk.py
------------------
Stage 2: Split raw documents into fixed-size token chunks.

Why tokens not words: embedding models have a max_tokens limit, not a
max_words limit. One token ≈ 0.75 words on average, but varies widely.
Using word count would produce chunks that randomly overflow the model's
context window. Tiktoken gives exact token counts.

Config: chunk_size=512 tokens, overlap=50 tokens (canonical — S_true is
computed against this config, so don't change it after the corpus is built).

Returns:
  list of {
    "chunk_id":    int,   # global counter across all docs
    "text":        str,
    "n_tokens":    int,
    "source_doc":  str,
    "domain":      str,
    "token_start": int,
    "token_end":   int,
  }
"""

import json
from pathlib import Path

import tiktoken

CHUNK_SIZE = 512 # tokens
CHUNK_OVERLAP = 50 # tokens
STRIDE = CHUNK_SIZE - CHUNK_OVERLAP
MIN_CHUNK_TOKENS = 100 # drop stup tail chunks shorter than this

_CORPORA_DIR = Path(__file__).parent.parent

def chunk_documents(
    docs: list[dict],
    domain: str,
    force_reload: bool = False,
) -> list[dict]:
    """
    Chunk all docs for a domain. Returns cached result if available.

    Args:
        docs:         Output of s1_load.load_documents(domain).
        domain:       "software" | "climate" | "medical"
        force_reload: If True, ignore cache and rechunk from scratch.
    """
    cache_path = _CORPORA_DIR / domain / "chunks.json"
    
    if not force_reload and cache_path.exists():
        print(f"      [s2] Loading {domain} chunks from cache ({cache_path})...")
        chunks = json.loads(cache_path.read_text())
        print(f"      [s2] Loaded {len(chunks)} chunks from cache")
        return chunks

    enc = tiktoken.get_encoding("cl100k_base")
    
    chunks: list[dict] = []
    chunk_id = 0
    
    for doc in docs:
        tokens = enc.encode(doc["text"])
        for i in range(0, len(tokens), STRIDE):
            chunk_tokens = tokens[i: i + CHUNK_SIZE]
            if len(chunk_tokens) < MIN_CHUNK_TOKENS:
                break
            chunk_text = enc.decode(chunk_tokens)
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "n_tokens": len(chunk_tokens),
                "source_doc": doc["source"],
                "domain": doc["domain"],
                "token_start": i,
                "token_end": i + len(chunk_tokens),
            })
            chunk_id += 1
            
    
    n_tokens_list = [c["n_tokens"] for c in chunks]
    avg_tokens = int(sum(n_tokens_list) / max(len(chunks), 1))
    print(f"      [s2] Chunked {len(docs)} docs → {len(chunks)} chunks (avg {avg_tokens} tokens)")
    
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2))
    print(f"      [s2] Cached {len(chunks)} chunks → {cache_path}")
    
    return chunks