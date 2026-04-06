"""
stages/s4_multihop.py
---------------------
Stage 4: Construct multi-hop queries for the medical domain only.

Multi-hop queries require BOTH chunks to answer — neither alone is sufficient.
These are used exclusively in Task 3 (MultiHopDebug) and tracked separately
via multi_hop_coverage in QualityMetrics.

Algorithm:
  1. Embed all medical chunks with the MEDICAL model (in-memory, not saved)
  2. Compute chunk-to-chunk cosine similarity
  3. Find candidate pairs: 0.85 < similarity < 0.97, from different source docs,
     restricted to mechanism-dense books (Pathoma, Pharmacology, Biochemistry, etc.)
     (captures clinically-related but non-redundant cross-book pairs)
  4. Prompt GPT-4o-mini to generate a bridging question for each pair
  5. Validate with cross-encoder: both chunks must score > CE_MIN_SCORE
  6. Keep non-SKIP responses; append to queries.json

Target: 5-8 multi-hop queries for the medical domain.

Appends to corpora/medical/queries.json (written by Stage 3).
"""

import json
import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

_CORPORA_DIR = Path(__file__).parent.parent

MEDICAL_MODEL = "NeuML/pubmedbert-base-embeddings"

# Only books with expository clinical mechanism content are eligible for
# multi-hop pair generation. Books dominated by leadership chapters, ICD
# codes, mortality statistics, staining procedure tables, or exam mnemonics
# produce pairs with no genuine clinical bridge and are excluded implicitly.
#
# Eligible books (mechanism-dense, cross-book bridges possible):
#   Pathoma_Husain      — pathophysiology mechanisms
#   Biochemistry_Lippinco — metabolic pathways
#   Immunology_Janeway  — immune mechanisms
#   Physiology_Levy     — physiological mechanisms
#   Pharmacology_Katzung — drug mechanisms
#   Pathology_Robbins   — pathological mechanisms
#   Cell_Biology_Alberts — cell biology
INCLUDE_BOOKS_MULTIHOP = {
    "medrag_textbooks:Pathoma_Husain",
    "medrag_textbooks:Biochemistry_Lippinco",
    "medrag_textbooks:Immunology_Janeway",
    "medrag_textbooks:Physiology_Levy",
    "medrag_textbooks:Pharmacology_Katzung",
    "medrag_textbooks:Pathology_Robbins",
    "medrag_textbooks:Cell_Biology_Alberts",
}

SIM_LOW             = 0.85   # below this: too dissimilar even for clinical text
SIM_HIGH            = 0.97   # above this: near-duplicate, one chunk answers alone
MIN_CHUNK_DISTANCE  = 15     # min index gap between chunk pairs — ensures different sections
MAX_CANDIDATES      = 60     # pairs to try (GPT will SKIP many)
TARGET_MIN          = 5
TARGET_MAX          = 8
CE_MIN_SCORE        = 0.0    # raw logit threshold — both chunks must score above this

MIN_CHUNK_TOKENS_MULTIHOP = 220
MIN_SHARED_TERMS = 1

TOC_NOISE_RE = re.compile(
    r"\b(summary|references|questions|board of trustees|all rights reserved|"
    r"isbn|library of congress|table of contents|dsm-5|index)\b",
    re.IGNORECASE,
)

# Terms that usually indicate mechanism-rich, bridgeable content.
MECHANISM_TERMS = {
    "receptor",
    "ligand",
    "cytokine",
    "chemokine",
    "antigen",
    "antibody",
    "complement",
    "macrophage",
    "neutrophil",
    "lymphocyte",
    "phagocyt",
    "endocyt",
    "signaling",
    "pathway",
    "enzyme",
    "kinase",
    "metabolism",
    "oxidative",
    "mitochond",
    "apoptosis",
    "inflammation",
    "transport",
    "transporter",
    "channel",
    "membrane",
    "ion",
    "atp",
    "gene",
    "transcription",
    "translation",
    "expression",
    "agonist",
    "antagonist",
    "homeostasis",
}

_PASSAGE_STOPWORDS = {
    "what",
    "which",
    "where",
    "when",
    "from",
    "with",
    "that",
    "this",
    "have",
    "does",
    "into",
    "between",
    "could",
    "would",
    "about",
    "their",
}

MULTIHOP_PROMPT = """\
You are building a multi-hop medical retrieval benchmark.

Chunk A:
{chunk_a}

Chunk B:
{chunk_b}

Shared mechanism terms found in both chunks:
{shared_terms}

Generate ONE question that satisfies ALL of the following:
- CANNOT be meaningfully answered by reading Chunk A alone
- CANNOT be meaningfully answered by reading Chunk B alone
- IS specifically and completely answered when BOTH chunks are read together
- The connection between the chunks must be clinically meaningful — not just that both mention the same anatomical region or body system
- Sounds like a question a physician, surgeon, or medical student would genuinely ask
- Uses concrete entities/processes from BOTH chunks (avoid vague wording)
- 10 to 28 words long

Return SKIP if ANY of the following are true:
- The two chunks describe topics that only share surface-level vocabulary (e.g. both mention "thoracic" but cover unrelated systems)
- Answering the question requires joining two separate sub-questions with "and" — each covering only one chunk
- One chunk alone is sufficient to fully answer the question without needing the other
- No meaningful clinical relationship exists between the content of both chunks
- The question can be asked without naming specific molecules, cells, receptors, pathways, or mediators

Respond with SKIP or with just the question — no preamble, no explanation.
"""


def _extract_mechanism_terms(text: str) -> set[str]:
    lower = text.lower()
    return {term for term in MECHANISM_TERMS if term in lower}


def _is_bridgeable_chunk(chunk: dict) -> bool:
    text = chunk["text"]
    if chunk.get("n_tokens", 0) < MIN_CHUNK_TOKENS_MULTIHOP:
        return False

    alpha_ratio = sum(ch.isalpha() for ch in text) / max(len(text), 1)
    if alpha_ratio < 0.72:
        return False

    toc_hits = len(TOC_NOISE_RE.findall(text))
    section_code_hits = len(re.findall(r"\b\d{1,2}[-–]\d{1,2}\b", text))
    all_caps_hits = len(re.findall(r"\b[A-Z]{3,}\b", text))
    if toc_hits >= 4 or section_code_hits >= 8 or all_caps_hits >= 25:
        return False

    if len(_extract_mechanism_terms(text)) < 2:
        return False

    return True


def _question_focused_passage(question: str, chunk_text: str, max_chars: int = 700) -> str:
    q_terms = {
        t for t in re.findall(r"[a-z]{4,}", question.lower())
        if t not in _PASSAGE_STOPWORDS
    }
    if not q_terms:
        return chunk_text[:max_chars]

    segments = [
        s.strip() for s in re.split(r"[\r\n]+|(?<=[.!?])\s+", chunk_text)
        if s.strip()
    ]
    if not segments:
        return chunk_text[:max_chars]

    scored_segments = []
    for seg in segments:
        seg_lower = seg.lower()
        overlap = sum(1 for t in q_terms if t in seg_lower)
        scored_segments.append((overlap, len(seg), seg))

    scored_segments.sort(key=lambda x: (x[0], x[1]), reverse=True)

    selected = []
    total_len = 0
    for overlap, _, seg in scored_segments:
        if not selected:
            selected.append(seg)
            total_len += len(seg)
            continue

        if overlap == 0 and total_len >= max_chars // 2:
            break
        if total_len + len(seg) > max_chars:
            continue

        selected.append(seg)
        total_len += len(seg)
        if total_len >= max_chars:
            break

    return " ".join(selected)[:max_chars]


def build_multihop_queries(
    chunks: list[dict],
    queries: list[dict],
    domain: str,
    force_reload: bool = False,
) -> list[dict]:
    """
    Generate multi-hop queries and append them to the queries list.

    For non-medical domains, returns queries unchanged immediately.
    For medical, appends multi-hop entries and updates queries.json.

    Args:
        chunks:       Output of `s2_chunk.chunk_documents("medical")`.
        queries:      Output of `s3_queries.generate_queries("medical")`.
        domain:       Domain string — only acts on "medical".
        force_reload: If True, regenerate even if multi-hop queries exist.
    """
    if domain != "medical":
        return queries

    # Check if already done
    existing_multihop = [q for q in queries if q.get("is_multi_hop")]
    if existing_multihop and not force_reload:
        print(f"      [s4] {len(existing_multihop)} multi-hop queries already in cache, skipping")
        return queries

    print(f"      [s4] Embedding {len(chunks)} medical chunks with MEDICAL model...")
    model = SentenceTransformer(MEDICAL_MODEL)
    chunk_texts = [c["text"] for c in chunks]
    chunk_vecs = model.encode(chunk_texts, batch_size=32, show_progress_bar=False)


    print(f"      [s4] Computing chunk-to-chunk similarity...")
    S = cosine_similarity(chunk_vecs, chunk_vecs)
    
    # Diagnostic: understand the similarity distribution
    upper = np.triu(S, k=1)  # upper triangle, excluding diagonal
    vals = upper[upper > 0].flatten()
    print(f"      [s4] Similarity stats — min: {vals.min():.3f}, max: {vals.max():.3f}, mean: {vals.mean():.3f}")
    print(f"      [s4] Pairs below {SIM_LOW}: {(vals < SIM_LOW).sum()}, in [{SIM_LOW},{SIM_HIGH}]: {((vals >= SIM_LOW) & (vals < SIM_HIGH)).sum()}, above {SIM_HIGH}: {(vals >= SIM_HIGH).sum()}")

    # Pre-filter to mechanism-rich, non-index-like chunks before pair search.
    eligible_indices = []
    for i, chunk in enumerate(chunks):
        if chunk["source_doc"] not in INCLUDE_BOOKS_MULTIHOP:
            continue
        if _is_bridgeable_chunk(chunk):
            eligible_indices.append(i)

    print(
        f"      [s4] Eligible chunks after quality/mechanism filters: "
        f"{len(eligible_indices)}/{len(chunks)}"
    )

    if len(eligible_indices) < 2:
        print("      [s4] Not enough eligible chunks after filtering, skipping multi-hop generation")
        return queries

    # Find candidate pairs (upper triangle only in the matrix to avoid duplicates)
    candidates = []
    sim_mid = (SIM_LOW + SIM_HIGH) / 2.0
    for idx_i, i in enumerate(eligible_indices):
        terms_i = _extract_mechanism_terms(chunks[i]["text"])
        for j in eligible_indices[idx_i + 1:]:
            if chunks[i]["source_doc"] == chunks[j]["source_doc"]:
                continue

            if j - i < MIN_CHUNK_DISTANCE:
                continue

            sim = float(S[i][j])
            if not (SIM_LOW < sim < SIM_HIGH):
                continue

            terms_j = _extract_mechanism_terms(chunks[j]["text"])
            shared_terms = sorted(terms_i & terms_j)
            if len(shared_terms) < MIN_SHARED_TERMS:
                continue

            sim_center_score = 1.0 - abs(sim - sim_mid)
            pair_score = sim_center_score + 0.05 * min(len(shared_terms), 4)
            candidates.append((i, j, sim, pair_score, shared_terms[:4]))

    print(f"      [s4] Found {len(candidates)} candidate pairs in similarity range [{SIM_LOW}, {SIM_HIGH}]")

    if not candidates:
        print("      [s4] No candidate pairs found, skipping multi-hop generation")
        return queries

    # Keep top-ranked pairs by bridgeability score.
    candidates.sort(key=lambda x: x[3], reverse=True)
    sampled = candidates[:MAX_CANDIDATES]

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set")
    client = OpenAI(api_key=api_key)

    print(f"      [s4] Loading cross-encoder for pair validation...")
    ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    next_id = max((q["query_id"] for q in queries), default=-1) + 1
    multihop_queries = []

    print(f"      [s4] Generating multi-hop queries from {len(sampled)} candidate pairs...")
    for idx, (i, j, sim, _, shared_terms) in enumerate(sampled):
        if len(multihop_queries) >= TARGET_MAX:
            break

        chunk_a = chunks[i]
        chunk_b = chunks[j]
        prompt = MULTIHOP_PROMPT.format(
            chunk_a=chunk_a["text"],
            chunk_b=chunk_b["text"],
            shared_terms=", ".join(shared_terms),
        )
        
        print(f"\n      --- [s4] Attempt {idx + 1}/{len(sampled)} (sim={sim:.2f}) ---")
        print(f"      [Shared terms]: {', '.join(shared_terms)}")
        print(f"      [Chunk A - ID {chunk_a['chunk_id']}]:\n        {chunk_a['text'][:250]}...")
        print(f"      [Chunk B - ID {chunk_b['chunk_id']}]:\n        {chunk_b['text'][:250]}...")

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=128,
            )
            text = response.choices[0].message.content.strip()
            print(f"      [LLM Response]:\n        {text}")
        except Exception as e:
            print(f"        [s4] ERROR on pair ({chunk_a['chunk_id']}, {chunk_b['chunk_id']}): {e}")
            continue

        if text.upper() == "SKIP" or not text:
            print(f"      [s4] -> Result: SKIPPED")
            continue

        # Validate: both chunks must be relevant to the generated question
        passage_a = _question_focused_passage(text, chunk_a["text"])
        passage_b = _question_focused_passage(text, chunk_b["text"])
        score_a = ce_model.predict([(text, passage_a)])[0]
        score_b = ce_model.predict([(text, passage_b)])[0]

        if score_a < CE_MIN_SCORE or score_b < CE_MIN_SCORE:
            print(
                f"      [s4] CE validation FAILED — "
                f"chunk {chunk_a['chunk_id']} score={score_a:.2f}, "
                f"chunk {chunk_b['chunk_id']} score={score_b:.2f}"
            )
            continue

        print(f"      [s4] -> Result: KEPT")
        multihop_queries.append({
            "query_id":       next_id,
            "text":           text,
            "type":           "multi_hop",
            "seed_chunk_ids": [chunk_a["chunk_id"], chunk_b["chunk_id"]],
            "is_multi_hop":   True,
            "domain":         "medical",
            "difficulty":     "hard",
        })
        next_id += 1

    print(f"      [s4] Generated {len(multihop_queries)} multi-hop queries from {len(sampled)} attempts")

    if len(multihop_queries) < TARGET_MIN:
        print(f"      [s4] WARNING: only {len(multihop_queries)} generated (target {TARGET_MIN}–{TARGET_MAX})")
        print(f"      [s4] Consider lowering SIM_LOW or increasing MAX_CANDIDATES")

    updated = queries + multihop_queries

    cache_path = _CORPORA_DIR / domain / "queries.json"
    cache_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2))
    print(f"      [s4] Updated queries.json → {len(updated)} total ({len(multihop_queries)} multi-hop)")

    return updated