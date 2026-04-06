"""
stages/s1_load.py
-----------------
Stage 1: Load raw documents from authoritative domain sources.

Sources by domain:
  software  — Python 3 official documentation (text archive)
              + HuggingFace documentation (m-ric/huggingface_doc)
  climate   — Wikipedia articles on climate science topics
              (authoritative, long-form, IPCC-citing, no script issues)
  medical   — PubMed QA passages aggregated into longer documents
              + Medical Meadow passages aggregated similarly

Target: 40-60 documents per domain → 200-300 chunks after Stage 2.

Returns:
  list of {"text": str, "source": str, "domain": str}
"""

import json
import re
import urllib.request
import zipfile
import io
import random
from pathlib import Path
from datasets import load_dataset

TARGET_DOCS = 50
MEDICAL_TARGET_DOCS = 90
MIN_WORDS   = 300   # raised from 200 — ensures meaningful chunking
MAX_WORDS   = 5000

# Cache lives next to the per-domain corpus directories, e.g.
# corpora/software/docs.json, corpora/climate/docs.json, etc.
_CORPORA_DIR = Path(__file__).parent.parent


def load_documents(domain: str, force_reload: bool = False) -> list[dict]:
    loaders = {
        "software": _load_software,
        "climate":  _load_climate,
        "medical":  _load_medical,
    }
    if domain not in loaders:
        raise ValueError(f"Unknown domain: {domain!r}")

    cache_path = _CORPORA_DIR / domain / "docs.json"

    if not force_reload and cache_path.exists():
        print(f"      [s1] Loading {domain} docs from cache ({cache_path})...")
        docs = json.loads(cache_path.read_text())
        avg_words = int(sum(len(d["text"].split()) for d in docs) / max(len(docs), 1))
        print(f"      Loaded {len(docs)} documents (avg {avg_words} words each)")
        return docs

    docs = loaders[domain]()
    docs = [d for d in docs if _is_usable(d["text"])]

    avg_words = int(sum(len(d["text"].split()) for d in docs) / max(len(docs), 1))
    print(f"      Loaded {len(docs)} documents (avg {avg_words} words each)")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(docs, ensure_ascii=False, indent=2))
    print(f"      [s1] Cached {len(docs)} docs → {cache_path}")

    return docs


# =============================================================================
# Software — unchanged, already working
# =============================================================================

def _load_software() -> list[dict]:
    docs = []

    print("      [s1] Fetching Python official docs...")
    try:
        python_docs = _fetch_python_docs()
        docs.extend(python_docs)
        print(f"      [s1] Got {len(python_docs)} pages from Python docs")
    except Exception as e:
        print(f"      [s1] Python docs failed ({e}), continuing...")

    print("      [s1] Loading HuggingFace documentation dataset...")
    try:
        ds = load_dataset("m-ric/huggingface_doc", split="train")
        seen = set()
        for row in ds:
            text = (row.get("text") or "").strip()
            if not _is_usable(text):
                continue
            key = text[:80]
            if key in seen:
                continue
            seen.add(key)
            docs.append({
                "text":   _clean(text),
                "source": "huggingface_doc",
                "domain": "software",
            })
            if len(docs) >= TARGET_DOCS:
                break
        print(f"      [s1] Got {len(docs) - len(python_docs)} pages from HF docs")
    except Exception as e:
        print(f"      [s1] HF docs failed ({e}), continuing...")

    return docs[:TARGET_DOCS]


def _fetch_python_docs() -> list[dict]:
    url = "https://docs.python.org/3/archives/python-3.14-docs-text.zip"
    INCLUDE_DIRS = {"tutorial", "library", "reference", "howto"}
    SKIP_FILES = {
        "contents.txt", "index.txt", "genindex.txt",
        "modindex.txt", "search.txt",
    }
    docs = []

    with urllib.request.urlopen(url, timeout=30) as response:
        zip_data = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        for name in zf.namelist():
            if not name.endswith(".txt"):
                continue
            parts = Path(name).parts
            if not any(d in parts for d in INCLUDE_DIRS):
                continue
            if Path(name).name in SKIP_FILES:
                continue

            text = zf.read(name).decode("utf-8", errors="replace").strip()
            text = _clean(text)

            if not _is_usable(text):
                continue

            docs.append({
                "text":   text,
                "source": f"python_docs:{name}",
                "domain": "software",
            })
            if len(docs) >= TARGET_DOCS // 2:
                break

    return docs


# =============================================================================
# Climate — Wikipedia articles on climate science topics
#
# Why Wikipedia over the previous sources:
#   - climate_fever: evidence passages are 1-3 sentences, too short
#   - scientific_papers (arXiv): uses deprecated loading script, fails
#   - Wikipedia: long-form (500-4000 words), authoritative, cites IPCC,
#     uses exactly the cross-disciplinary vocabulary we need, no script issues
# =============================================================================

# Wikipedia article titles covering the climate domain we want
# These are chosen to cover: science, policy, impacts, solutions
# so query diversity is high
CLIMATE_ARTICLES = [
    # Core climate science — confirmed long articles
    "Greenhouse gas",
    "Global warming",
    "Climate change",
    "Carbon dioxide",
    "Methane",
    "Greenhouse effect",
    "Climate model",
    "Carbon cycle",
    "Permafrost",
    "Sea level rise",
    "Ocean acidification",
    "Heat wave",
    "Drought",
    "Flood",
    "Tropical cyclone",
    "Wildfire",
    "Glacier",
    "Ice age",
    "Atlantic Ocean",
    "Ocean current",
    # Policy and agreements
    "Paris Agreement",
    "Kyoto Protocol",
    "Carbon tax",
    "Emissions trading",
    "Intergovernmental Panel on Climate Change",
    "Carbon offset",
    "Carbon footprint",
    "Sustainable development",
    "Environmental policy",
    "Air pollution",
    # Energy and technology
    "Renewable energy",
    "Solar energy",
    "Wind power",
    "Nuclear power",
    "Electric vehicle",
    "Carbon capture and storage",
    "Hydrogen",
    "Energy storage",
    "Energy efficiency",
    "Fossil fuel",
    "Coal",
    "Natural gas",
    "Oil",
    # Impacts and ecosystems
    "Biodiversity",
    "Deforestation",
    "Desertification",
    "Food security",
    "Water resources",
    "Coral reef",
    "Ecosystem",
    "Tundra",
    "Amazon rainforest",
    "Arctic",
    "Antarctica",
    "Ozone depletion",
    "Acid rain",
    "Pollution",
]


def _load_climate() -> list[dict]:
    """
    Load Wikipedia articles on climate topics via the Wikipedia REST API.
    """
    print("      [s1] Loading Wikipedia climate articles...")
    docs = _load_wikipedia_articles(
        article_titles=CLIMATE_ARTICLES,
        domain="climate",
        target=TARGET_DOCS,
    )
    return docs[:TARGET_DOCS]


def _load_wikipedia_articles(
    article_titles: list[str],
    domain: str,
    target: int,
) -> list[dict]:
    """
    Fetch Wikipedia articles by title using the Wikipedia REST API.

    The HF Wikipedia dataset requires a deprecated loading script.
    The Wikipedia REST API is the reliable primary path.

    Rate limit: Wikipedia requests a max of ~200 requests/second for
    anonymous clients. We add a 0.1s delay between requests to be safe.
    """
    import urllib.parse
    import json
    import time

    docs = []
    api_base = "https://en.wikipedia.org/w/api.php"
    headers = {"User-Agent": "RAGDebugEnv/1.0 (research project; contact@example.com)"}

    print(f"      [s1] Fetching up to {len(article_titles)} Wikipedia articles ({domain}) via API...")

    for title in article_titles:
        if len(docs) >= target:
            break
        try:
            params = {
                "action":           "query",
                "titles":           title,
                "prop":             "extracts",
                "explaintext":      "1",       # plain text, no HTML
                "exsectionformat":  "plain",
                "format":           "json",
                "redirects":        "1",       # follow redirects (must be "1" not "true")
            }
            url = api_base + "?" + urllib.parse.urlencode(params)
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())

            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                if page.get("pageid", -1) == -1:
                    # Page does not exist
                    break
                raw = _clean(page.get("extract", "").strip())
                # Truncate BEFORE the usability check so long Wikipedia
                # articles (10,000+ words) are not rejected by MAX_WORDS.
                # MAX_WORDS * 6 is a character-level approximation of MAX_WORDS.
                text = raw[:MAX_WORDS * 6]
                if not _is_usable(text):
                    break
                docs.append({
                    "text":   text,
                    "source": f"wikipedia:{title}",
                    "domain": domain,
                })
                break

            # Polite delay — avoids rate limiting across 50 sequential requests
            time.sleep(1)

        except Exception as e:
            print(f"      [s1] Skipped '{title}': {e}")
            continue

    print(f"      [s1] Got {len(docs)} articles from Wikipedia API ({domain})")
    return docs


# =============================================================================
# Medical — Medical textbooks + Wikipedia medical articles
#
# Why these sources instead of medqa/medical_meadow:
#   - medqa/medical_meadow entries are MCQ exam questions (vignette + A/B/C/D/E)
#     They are question format, not expository prose. A query like
#     "How does insulin resistance develop?" cannot be answered by a
#     document that IS itself a question about a patient case.
#   - We need expository documents: text that explains mechanisms,
#     treatments, and clinical relationships in continuous prose.
#
# Sources:
#   A) medrag/textbooks — actual medical textbook chapters (Harrison's,
#      Robbins, Gray's etc.). Long-form, authoritative, dense terminology.
#      This is exactly what exposes the wrong_embedding_model fault
#      because general models struggle with clinical vocabulary.
#   B) Wikipedia medical articles — long-form, reliable, well-structured.
#      Covers diseases, treatments, anatomy, pharmacology.
# =============================================================================

MEDICAL_WIKI_ARTICLES = [
    "Diabetes mellitus type 2",
    "Hypertension",
    "Myocardial infarction",
    "Atherosclerosis",
    "Chronic kidney disease",
    "Heart failure",
    "Stroke",
    "Chronic obstructive pulmonary disease",
    "Asthma",
    "Pneumonia",
    "Sepsis",
    "Liver cirrhosis",
    "Inflammatory bowel disease",
    "Rheumatoid arthritis",
    "Systemic lupus erythematosus",
    "Multiple sclerosis",
    "Parkinson's disease",
    "Alzheimer's disease",
    "Epilepsy",
    "Major depressive disorder",
    "Schizophrenia",
    "Bipolar disorder",
    "Anxiety disorder",
    "Insulin resistance",
    "Metabolic syndrome",
    "Obesity",
    "Hypothyroidism",
    "Hyperthyroidism",
    "Anemia",
    "Leukemia",
    "Breast cancer",
    "Colorectal cancer",
    "Lung cancer",
    "Prostate cancer",
    "HIV/AIDS",
    "Tuberculosis",
    "Malaria",
    "Dengue fever",
    "COVID-19",
    "Influenza",
    "Antibiotic resistance",
    "Vaccination",
    "Osteoporosis",
    "Osteoarthritis",
    "Gout",
    "Glomerulonephritis",
    "Pancreatitis",
    "Peptic ulcer disease",
    "Appendicitis",
    "Pharmacokinetics",
]


def _load_medical() -> list[dict]:
    docs = []

    # Source A: medrag/textbooks — actual medical textbook content
    # Each row is a pre-chunked passage (~150 words) from Harrison's,
    # Robbins, Gray's Anatomy, etc. We aggregate 4 consecutive passages
    # from the same book to produce ~600-word documents.
    print("      [s1] Loading medical textbooks (MedRAG/textbooks)...")
    try:
        ds = load_dataset("MedRAG/textbooks", split="train")
        seen = set()
        passages_by_book: dict[str, list[str]] = {}

        for row in ds:
            # MedRAG uses "content" as the field name
            text = (row.get("content") or row.get("text") or "").strip()
            book = (row.get("title") or "unknown_book").strip()
            if len(text.split()) < 30:   # MedRAG passages are ~130 words
                continue
            key = text[:80]
            if key in seen:
                continue
            seen.add(key)
            passages_by_book.setdefault(book, []).append(text)

        # Aggregate 10 consecutive passages per book → ~1300-word documents
        # MedRAG passages are ~130 words each: 10 × 130 = ~1300 words
        # This gives 2-3 chunks per document at chunk_size=512 tokens
        PASSAGES_PER_AGG = 10

        # Step 1: Build all possible document batches per book
        docs_by_book: dict[str, list[str]] = {}
        for book, passages in passages_by_book.items():
            book_docs = []
            for i in range(0, len(passages), PASSAGES_PER_AGG):
                batch = passages[i : i + PASSAGES_PER_AGG]
                if len(batch) < 4:
                    continue
                combined = "\n\n".join(batch)
                if not _is_usable(combined):
                    continue
                book_docs.append(combined)
            if book_docs:
                docs_by_book[book] = book_docs

        # Step 2: Round-robin across books — one document per book per rotation
        books = list(docs_by_book.keys())
        random.shuffle(books)  # randomise starting order
        book_indices = {book: 0 for book in books}

        while len(docs) < MEDICAL_TARGET_DOCS:
            made_progress = False
            for book in books:
                if len(docs) >= MEDICAL_TARGET_DOCS:
                    break
                idx = book_indices[book]
                if idx < len(docs_by_book[book]):
                    docs.append({
                        "text":   _clean(docs_by_book[book][idx]),
                        "source": f"medrag_textbooks:{book}",
                        "domain": "medical",
                    })
                    book_indices[book] += 1
                    made_progress = True
            if not made_progress:
                break  # all books exhausted before hitting target

        print(f"      [s1] Got {len(docs)} documents from medical textbooks")
    except Exception as e:
        print(f"      [s1] MedRAG/textbooks failed ({e}), continuing...")

    # Source B: Wikipedia medical articles via API
    # Fills up to MEDICAL_TARGET_DOCS if textbooks didn't reach it
    if len(docs) < MEDICAL_TARGET_DOCS:
        remaining = MEDICAL_TARGET_DOCS - len(docs)
        print(f"      [s1] Loading {remaining} Wikipedia medical articles to reach target...")
        wiki_docs = _load_wikipedia_articles(
            article_titles=MEDICAL_WIKI_ARTICLES,
            domain="medical",
            target=remaining,
        )
        docs.extend(wiki_docs)

    return docs[:MEDICAL_TARGET_DOCS]


# =============================================================================
# Shared helpers
# =============================================================================

def _is_usable(text: str) -> bool:
    if not text:
        return False
    words = text.split()
    if len(words) < MIN_WORDS:
        return False
    if len(words) > MAX_WORDS:
        return False
    alpha_chars = sum(1 for c in text if c.isalpha())
    if alpha_chars / max(len(text), 1) < 0.5:
        return False
    return True


def _clean(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"  +", " ", text)
    return text.strip()