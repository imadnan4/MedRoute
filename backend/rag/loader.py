"""Stage 0b — RAG Corpus Loader.

Loads WHO ICD-11 (via REST API) and Medical Meadow WikiDoc (via HuggingFace
datasets) into a ChromaDB vector store indexed by EmbeddingGemma-300M Medical.

Call `load_all()` once on startup to populate the vector store.
"""
from __future__ import annotations

import logging
from config import settings
from rag.retriever import get_or_create_collection

log = logging.getLogger(__name__)

WIKIDOC_CAP = 5000


def _load_wikidoc(collection) -> int:
    """Load Medical Meadow WikiDoc into ChromaDB. Returns doc count."""
    try:
        from datasets import load_dataset
    except ImportError:
        log.warning("datasets not installed; skipping WikiDoc load")
        return 0

    try:
        ds = load_dataset("medalpaca/medical_meadow_wikidoc", split="train")
    except Exception as exc:
        log.warning("Failed to load WikiDoc dataset: %s", exc)
        return 0

    count = 0
    for i, row in enumerate(ds):
        if count >= WIKIDOC_CAP:
            break
        text = row["output"] or row.get("text") or ""
        if len(text) < 50:
            continue
        try:
            collection.add(
                documents=[text[:1000]],
                metadatas=[{"source": "wikidoc", "index": i}],
                ids=[f"wikidoc_{i}"],
            )
            count += 1
        except Exception:
            continue

    log.info("Loaded %d WikiDoc documents into ChromaDB", count)
    return count


def _load_icd(collection) -> int:
    """Load WHO ICD-11 chapters/definitions via REST API."""
    import requests

    headers = {"Accept": "application/json"}
    if settings.icd_api_key:
        headers["APIKey"] = settings.icd_api_key
        headers["Authorization"] = f"Bearer {settings.icd_api_key}"

    base = f"{settings.icd_api_base}/release/11/2024-01/mms/eng"
    count = 0

    try:
        resp = requests.get(base, headers=headers, timeout=15)
        resp.raise_for_status()
        chapters = resp.json().get("child", [])
    except Exception as exc:
        log.warning("WHO ICD-11 API unavailable: %s", exc)
        return 0

    for i, chapter in enumerate(chapters):
        try:
            url = chapter.get("href") if isinstance(chapter, dict) else chapter
            if not url:
                continue
            cresp = requests.get(url, headers=headers, timeout=10)
            cresp.raise_for_status()
            data = cresp.json()
            title = data.get("title", {}).get("@value", "")
            definition = data.get("definition", {}).get("@value", "") if "definition" in data else ""
            text = f"{title}\n\n{definition}".strip()
            if len(text) < 30:
                continue
            collection.add(
                documents=[text[:1000]],
                metadatas=[{"source": "icd11", "index": i, "title": title}],
                ids=[f"icd11_{i}"],
            )
            count += 1
        except Exception:
            continue

    log.info("Loaded %d ICD-11 entries into ChromaDB", count)
    return count


def load_all() -> int:
    """Populate the ChromaDB collection from all sources. Returns total count."""
    collection = get_or_create_collection()

    existing = collection.count()
    if existing > 0:
        log.info("ChromaDB already has %d documents; skipping load", existing)
        return existing

    total = _load_wikidoc(collection) + _load_icd(collection)
    log.info("Load complete: %d total documents in ChromaDB", total)
    return total
