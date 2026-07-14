"""Stage 0c — RAG Retriever.

Wraps ChromaDB + EmbeddingGemma-300M Medical for semantic retrieval, with a
built-in seed knowledge base so triage still has grounding when the corpus is
empty or chromadb is not installed.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from config import settings

log = logging.getLogger(__name__)

_collection_cache = None

# Compact primary-care triage knowledge for offline / empty-chroma demos
SEED_KNOWLEDGE: list[tuple[str, str]] = [
    (
        "viral uri cold cough fever headache sore throat",
        "WHO primary care: Uncomplicated viral upper respiratory infection is common. "
        "Supportive care (fluids, rest, antipyretics). Antibiotics are not indicated without "
        "evidence of bacterial infection. Return if dyspnea, chest pain, persistent high fever, "
        "or inability to drink.",
    ),
    (
        "chest pain arm pain sweating dyspnea myocardial infarction acs",
        "Suspected acute coronary syndrome: chest pain/pressure with radiation, diaphoresis, "
        "dyspnea, or syncope requires emergency evaluation. Do not delay for outpatient workup. "
        "Aspirin if not contraindicated per local protocol; urgent ECG and transfer.",
    ),
    (
        "stroke facial droop speech weakness FAST TIA",
        "Stroke/TIA (FAST): Face drooping, Arm weakness, Speech difficulty, Time to call emergency. "
        "Sudden severe headache or acute confusion also warrant emergency assessment. "
        "Time-critical — do not wait for automated triage confirmation.",
    ),
    (
        "fever vomiting diarrhea fatigue sepsis infection",
        "Suspected sepsis: infection with systemic signs (fever, altered mentation, marked "
        "weakness, poor perfusion) needs urgent clinical assessment. Early antibiotics and "
        "fluids per local sepsis pathway when criteria met. Reassess frequently.",
    ),
    (
        "weight loss night sweats fever lymph nodes tuberculosis lymphoma B symptoms",
        "Constitutional B-symptoms (fever, night sweats, weight loss) with or without "
        "lymphadenopathy: evaluate for tuberculosis, HIV, lymphoma, and chronic infection. "
        "Urgent clinician review, labs, and imaging guided by local epidemiology.",
    ),
    (
        "infant fever newborn under 3 months",
        "Fever in infants under 3 months is a medical emergency until proven otherwise. "
        "Immediate clinician assessment; do not rely on home care alone.",
    ),
    (
        "pregnancy bleeding abdominal pain headache dizziness obstetric",
        "Obstetric red flags: vaginal bleeding, severe abdominal pain, severe headache, "
        "visual changes, or syncope in pregnancy require urgent obstetric evaluation.",
    ),
    (
        "vomiting diarrhea dehydration oral rehydration",
        "Acute gastroenteritis: oral rehydration solution is first-line. Escalate if "
        "persistent vomiting, bloody stools, severe abdominal pain, or signs of dehydration "
        "(anuria, lethargy, sunken eyes).",
    ),
    (
        "shortness of breath stridor cyanosis respiratory emergency",
        "Respiratory emergency: severe dyspnea, stridor, cyanosis, or inability to speak "
        "full sentences requires emergency airway/breathing support and immediate transfer.",
    ),
    (
        "anaphylaxis swelling throat rash dyspnea",
        "Anaphylaxis: face/throat swelling, breathing difficulty, rash/hives, collapse — "
        "emergency. Epinephrine IM per protocol; airway support; urgent transfer.",
    ),
]


def _seed_retrieve(query: str, top_k: int) -> list[str]:
    """Keyword overlap ranking over seed knowledge (no embeddings required)."""
    tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    scored: list[tuple[int, str]] = []
    for keys, text in SEED_KNOWLEDGE:
        key_tokens = set(keys.split())
        overlap = len(tokens & key_tokens)
        if overlap > 0:
            scored.append((overlap, text))
    scored.sort(key=lambda x: -x[0])
    return [t for _, t in scored[:top_k]]


def get_or_create_collection():
    """Return the ChromaDB collection (cached after first call).

    Raises ImportError if chromadb is not installed.
    """
    global _collection_cache
    if _collection_cache is not None:
        return _collection_cache

    import chromadb
    from chromadb.utils import embedding_functions as ef

    client = chromadb.PersistentClient(path=settings.chroma_path)
    embedding_fn = ef.SentenceTransformerEmbeddingFunction(model_name=settings.embed_model)

    collection = client.get_or_create_collection(
        name="medical_knowledge",
        embedding_function=embedding_fn,
    )
    _collection_cache = collection
    log.info("ChromaDB collection ready (%d docs)", collection.count())
    return collection


def retrieve(query: str, top_k: Optional[int] = None) -> list[str]:
    """Semantic search over the medical corpus, with seed fallback."""
    k = top_k or settings.rag_top_k

    try:
        collection = get_or_create_collection()
        if collection.count() > 0:
            results = collection.query(query_texts=[query], n_results=min(k, collection.count()))
            documents = results.get("documents", [[]])[0]
            docs = [d for d in (documents or []) if d]
            if docs:
                return docs
            log.warning("ChromaDB query returned empty — using seed knowledge")
        else:
            log.warning("ChromaDB is empty — using seed knowledge")
    except Exception as exc:
        log.warning("ChromaDB unavailable (%s) — using seed knowledge", exc)

    return _seed_retrieve(query, k)
