"""Tests for the RAG retriever — offline seed-knowledge fallback only.

No network: we exercise ``_seed_retrieve`` directly and ``retrieve`` with a
fake empty collection so the ChromaDB/embedding path is never touched.
"""

from __future__ import annotations

from rag.retriever import _seed_retrieve, retrieve


class _FakeEmptyCollection:
    def count(self) -> int:
        return 0


def test_seed_retrieve_returns_relevant_knowledge() -> None:
    out = _seed_retrieve("chest pain and sweating", 3)
    assert out, "seed retriever should return matches for a known cluster"
    assert any("coronary" in text.lower() for text in out)


def test_retrieve_falls_back_to_seed_offline(monkeypatch) -> None:
    import rag.retriever as retriever

    monkeypatch.setattr(
        retriever, "get_or_create_collection", lambda: _FakeEmptyCollection()
    )
    out = retrieve("fever and cough")
    assert out and any("viral" in text.lower() for text in out)
