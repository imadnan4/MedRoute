"""Tool: WHO / medical guideline RAG search via ChromaDB + seed knowledge."""
from __future__ import annotations

from langchain_core.tools import tool

from rag.retriever import retrieve


@tool
def rag_search(query: str) -> str:
    """Search WHO medical guidelines and medical literature for information about
    symptoms, conditions, and treatments. Call this before making a diagnosis on
    moderate/complex cases.
    """
    docs = retrieve(query)
    if not docs:
        return "No relevant medical guidelines found."
    return "\n\n---\n\n".join(docs)
