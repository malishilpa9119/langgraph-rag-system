from langchain_core.documents import Document

from app.ingestion.chunker import chunk_documents
from app.ingestion.loader import load_many
from app.services.vectorstore import get_vectorstore


def index_documents(docs: list[Document]) -> dict:
    chunks = chunk_documents(docs)
    vs = get_vectorstore()
    vs.add_documents(chunks)
    return {
        "documents_added": len(docs),
        "chunks_added": len(chunks),
        "sources": sorted({d.metadata.get("source", "unknown") for d in docs}),
    }


def index_from_sources(sources: list[str]) -> dict:
    docs = load_many(sources)
    return index_documents(docs)


def list_indexed_sources() -> list[dict]:
    vs = get_vectorstore()
    counts: dict[str, int] = {}
    for m in vs.list_metadatas():
        source = (m or {}).get("source", "unknown")
        counts[source] = counts.get(source, 0) + 1
    return [{"source": s, "chunk_count": c} for s, c in sorted(counts.items())]
