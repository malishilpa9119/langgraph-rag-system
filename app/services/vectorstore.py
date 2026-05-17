from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import get_settings
from app.services.embeddings import get_embeddings


class VectorStore:
    """Thin wrapper around FAISS that handles lazy init + persistence."""

    def __init__(self) -> None:
        self._vs: FAISS | None = None
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        settings = get_settings()
        path = settings.vector_store_path
        if (path / "index.faiss").exists():
            self._vs = FAISS.load_local(
                str(path),
                get_embeddings(),
                allow_dangerous_deserialization=True,
            )

    def _save(self) -> None:
        if self._vs is None:
            return
        settings = get_settings()
        self._vs.save_local(str(settings.vector_store_path))

    def add_documents(self, docs: list[Document]) -> None:
        if not docs:
            return
        if self._vs is None:
            self._vs = FAISS.from_documents(docs, get_embeddings())
        else:
            self._vs.add_documents(docs)
        self._save()

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        if self._vs is None:
            return []
        return self._vs.similarity_search(query, k=k)

    def list_metadatas(self) -> list[dict]:
        if self._vs is None:
            return []
        return [
            doc.metadata
            for doc in self._vs.docstore._dict.values()  # type: ignore[attr-defined]
        ]

    @property
    def is_empty(self) -> bool:
        return self._vs is None


@lru_cache
def get_vectorstore() -> VectorStore:
    return VectorStore()


def reset_vectorstore_cache() -> None:
    get_vectorstore.cache_clear()
