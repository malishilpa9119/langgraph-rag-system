from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import get_settings


@lru_cache
def get_embeddings() -> HuggingFaceEmbeddings:
    settings = get_settings()
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu",
            "model_kwargs": {
                "torch_dtype": "float32",
                "low_cpu_mem_usage": False,
            },
        },
        encode_kwargs={"normalize_embeddings": True},
    )
