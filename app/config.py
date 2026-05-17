from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    GROQ_API_KEY: str = ""

    LLM_MODEL: str = "llama-3.1-8b-instant"
    GRADER_MODEL: str = "llama-3.1-8b-instant"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    VECTOR_STORE_DIR: str = "./data/faiss_index"
    COLLECTION_NAME: str = "tech_docs"

    TOP_K: int = 4
    MAX_RETRIES: int = 2

    ENABLE_HALLUCINATION_CHECK: bool = True

    FEEDBACK_FILE: str = "./data/feedback.jsonl"

    @property
    def vector_store_path(self) -> Path:
        path = Path(self.VECTOR_STORE_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def feedback_path(self) -> Path:
        path = Path(self.FEEDBACK_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()
