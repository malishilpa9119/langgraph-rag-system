from functools import lru_cache

from langchain_groq import ChatGroq

from app.config import get_settings


@lru_cache
def get_llm(temperature: float = 0.0) -> ChatGroq:
    settings = get_settings()
    if not settings.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file. "
            "Get a free key at https://console.groq.com/keys"
        )
    return ChatGroq(
        model=settings.LLM_MODEL,
        temperature=temperature,
        api_key=settings.GROQ_API_KEY,
    )


@lru_cache
def get_grader_llm() -> ChatGroq:
    settings = get_settings()
    if not settings.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set.")
    return ChatGroq(
        model=settings.GRADER_MODEL,
        temperature=0.0,
        api_key=settings.GROQ_API_KEY,
    )
