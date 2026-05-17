from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class SourceCitation(BaseModel):
    id: int
    source: str
    snippet: str


class QueryResponse(BaseModel):
    question: str
    rewritten_question: str | None = None
    query_type: str | None = None
    answer: str
    sources: list[SourceCitation]
    retries: int
    hallucination_grounded: bool | None = None


class IngestURLRequest(BaseModel):
    urls: list[HttpUrl] = Field(..., min_length=1, max_length=20)


class IngestResponse(BaseModel):
    documents_added: int
    chunks_added: int
    sources: list[str]


class DocumentInfo(BaseModel):
    source: str
    chunk_count: int


class DocumentsResponse(BaseModel):
    total_sources: int
    total_chunks: int
    documents: list[DocumentInfo]


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=2000)


class FeedbackResponse(BaseModel):
    status: str
    feedback_id: str
