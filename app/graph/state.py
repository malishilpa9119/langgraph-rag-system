from typing import Literal, TypedDict

from langchain_core.documents import Document

QueryType = Literal["conceptual", "how_to", "troubleshooting", "api_reference", "unknown"]


class GraphState(TypedDict, total=False):
    """State that flows between nodes in the RAG workflow.

    Field-by-field rationale:
    - original_question: never mutated; used for the final answer + citations
    - current_question: may be rewritten on each retry
    - query_type: optional classification to guide downstream prompting
    - documents: candidate chunks from retrieval
    - relevant_documents: filtered subset after grading
    - retries: counter so the loop can stop (PDF calls this out explicitly)
    - answer: final generated text
    - sources: citation list shown to the user
    - hallucination_grounded: bonus check flag
    - route_decision: last routing decision (for trace/debug)
    """

    original_question: str
    current_question: str
    query_type: QueryType

    documents: list[Document]
    relevant_documents: list[Document]

    retries: int

    answer: str
    sources: list[dict]

    hallucination_grounded: bool
    route_decision: str
