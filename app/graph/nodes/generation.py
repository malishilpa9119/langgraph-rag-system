from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from app.graph.state import GraphState
from app.services.llm import get_llm

GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a precise technical documentation assistant. Answer the "
        "user's question using ONLY the provided context. Rules:\n"
        "1. If the context does not contain the answer, say so explicitly.\n"
        "2. Do not invent function names, parameters, or behavior.\n"
        "3. Cite sources inline as [1], [2], ... matching the numbered "
        "context blocks below.\n"
        "4. Prefer short, structured answers. Use code blocks for code.\n",
    ),
    (
        "human",
        "Question:\n{question}\n\nContext:\n{context}\n\nAnswer:",
    ),
])


def _format_context(docs: list[Document]) -> tuple[str, list[dict]]:
    parts: list[str] = []
    sources: list[dict] = []
    for i, doc in enumerate(docs, start=1):
        src = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] (source: {src})\n{doc.page_content}")
        sources.append({
            "id": i,
            "source": src,
            "snippet": doc.page_content[:240],
        })
    return "\n\n---\n\n".join(parts), sources


def generation_node(state: GraphState) -> dict:
    question = state["original_question"]
    docs = state.get("relevant_documents") or state.get("documents") or []

    if not docs:
        return {
            "answer": (
                "I don't have enough information in the indexed documentation "
                "to answer this question confidently."
            ),
            "sources": [],
        }

    context, sources = _format_context(docs)
    llm = get_llm(temperature=0.1)
    chain = GENERATION_PROMPT | llm
    answer = chain.invoke({"question": question, "context": context}).content
    return {"answer": answer, "sources": sources}


def no_answer_node(state: GraphState) -> dict:
    """Terminal node used when retries are exhausted with zero relevant docs."""
    return {
        "answer": (
            "I couldn't find relevant information in the indexed documents "
            "to answer that confidently. You could try rephrasing the question "
            "or ingesting more documentation."
        ),
        "sources": [],
        "relevant_documents": [],
    }
