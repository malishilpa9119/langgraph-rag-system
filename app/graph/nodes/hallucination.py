from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.graph.state import GraphState
from app.services.llm import get_grader_llm


class GroundednessGrade(BaseModel):
    grounded: bool = Field(
        description="True if every factual claim in the answer is supported by the context."
    )


HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You check whether an answer is grounded in the provided context. "
        "Mark grounded=true only if every concrete factual claim is supported. "
        "Citations like [1] do not need to be in the context — only the facts do.",
    ),
    (
        "human",
        "Context:\n{context}\n\nAnswer:\n{answer}\n\nIs the answer grounded?",
    ),
])


def hallucination_node(state: GraphState) -> dict:
    docs = state.get("relevant_documents") or state.get("documents") or []
    answer = state.get("answer", "")
    if not answer or not docs:
        return {"hallucination_grounded": False}

    context = "\n\n".join(d.page_content for d in docs)
    grader = get_grader_llm().with_structured_output(GroundednessGrade)
    try:
        result = (HALLUCINATION_PROMPT | grader).invoke({
            "context": context[:6000],
            "answer": answer,
        })
        return {"hallucination_grounded": bool(result.grounded)}
    except Exception:
        return {"hallucination_grounded": True}
