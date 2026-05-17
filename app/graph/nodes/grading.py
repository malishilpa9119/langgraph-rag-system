from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.graph.state import GraphState
from app.services.llm import get_grader_llm


class GradeDocument(BaseModel):
    """Binary relevance grade for a single retrieved chunk."""

    relevant: bool = Field(description="True if the document helps answer the question.")


GRADE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a strict relevance grader for a RAG pipeline. Given a user "
        "question and a retrieved document chunk, decide if the chunk contains "
        "information that would meaningfully help answer the question. Mark "
        "relevant=true ONLY if there is a clear semantic match — not just "
        "shared keywords. When in doubt, mark relevant=false.",
    ),
    (
        "human",
        "Question:\n{question}\n\nDocument chunk:\n{document}\n\nGrade this chunk.",
    ),
])


def grading_node(state: GraphState) -> dict:
    question = state.get("current_question") or state["original_question"]
    docs = state.get("documents", [])
    if not docs:
        return {"relevant_documents": []}

    grader = get_grader_llm().with_structured_output(GradeDocument)
    chain = GRADE_PROMPT | grader

    relevant: list = []
    for doc in docs:
        try:
            result = chain.invoke({
                "question": question,
                "document": doc.page_content[:2000],
            })
            if result.relevant:
                relevant.append(doc)
        except Exception:
            relevant.append(doc)

    return {"relevant_documents": relevant}
