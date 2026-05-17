from langchain_core.prompts import ChatPromptTemplate

from app.graph.state import GraphState
from app.services.llm import get_llm

REWRITE_FOR_RETRY = ChatPromptTemplate.from_messages([
    (
        "system",
        "The previous retrieval attempt returned irrelevant documents. "
        "Reformulate the user's question to surface different but related "
        "concepts, synonyms, or alternative phrasings the docs might use. "
        "Output ONE sentence. Do not answer the question.",
    ),
    (
        "human",
        "Original question:\n{original}\n\n"
        "Previous attempt:\n{previous}\n\n"
        "New rewritten question:",
    ),
])


def rewrite_node(state: GraphState) -> dict:
    llm = get_llm(temperature=0.2)
    chain = REWRITE_FOR_RETRY | llm
    new_q = chain.invoke({
        "original": state["original_question"],
        "previous": state.get("current_question", state["original_question"]),
    }).content.strip()
    if not new_q:
        new_q = state["original_question"]
    return {
        "current_question": new_q,
        "retries": state.get("retries", 0) + 1,
    }
