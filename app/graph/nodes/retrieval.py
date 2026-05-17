from app.config import get_settings
from app.graph.state import GraphState
from app.services.vectorstore import get_vectorstore


def retrieval_node(state: GraphState) -> dict:
    settings = get_settings()
    question = state.get("current_question") or state["original_question"]
    vs = get_vectorstore()
    docs = vs.similarity_search(question, k=settings.TOP_K)
    return {"documents": docs}
