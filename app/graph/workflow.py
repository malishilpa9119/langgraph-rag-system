from functools import lru_cache
from typing import Literal

from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.graph.nodes.generation import generation_node, no_answer_node
from app.graph.nodes.grading import grading_node
from app.graph.nodes.hallucination import hallucination_node
from app.graph.nodes.query_analysis import query_analysis_node
from app.graph.nodes.retrieval import retrieval_node
from app.graph.nodes.rewrite import rewrite_node
from app.graph.state import GraphState


def _route_after_grading(state: GraphState) -> Literal["generate", "rewrite", "no_answer"]:
    settings = get_settings()
    if state.get("relevant_documents"):
        return "generate"
    if state.get("retries", 0) < settings.MAX_RETRIES:
        return "rewrite"
    return "no_answer"


def _route_after_generation(state: GraphState) -> Literal["accept", "no_answer"]:
    settings = get_settings()
    if not settings.ENABLE_HALLUCINATION_CHECK:
        return "accept"
    return "accept" if state.get("hallucination_grounded", True) else "no_answer"


def build_workflow():
    graph = StateGraph(GraphState)

    graph.add_node("query_analysis", query_analysis_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("grading", grading_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("generation", generation_node)
    graph.add_node("hallucination", hallucination_node)
    graph.add_node("no_answer", no_answer_node)

    graph.add_edge(START, "query_analysis")
    graph.add_edge("query_analysis", "retrieval")
    graph.add_edge("retrieval", "grading")

    graph.add_conditional_edges(
        "grading",
        _route_after_grading,
        {
            "generate": "generation",
            "rewrite": "rewrite",
            "no_answer": "no_answer",
        },
    )

    graph.add_edge("rewrite", "retrieval")
    graph.add_edge("generation", "hallucination")

    graph.add_conditional_edges(
        "hallucination",
        _route_after_generation,
        {"accept": END, "no_answer": "no_answer"},
    )

    graph.add_edge("no_answer", END)

    return graph.compile()


@lru_cache
def get_workflow():
    return build_workflow()


def run_query(question: str) -> dict:
    workflow = get_workflow()
    initial: GraphState = {
        "original_question": question,
        "current_question": question,
        "retries": 0,
    }
    final = workflow.invoke(initial)
    return {
        "question": final.get("original_question", question),
        "rewritten_question": final.get("current_question"),
        "query_type": final.get("query_type"),
        "answer": final.get("answer", ""),
        "sources": final.get("sources", []),
        "retries": final.get("retries", 0),
        "hallucination_grounded": final.get("hallucination_grounded"),
    }
