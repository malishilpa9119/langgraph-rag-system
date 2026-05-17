from langchain_core.prompts import ChatPromptTemplate

from app.graph.state import GraphState
from app.services.llm import get_llm

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You rewrite user questions to improve retrieval over a technical "
        "documentation corpus. Expand abbreviations, add likely synonyms or "
        "framework-specific terms, and resolve obvious ambiguity. Keep it ONE "
        "sentence. Do not answer the question. Do not invent facts.",
    ),
    ("human", "Original question:\n{question}\n\nRewritten question:"),
])


CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Classify the user's question into exactly ONE of these categories:\n"
        "- conceptual: asks what something is, definitions, theory\n"
        "- how_to: asks how to do something, steps, tutorial\n"
        "- troubleshooting: asks about errors, debugging, fixing issues\n"
        "- api_reference: asks about specific function/method/parameter signatures\n"
        "\n"
        "Respond with ONLY the single category word — no punctuation, no explanation, "
        "no quotes. Just the word.",
    ),
    ("human", "{question}"),
])

ALLOWED_TYPES = {"conceptual", "how_to", "troubleshooting", "api_reference"}


def query_analysis_node(state: GraphState) -> dict:
    question = state["original_question"]
    llm = get_llm(temperature=0.0)

    rewrite_chain = REWRITE_PROMPT | llm
    rewritten = rewrite_chain.invoke({"question": question}).content.strip()
    if not rewritten:
        rewritten = question

    qtype = "unknown"
    try:
        raw = (CLASSIFY_PROMPT | llm).invoke({"question": question}).content
        cleaned = raw.strip().lower().strip(".,'\"`*- \t\n")
        for token in cleaned.replace("\n", " ").split():
            token = token.strip(".,'\"`*- \t")
            if token in ALLOWED_TYPES:
                qtype = token
                break
    except Exception:
        qtype = "unknown"

    return {
        "current_question": rewritten,
        "query_type": qtype,
        "retries": state.get("retries", 0),
    }
