"""Streamlit UI for the RAG documentation assistant.

Run alongside the FastAPI backend:
    uvicorn app.main:app --reload --port 8000
    streamlit run ui/streamlit_app.py
"""

import os
from typing import Any

import requests
import streamlit as st

API_BASE = os.getenv("RAG_API_BASE", "http://localhost:8000")

st.set_page_config(
    page_title="RAG Documentation Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 5rem; max-width: 1100px; }
        .metadata-pill {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 12px;
            margin-right: 8px;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.12);
        }
        .source-block {
            border-left: 3px solid #4a90e2;
            padding: 8px 12px;
            margin: 6px 0;
            background: rgba(74,144,226,0.06);
            border-radius: 4px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def call_query(question: str) -> dict[str, Any]:
    resp = requests.post(f"{API_BASE}/query", json={"question": question}, timeout=120)
    resp.raise_for_status()
    return resp.json()


def call_documents() -> dict[str, Any]:
    resp = requests.get(f"{API_BASE}/documents", timeout=30)
    resp.raise_for_status()
    return resp.json()


def call_ingest(urls: list[str]) -> dict[str, Any]:
    resp = requests.post(f"{API_BASE}/ingest", json={"urls": urls}, timeout=300)
    resp.raise_for_status()
    return resp.json()


def call_feedback(question: str, answer: str, rating: str, comment: str = "") -> dict[str, Any]:
    payload = {"question": question, "answer": answer, "rating": rating, "comment": comment}
    resp = requests.post(f"{API_BASE}/feedback", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def render_sources(sources: list[dict], expanded: bool = False) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", expanded=expanded):
        for s in sources:
            sid = s.get("id", "?")
            src = s.get("source", "")
            snippet = (s.get("snippet") or "").strip().replace("\n", " ")[:280]
            st.markdown(
                f"""
                <div class="source-block">
                    <strong>[{sid}]</strong> <a href="{src}" target="_blank">{src}</a><br>
                    <span style="opacity:0.75; font-size:13px;">{snippet}{'...' if len(snippet) >= 280 else ''}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_metadata(result: dict) -> None:
    grounded = result.get("hallucination_grounded")
    grounded_str = (
        "✓ grounded" if grounded is True
        else ("✗ rejected" if grounded is False else "—")
    )
    qtype = result.get("query_type", "—")
    retries = result.get("retries", 0)
    n_sources = len(result.get("sources", []))

    st.markdown(
        f"""
        <div style="margin: 8px 0 12px 0;">
            <span class="metadata-pill">Type: <b>{qtype}</b></span>
            <span class="metadata-pill">Retries: <b>{retries}</b></span>
            <span class="metadata-pill">Grounded: <b>{grounded_str}</b></span>
            <span class="metadata-pill">Sources: <b>{n_sources}</b></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None


with st.sidebar:
    st.markdown("### Indexed Corpus")

    try:
        docs_info = call_documents()
        # st.metric("Sources", docs_info["total_sources"])
        # st.metric("Chunks", docs_info["total_chunks"])
        with st.expander("View sources", expanded=False):
            for d in docs_info["documents"]:
                st.markdown(f"- `{d['source']}` — {d['chunk_count']} chunks")
    except Exception as e:
        st.warning(f"Could not reach backend at {API_BASE}\n\n{e}")

    st.divider()
    st.markdown("### Ingest New Sources")
    new_urls = st.text_area(
        "URLs (one per line)",
        placeholder="https://fastapi.tiangolo.com/tutorial/security/",
        height=80,
    )
    if st.button("Ingest", use_container_width=True):
        urls = [u.strip() for u in new_urls.splitlines() if u.strip()]
        if not urls:
            st.warning("Enter at least one URL.")
        else:
            with st.spinner("Ingesting..."):
                try:
                    result = call_ingest(urls)
                    st.success(
                        f"Added {result['documents_added']} doc(s) "
                        f"→ {result['chunks_added']} chunks"
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Ingest failed: {e}")

    st.divider()
    if st.button("Clear chat history", use_container_width=True):
        st.session_state.history = []
        st.session_state.last_result = None
        st.rerun()


st.title("📚 Technical Documentation RAG Assistant")
st.caption(
    "Self-corrective RAG over an indexed FastAPI documentation corpus — "
    "powered by LangGraph + Groq + FAISS."
)


for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        if entry.get("result"):
            render_metadata(entry["result"])
            if (
                entry["result"].get("rewritten_question")
                and entry["result"]["rewritten_question"] != entry["question"]
            ):
                with st.expander("Rewritten query"):
                    st.write(entry["result"]["rewritten_question"])
            render_sources(entry["result"].get("sources", []), expanded=False)


question = st.chat_input("Ask a question about the indexed docs...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Running LangGraph workflow..."):
            try:
                result = call_query(question)
            except Exception as e:
                st.error(f"Backend error: {e}")
                st.stop()

        st.session_state.last_result = result
        st.session_state.history.append({
            "question": question,
            "answer": result["answer"],
            "result": result,
        })

        st.write(result["answer"])
        render_metadata(result)

        if (
            result.get("rewritten_question")
            and result["rewritten_question"] != question
        ):
            with st.expander("Rewritten query"):
                st.write(result["rewritten_question"])

        render_sources(result.get("sources", []), expanded=True)


if st.session_state.last_result and st.session_state.history:
    st.divider()
    st.markdown("##### Was this answer helpful?")
    fb_cols = st.columns([1, 1, 6])
    last = st.session_state.history[-1]

    if fb_cols[0].button("👍 Yes", key="fb_up"):
        try:
            call_feedback(last["question"], last["answer"], "up")
            st.toast("Feedback recorded — thank you!", icon="✅")
        except Exception as e:
            st.error(f"Feedback failed: {e}")

    if fb_cols[1].button("👎 No", key="fb_down"):
        try:
            call_feedback(last["question"], last["answer"], "down")
            st.toast("Feedback recorded — we'll improve.", icon="✅")
        except Exception as e:
            st.error(f"Feedback failed: {e}")
