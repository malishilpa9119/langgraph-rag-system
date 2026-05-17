# LangGraph RAG System Documentation

A self-corrective RAG system that answers questions about technical documentation.
Built with **LangGraph** + **FastAPI** + **Groq** + **FAISS**.

Ask it anything about the indexed FastAPI docs, it retrieves the relevant chunks,
grades them for relevance, rewrites the query and retries if nothing fits,
generates a cited answer, and checks the answer for hallucination before returning it.

---

## What's Inside

- LangGraph `StateGraph` with 4 nodes — query analysis, retrieval, grading, generation
- Per-chunk LLM grading + bounded retry loop (`MAX_RETRIES=2`)
- Hallucination / groundedness check after generation
- Streamlit chat UI
- Multi-format ingestion — Markdown, plain text, HTML, PDF, URLs
- All 4 PDF endpoints — `/query`, `/ingest`, `/documents`, `/feedback`
- Interactive Swagger UI at `/docs`

---

## Tech Stack

| Layer | Choice |
|---|---|
| Workflow | LangGraph 0.2 |
| API | FastAPI + Uvicorn |
| LLM | Groq (Llama 3.1 8B Instant) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, 384-dim) |
| Vector store | FAISS (CPU) |
| UI | Streamlit |

---

## Quick Start

You need Python **3.12** and a free Groq API key from [console.groq.com/keys](https://console.groq.com/keys).

```bash
# 1. Setup
git clone <your-repo-url> langgraph-rag-system
cd langgraph-rag-system
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows
# source .venv/bin/activate         # macOS / Linux
pip install -r requirements.txt

# 2. Add your Groq key
cp .env.example .env
# open .env, paste GROQ_API_KEY=gsk_...

# 3. Ingest the default FastAPI docs corpus
python scripts/ingest_docs.py

# 4. Start the API
uvicorn app.main:app --reload --port 8000
```

API ready at **http://localhost:8000/docs**.

### Launch the Streamlit Chat UI (optional)

In a **second terminal** (keep uvicorn running):

```bash
.\.venv\Scripts\Activate.ps1
streamlit run ui/streamlit_app.py
```

UI opens at **http://localhost:8501** — chat with the docs, ingest URLs from the sidebar,
give thumbs-up / thumbs-down feedback per answer.

---

## How It Works

```
        START
          │
          ▼
   query_analysis      rewrite + classify the question
          │
          ▼
      retrieval        top-k chunks from FAISS
          │
          ▼
       grading         LLM marks each chunk relevant / irrelevant
          │
   ┌──────┼────────────────────────┐
   │ relevant?       retries<MAX?   │ retries exhausted
   ▼                  ▼              ▼
generation         rewrite        no_answer
   │                  │              │
   ▼                  └──► retrieval │
hallucination                         │
   │                                  │
   ▼                                  ▼
  END  ◄────────── grounded? ────────END
```

Each node is a pure function `state -> partial_state_update` — see [app/graph/nodes/](app/graph/nodes/).
The retry counter lives in `GraphState` so the conditional edge can route declaratively
(the PDF specifically asks how this is tracked).

---

## Project Layout

```
langgraph-rag-system/
├── app/
│   ├── main.py                FastAPI entry
│   ├── config.py              Settings via pydantic-settings
│   ├── api/
│   │   ├── routes.py          /query /ingest /documents /feedback /health
│   │   └── schemas.py         Pydantic request / response models
│   ├── graph/
│   │   ├── state.py           GraphState TypedDict
│   │   ├── workflow.py        StateGraph builder + run_query()
│   │   └── nodes/             query_analysis, retrieval, grading,
│   │                          rewrite, generation, hallucination
│   ├── ingestion/
│   │   ├── loader.py          file / URL / PDF / HTML loaders
│   │   ├── chunker.py         RecursiveCharacterTextSplitter
│   │   └── indexer.py         chunk -> embed -> store
│   ├── services/
│   │   ├── llm.py             Groq client
│   │   ├── embeddings.py      HuggingFace MiniLM
│   │   └── vectorstore.py     FAISS wrapper (lazy load + persist)
│   └── storage/
│       └── feedback.py        Append-only JSONL feedback log
├── ui/
│   └── streamlit_app.py       Streamlit chat UI
├── scripts/
│   └── ingest_docs.py         CLI ingestion
├── data/                      gitignored — FAISS index + feedback log
├── requirements.txt
├── .env.example
└── README.md
```

---

## API Reference

| Method | Endpoint | What it does |
|---|---|---|
| `POST` | `/query` | Ask a question, get a cited answer |
| `POST` | `/ingest` | Ingest documents from URLs |
| `POST` | `/ingest/files` | Ingest from uploaded files (`.md`, `.txt`, `.html`, `.pdf`) |
| `GET` | `/documents` | List indexed sources + chunk counts |
| `POST` | `/feedback` | Submit thumbs-up / thumbs-down on an answer |
| `GET` | `/health` | Health check |

### `POST /query`

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I declare a path parameter in FastAPI?"}'
```

Response (truncated):

```json
{
  "question": "How do I declare a path parameter in FastAPI?",
  "rewritten_question": "How do you define path parameters in a FastAPI route?",
  "query_type": "how_to",
  "answer": "In FastAPI, path parameters are declared using Python format-string syntax... [1]",
  "sources": [
    {
      "id": 1,
      "source": "https://fastapi.tiangolo.com/tutorial/path-params/",
      "snippet": "You can declare path parameters or variables..."
    }
  ],
  "retries": 0,
  "hallucination_grounded": true
}
```

### `POST /ingest`

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://fastapi.tiangolo.com/tutorial/security/"]}'
```

### `POST /ingest/files`

```bash
curl -X POST http://localhost:8000/ingest/files \
  -F "files=@./my_doc.md" \
  -F "files=@./api_reference.pdf"
```

### `POST /feedback`

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I declare a path parameter?",
    "answer": "...",
    "rating": "up",
    "comment": "Clear answer with good citation."
  }'
```

Feedback is appended to `./data/feedback.jsonl`.

---

## Chunking & Embedding

`RecursiveCharacterTextSplitter` with:

- `chunk_size = 800` characters (~200 tokens)
- `chunk_overlap = 120` (~15 %)
- Section-first separators: `## ` → `### ` → `\n\n` → code fence → `\n` → `. ` → space → char

Why these choices:

- **800 chars** keeps a chunk small enough for one concept but large enough to hold code + its explanation.
- **15 % overlap** preserves meaning across boundaries.
- Splitting on Markdown headings first aligns chunks with documented concepts.
- We split **before** code fences so code blocks usually stay intact.

Embeddings use `all-MiniLM-L6-v2` with `normalize_embeddings=True`, so FAISS's L2 distance equals cosine similarity.

---

## Design Decisions

**LLM — Groq (Llama 3.1 8B Instant).** Free tier, ~500 tok/s. Plenty for narrow tasks like grading and rewriting. A bigger model would help on ambiguous questions but isn't worth the cost here.

**Embeddings — `all-MiniLM-L6-v2`.** Runs locally, no API key, no rate limits, strong baseline. A domain-tuned model (e.g. `bge-large`) would improve recall on code-heavy queries.

**Vector store — FAISS.** Zero-config, persists to disk, pre-built wheels for every platform (no C++ compiler needed on Windows, unlike some Chroma versions). For 100K+ chunks I'd switch to Qdrant or pgvector.

**Per-chunk binary grading.** Each chunk is graded `relevant: bool` independently. The grader prompt is strict ("when in doubt, mark irrelevant") because false positives drag the answer off-topic. Costs N grader calls per query — fast enough on Groq.

**Retry loop.** If everything is irrelevant, rewrite the question and re-retrieve. `MAX_RETRIES=2` prevents infinite loops. After exhaustion the system returns a graceful "I don't know" — never an invented answer.

**Hallucination check.** A second LLM call asks: "is every claim in the answer supported by the context?" If not, we drop the answer and return the no-answer message. Tradeoff: it's an LLM grading itself — has false negatives. Sentence-level NLI would be more rigorous.

**One node per file.** Each node is a pure `state -> partial_state_update` function — easy to read, easy to swap (e.g. replace the grader with a cross-encoder).

---

## What I'd Improve With More Time

1. **Hybrid retrieval** (BM25 + dense, reciprocal-rank fusion) — better recall on literal symbol names.
2. **Reranker** (`bge-reranker-base`) on top-20 dense hits before grading — cheaper, more precise.
3. **Streaming generation** via FastAPI SSE — bigger UX win in the UI.
4. **LangGraph checkpointing** for conversation memory (follow-up questions).
5. **Evaluation harness** — nightly run on a question/expected-source set, catch regressions.
6. **Citation deduplication** — merge multiple chunks from the same source.

---

## Assumptions

- Python **3.12**, ~500 MB free disk (embedding model + FAISS index).
- Corpus ≤ a few thousand chunks — no need for shards or remote vector store.
- English documents only (MiniLM is English-only).
- Citations are `[N]` inline markers tied to a separate `sources` list — not full URLs in prose.
- Feedback is for offline review; not yet fed back into retrieval as a re-ranking signal.

---