from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.services.embeddings import get_embeddings
from app.services.vectorstore import get_vectorstore


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm the embedding model and vector store at startup
    # so the first user request doesn't pay the cold-load cost.
    print("[startup] Loading embedding model...")
    get_embeddings()
    print("[startup] Loading FAISS index (if exists)...")
    get_vectorstore()
    print("[startup] Ready.")
    yield


app = FastAPI(
    title="RAG Technical Documentation Assistant",
    description=(
        "Self-corrective RAG system built with LangGraph + FastAPI. "
        "Answers questions over an indexed technical documentation corpus."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "service": "RAG Technical Documentation Assistant",
        "docs": "/docs",
        "endpoints": ["/query", "/ingest", "/ingest/files", "/documents", "/feedback", "/health"],
    }
