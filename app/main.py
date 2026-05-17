from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

app = FastAPI(
    title="RAG Technical Documentation Assistant",
    description=(
        "Self-corrective RAG system built with LangGraph + FastAPI. "
        "Answers questions over an indexed technical documentation corpus."
    ),
    version="1.0.0",
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
