import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.api.schemas import (
    DocumentInfo,
    DocumentsResponse,
    FeedbackRequest,
    FeedbackResponse,
    IngestResponse,
    IngestURLRequest,
    QueryRequest,
    QueryResponse,
    SourceCitation,
)
from app.graph.workflow import run_query
from app.ingestion.indexer import (
    index_documents,
    index_from_sources,
    list_indexed_sources,
)
from app.ingestion.loader import load_from_file
from app.storage.feedback import save_feedback

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def post_query(req: QueryRequest):
    try:
        result = run_query(req.question)
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow failed: {e!s}",
        )

    return QueryResponse(
        question=result["question"],
        rewritten_question=result.get("rewritten_question"),
        query_type=result.get("query_type"),
        answer=result["answer"],
        sources=[SourceCitation(**s) for s in result.get("sources", [])],
        retries=result.get("retries", 0),
        hallucination_grounded=result.get("hallucination_grounded"),
    )


@router.post("/ingest", response_model=IngestResponse)
def post_ingest_urls(req: IngestURLRequest):
    try:
        result = index_from_sources([str(u) for u in req.urls])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Ingestion failed: {e!s}",
        )
    return IngestResponse(**result)


@router.post("/ingest/files", response_model=IngestResponse)
async def post_ingest_files(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    tmp_paths: list[Path] = []
    try:
        for f in files:
            suffix = Path(f.filename or "doc.txt").suffix or ".txt"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            content = await f.read()
            tmp.write(content)
            tmp.close()
            tmp_paths.append(Path(tmp.name))

        docs = []
        for p, orig in zip(tmp_paths, files):
            doc = load_from_file(p)
            doc.metadata["source"] = orig.filename or doc.metadata["source"]
            docs.append(doc)
        result = index_documents(docs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File ingestion failed: {e!s}")
    finally:
        for p in tmp_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass

    return IngestResponse(**result)


@router.get("/documents", response_model=DocumentsResponse)
def get_documents():
    items = list_indexed_sources()
    return DocumentsResponse(
        total_sources=len(items),
        total_chunks=sum(i["chunk_count"] for i in items),
        documents=[DocumentInfo(**i) for i in items],
    )


@router.post("/feedback", response_model=FeedbackResponse)
def post_feedback(req: FeedbackRequest):
    fid = save_feedback(req.question, req.answer, req.rating, req.comment)
    return FeedbackResponse(status="recorded", feedback_id=fid)


@router.get("/health")
def health():
    return {"status": "ok"}
