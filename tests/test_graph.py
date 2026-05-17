"""Lightweight smoke test for the workflow.

Run after ingestion:
    pytest tests/ -s
"""
import os

import pytest

from app.graph.workflow import build_workflow


def test_workflow_compiles():
    """Graph should compile without errors even before any ingestion."""
    wf = build_workflow()
    assert wf is not None


@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
def test_workflow_runs_end_to_end():
    """Smoke test — requires GROQ_API_KEY and a populated vector store."""
    from app.graph.workflow import run_query

    result = run_query("What is FastAPI?")
    assert "answer" in result
    assert isinstance(result["sources"], list)
