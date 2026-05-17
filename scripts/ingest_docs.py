"""Standalone ingestion script.

Usage:
    python scripts/ingest_docs.py              # ingest default FastAPI doc URLs
    python scripts/ingest_docs.py url1 url2    # ingest specific URLs / files
"""
import sys
from pathlib import Path

# Make `app` importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingestion.indexer import index_from_sources

DEFAULT_SOURCES = [
    "https://fastapi.tiangolo.com/tutorial/first-steps/",
    "https://fastapi.tiangolo.com/tutorial/path-params/",
    "https://fastapi.tiangolo.com/tutorial/query-params/",
    "https://fastapi.tiangolo.com/tutorial/body/",
    "https://fastapi.tiangolo.com/tutorial/dependencies/",
]


def main() -> None:
    sources = sys.argv[1:] or DEFAULT_SOURCES
    print(f"Ingesting {len(sources)} source(s)...")
    for s in sources:
        print(f"  - {s}")
    result = index_from_sources(sources)
    print("\nDone.")
    print(f"  documents_added: {result['documents_added']}")
    print(f"  chunks_added:    {result['chunks_added']}")
    print(f"  sources:")
    for s in result["sources"]:
        print(f"    - {s}")


if __name__ == "__main__":
    main()
