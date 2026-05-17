from io import BytesIO
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document


def load_from_file(path: str | Path) -> Document:
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".pdf":
        text = _pdf_bytes_to_text(p.read_bytes())
    elif suffix in {".html", ".htm"}:
        text = _html_to_text(p.read_text(encoding="utf-8", errors="ignore"))
    else:
        text = p.read_text(encoding="utf-8", errors="ignore")

    return Document(
        page_content=text,
        metadata={"source": p.name, "source_type": "file", "path": str(p)},
    )


def load_from_url(url: str, timeout: int = 30) -> Document:
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "rag-ingest/1.0"})
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "").lower()

    if "pdf" in content_type or url.lower().endswith(".pdf"):
        text = _pdf_bytes_to_text(resp.content)
    elif "html" in content_type or url.endswith((".html", ".htm")) or "<html" in resp.text[:200].lower():
        text = _html_to_text(resp.text)
    else:
        text = resp.text

    return Document(
        page_content=text,
        metadata={"source": url, "source_type": "url"},
    )


def load_many(sources: Iterable[str]) -> list[Document]:
    docs: list[Document] = []
    for src in sources:
        if src.startswith(("http://", "https://")):
            docs.append(load_from_url(src))
        else:
            docs.append(load_from_file(src))
    return docs


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    return main.get_text(separator="\n", strip=True)


def _pdf_bytes_to_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data))
    parts: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if txt.strip():
            parts.append(f"[page {i}]\n{txt}")
    return "\n\n".join(parts)
