from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    docs: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Document]:
    """Split docs into chunks tuned for technical documentation.

    Splitter respects code blocks and section structure by preferring
    paragraph/line breaks before falling back to word/char splits.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n## ",
            "\n### ",
            "\n#### ",
            "\n\n",
            "\n```",
            "\n",
            ". ",
            " ",
            "",
        ],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata = {**c.metadata, "chunk_id": i}
    return chunks
