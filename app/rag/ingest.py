import asyncio
from pathlib import Path

from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv"}


def get_loader(file_path: str):
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path)
    if ext in (".txt", ".md"):
        return TextLoader(file_path)
    if ext == ".csv":
        return CSVLoader(file_path)
    raise ValueError(f"Unsupported file type: {ext}")


def _process_sync(file_path: str, filename: str) -> list[Document]:
    loader = get_loader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata["source_filename"] = filename

    return chunks


async def process_document(file_path: str, filename: str) -> list[Document]:
    return await asyncio.to_thread(_process_sync, file_path, filename)
