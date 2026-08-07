"""
pdf_chat.py
===========
Retrieval-Augmented Generation over uploaded PDFs.

Pipeline:
    upload -> extract text (PyPDF) -> split into chunks (LangChain) ->
    embed (Sentence Transformers) -> store (ChromaDB) -> retrieve on query ->
    build context -> send to Ollama -> return answer + sources.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from utils.constants import (
    CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, VECTORSTORE_DIR,
    UPLOADS_DIR, TOP_K_RESULTS, NO_ANSWER_MESSAGE, MAX_PDF_SIZE_MB,
)
from core.ollama_client import OllamaClient, OllamaConnectionError


class PDFProcessingError(Exception):
    """Raised for any failure while ingesting or reading a PDF."""


@dataclass
class RetrievedChunk:
    """A single retrieved chunk of context, with its source metadata."""

    content: str
    document_name: str
    page_number: int


class PDFChatEngine:
    """Manages the vector store and RAG workflow for uploaded PDFs."""

    def __init__(self) -> None:
        self._embeddings = None  # lazy-loaded (expensive to initialize)
        self._vectorstore: Chroma | None = None

    # ------------------------------------------------------------------
    # Lazy-loaded embedding model (avoids slow app startup)
    # ------------------------------------------------------------------
    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        return self._embeddings

    @property
    def vectorstore(self) -> Chroma:
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                collection_name="pdf_documents",
                embedding_function=self.embeddings,
                persist_directory=str(VECTORSTORE_DIR),
            )
        return self._vectorstore

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def validate_pdf(self, file_path: Path) -> None:
        """Validate file size and that the PDF isn't corrupted before ingesting."""
        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_PDF_SIZE_MB:
            raise PDFProcessingError(
                f"'{file_path.name}' is {size_mb:.1f} MB, which exceeds the "
                f"{MAX_PDF_SIZE_MB} MB limit."
            )
        try:
            reader = PdfReader(str(file_path))
            if len(reader.pages) == 0:
                raise PDFProcessingError(f"'{file_path.name}' has no readable pages.")
        except PdfReadError as exc:
            raise PDFProcessingError(
                f"'{file_path.name}' appears to be corrupted or is not a valid PDF."
            ) from exc

    def ingest_pdf(self, file_path: Path) -> int:
        """
        Extract text from a PDF, chunk it, embed it, and store it in the
        vector store.

        Returns:
            The number of chunks stored.

        Raises:
            PDFProcessingError: on any extraction/validation failure.
        """
        self.validate_pdf(file_path)

        try:
            reader = PdfReader(str(file_path))
            pages_text = [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
        except Exception as exc:  # noqa: BLE001 - surface any pypdf failure cleanly
            raise PDFProcessingError(
                f"Failed to extract text from '{file_path.name}': {exc}"
            ) from exc

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

        documents: list[Document] = []
        for page_num, text in pages_text:
            if not text.strip():
                continue
            for chunk in splitter.split_text(text):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "document_name": file_path.name,
                            "page_number": page_num,
                            "chunk_hash": hashlib.md5(chunk.encode()).hexdigest(),
                        },
                    )
                )

        if not documents:
            raise PDFProcessingError(
                f"'{file_path.name}' contains no extractable text "
                "(it may be a scanned/image-only PDF)."
            )

        # Avoid duplicate embeddings: skip chunks we've already stored.
        existing = self._existing_chunk_hashes()
        new_docs = [
            d for d in documents if d.metadata["chunk_hash"] not in existing
        ]
        if new_docs:
            try:
                self.vectorstore.add_documents(new_docs)
                self.vectorstore.persist()
            except Exception as exc:
                raise PDFProcessingError(
                    f"Failed to store extracted text in the database: {exc}"
                ) from exc

        return len(new_docs)

    def _existing_chunk_hashes(self) -> set[str]:
        """Return the set of chunk hashes already stored, to dedupe re-uploads."""
        try:
            data = self.vectorstore.get(include=["metadatas"])
            return {m.get("chunk_hash", "") for m in data.get("metadatas", [])}
        except Exception:  # noqa: BLE001 - empty/uninitialized store
            return set()

    def remove_document(self, document_name: str) -> None:
        """Delete all chunks belonging to a given document from the store."""
        try:
            self.vectorstore.delete(where={"document_name": document_name})
            self.vectorstore.persist()
        except Exception as exc:  # noqa: BLE001
            raise PDFProcessingError(
                f"Could not remove '{document_name}' from the vector store: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Retrieval + generation
    # ------------------------------------------------------------------
    def retrieve(self, query: str, k: int = TOP_K_RESULTS) -> list[RetrievedChunk]:
        """Retrieve the top-k most relevant chunks for a query."""
        try:
            results = self.vectorstore.similarity_search(query, k=k)
        except Exception as exc:  # noqa: BLE001 - e.g. empty collection
            print(f"Retrieval error: {exc}")
            return []
        return [
            RetrievedChunk(
                content=doc.page_content,
                document_name=doc.metadata.get("document_name", "unknown"),
                page_number=doc.metadata.get("page_number", 0),
            )
            for doc in results
        ]

    def answer_question(
        self,
        query: str,
        model: str,
        client: OllamaClient,
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        history_messages: list[dict[str, str]] | None = None,
    ) -> tuple[Generator[str, None, None], list[RetrievedChunk]]:
        """
        Search the vectorstore for `query`, build a RAG prompt (including conversation history if provided),
        and yield the assistant's streamed response.
        """
        chunks = self.retrieve(query)
        if not chunks:
            def _no_answer() -> Generator[str, None, None]:
                yield NO_ANSWER_MESSAGE
            return _no_answer(), []

        context = "\n\n".join(
            f"[Source: {c.document_name}, page {c.page_number}]\n{c.content}"
            for c in chunks
        )
        system_prompt = (
            "You are a helpful assistant that answers questions using ONLY "
            "the provided document excerpts. If the excerpts don't contain "
            f"the answer, respond exactly with: \"{NO_ANSWER_MESSAGE}\". "
            "Cite the document name and page number when relevant."
        )
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        if history_messages:
            messages.extend(history_messages[-6:])
        messages.append(
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        )

        generator = client.stream_chat(
            model=model, messages=messages, temperature=temperature,
            top_p=top_p, max_tokens=max_tokens,
        )
        return generator, chunks
