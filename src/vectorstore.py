"""Vector store management for form document embeddings.

Uses ChromaDB for persistent vector storage and retrieval,
enabling semantic search across multiple forms.
Uses sentence-transformers for local embeddings (no API needed).
Section-aware chunking keeps form sections intact.
"""

import re
from typing import Dict, List, Optional

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from .config import Config


class SectionAwareTextSplitter:
    """Splits form text by sections first, then by size if needed.

    Keeps logical form sections (delimited by === headers, --- lines,
    or double newlines) intact whenever possible. Prepends document
    header to each chunk for context.
    """

    def __init__(self, max_chunk_size: int = 2000, chunk_overlap: int = 200):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self._fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )

    def _extract_header(self, text: str) -> str:
        """Extract the first few lines as document header/title."""
        lines = text.strip().split("\n")
        header_lines = []
        for line in lines[:5]:
            line = line.strip()
            if not line:
                break
            header_lines.append(line)
            # Stop after finding what looks like a title
            if len(header_lines) >= 3:
                break
        return "\n".join(header_lines)

    def split_text(self, text: str) -> List[str]:
        """Split text into section-aware chunks."""
        # Extract header to prepend to each chunk
        header = self._extract_header(text)

        # Split on section markers: === SECTION === or lines of ===
        sections = re.split(r'\n\s*={3,}\s*([^=\n]*?)\s*={3,}\s*\n', text)

        # Rebuild sections with their headers
        parsed_sections = []
        i = 0
        while i < len(sections):
            section_text = sections[i].strip()
            section_title = ""
            # If next element is a section title (captured group)
            if i + 1 < len(sections):
                section_title = sections[i + 1].strip()
                i += 1
            if section_text:
                if section_title:
                    parsed_sections.append(f"=== {section_title} ===\n{section_text}")
                else:
                    parsed_sections.append(section_text)
            i += 1

        # If no sections found, try splitting on double newlines
        if len(parsed_sections) <= 1:
            parsed_sections = [s.strip() for s in text.split("\n\n") if s.strip()]

        # Build chunks, keeping sections together when possible
        chunks = []
        current_chunk = ""

        for section in parsed_sections:
            # If this single section is already too large, split it
            if len(section) > self.max_chunk_size:
                # Flush current chunk first
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                # Split the large section
                sub_chunks = self._fallback_splitter.split_text(section)
                chunks.extend(sub_chunks)
            elif current_chunk and len(current_chunk) + len(section) + 2 > self.max_chunk_size:
                # Flush and start new chunk
                chunks.append(current_chunk.strip())
                current_chunk = section
            else:
                current_chunk += "\n\n" + section if current_chunk else section

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Prepend header context to each chunk (if it's not already the first chunk)
        final_chunks = []
        for i, chunk in enumerate(chunks):
            if i == 0 or header in chunk:
                final_chunks.append(chunk)
            else:
                # Add header as context prefix so LLM knows which document this is from
                final_chunks.append(f"[Document: {header.split(chr(10))[0]}]\n\n{chunk}")

        return final_chunks if final_chunks else [text]

    def create_documents(self, texts: List[str], metadatas: List[dict]) -> List[Document]:
        """Create Document objects from texts."""
        documents = []
        for text, metadata in zip(texts, metadatas):
            chunks = self.split_text(text)
            for i, chunk in enumerate(chunks):
                doc_metadata = {**metadata, "chunk_index": i, "total_chunks": len(chunks)}
                documents.append(Document(page_content=chunk, metadata=doc_metadata))
        return documents


class FormVectorStore:
    """Manages vector embeddings for form documents."""

    def __init__(self):
        Config.validate()
        self._embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
        self._splitter = SectionAwareTextSplitter(
            max_chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )
        self._vectorstore: Optional[Chroma] = None

    @property
    def vectorstore(self) -> Chroma:
        """Get or create the vector store instance."""
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                collection_name="forms",
                embedding_function=self._embeddings,
                persist_directory=Config.CHROMA_PERSIST_DIR,
            )
        return self._vectorstore

    def add_document(self, text: str, metadata: dict) -> int:
        """Add a form document to the vector store.

        Args:
            text: The extracted text content of the form.
            metadata: Metadata dict (source, filename, etc.)

        Returns:
            Number of chunks added.
        """
        chunks = self._splitter.create_documents(
            texts=[text],
            metadatas=[metadata],
        )
        self.vectorstore.add_documents(chunks)
        return len(chunks)

    def search(self, query: str, k: int = 5, filter_metadata: Optional[dict] = None) -> list:
        """Semantic search across stored form documents.

        Args:
            query: The search query.
            k: Number of results to return.
            filter_metadata: Optional metadata filter.

        Returns:
            List of relevant document chunks.
        """
        kwargs = {"k": k}
        if filter_metadata:
            kwargs["filter"] = filter_metadata
        return self.vectorstore.similarity_search(query, **kwargs)

    def get_all_documents(self) -> list:
        """Retrieve all documents in the store."""
        return self.vectorstore.get()

    def get_document_sources(self) -> List[str]:
        """Get list of unique source filenames in the store."""
        all_docs = self.get_all_documents()
        sources = set()
        if all_docs and "metadatas" in all_docs:
            for meta in all_docs["metadatas"]:
                if meta and "filename" in meta:
                    sources.add(meta["filename"])
        return sorted(sources)

    def clear(self):
        """Clear all documents from the vector store."""
        self._vectorstore = None
        import shutil
        import os
        if os.path.exists(Config.CHROMA_PERSIST_DIR):
            shutil.rmtree(Config.CHROMA_PERSIST_DIR)
