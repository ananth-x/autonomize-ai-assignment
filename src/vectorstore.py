"""Vector store management for form document embeddings.

Uses ChromaDB for persistent vector storage and retrieval,
enabling semantic search across multiple forms.
Uses sentence-transformers for local embeddings (no API needed).

Chunking strategy:
- Detects record/page boundaries (any repeated separator pattern)
- Falls back to paragraph-based splitting
- Prepends detected context identifiers to each chunk
- Works with any document format (PDF text, OCR output, structured forms, free text)
"""

import re
from typing import Dict, List, Optional

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from .config import Config


class AdaptiveTextSplitter:
    """Splits document text adaptively based on detected structure.

    Works with any document format by detecting boundaries dynamically:
    1. Page markers ([Page X]) from PDF extraction
    2. Repeated separator lines (===, ---, ***, etc.)
    3. Double newlines (paragraph boundaries)
    4. Size-based fallback

    Each chunk gets a context prefix extracted from nearby content
    (e.g., a name, a heading) so the LLM can identify what it's reading.
    """

    def __init__(self, max_chunk_size: int = 1500, chunk_overlap: int = 150):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self._fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )

    def _detect_record_boundaries(self, text: str) -> Optional[str]:
        """Detect if text has repeated separator patterns indicating records.

        Returns the regex pattern if found, None otherwise.
        """
        # Common separator patterns: lines of repeated chars (=, -, *, #)
        patterns = [
            r'\n\s*[=]{10,}\s*\n',   # ========
            r'\n\s*[-]{10,}\s*\n',   # --------
            r'\n\s*[*]{10,}\s*\n',   # ********
            r'\n\s*[#]{10,}\s*\n',   # ########
            r'\n\s*_{10,}\s*\n',     # ________
            r'\[Page \d+\]',          # [Page X] from PDF extraction
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if len(matches) >= 2:  # At least 2 boundaries = multiple records
                return pattern
        return None

    def _extract_context_id(self, block: str) -> str:
        """Extract a short identifier from a text block.

        Looks for common patterns: names, titles, record numbers.
        Falls back to first meaningful line.
        """
        lines = block.strip().split("\n")
        for line in lines[:15]:
            line = line.strip()
            if not line or len(line) < 3:
                continue
            # Skip pure separator lines
            if re.match(r'^[=\-*#_\s]+$', line):
                continue
            # Look for name-like fields
            name_match = re.search(
                r'(?:name|patient|applicant|taxpayer|client)\s*[:\-]\s*(.+)',
                line, re.IGNORECASE
            )
            if name_match:
                return name_match.group(1).strip()[:50]
            # Look for record identifiers
            id_match = re.search(
                r'(?:APPLICANT|PATIENT|RETURN|RECORD|CASE)\s+\d+',
                line, re.IGNORECASE
            )
            if id_match:
                return id_match.group(0)
        # Fallback: first non-empty, non-separator line
        for line in lines[:10]:
            line = line.strip()
            if line and not re.match(r'^[=\-*#_\s]+$', line) and len(line) > 5:
                return line[:60]
        return ""

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks with adaptive boundary detection."""
        # Step 1: Try to split on record boundaries
        boundary_pattern = self._detect_record_boundaries(text)

        if boundary_pattern:
            blocks = re.split(boundary_pattern, text)
            blocks = [b.strip() for b in blocks if b.strip() and len(b.strip()) > 30]
        else:
            blocks = [text]

        # Step 2: For each block, chunk by paragraphs respecting size limit
        all_chunks = []

        for block in blocks:
            context_id = self._extract_context_id(block)
            prefix = f"[{context_id}]" if context_id else ""

            # Split block into paragraphs
            paragraphs = re.split(r'\n\s*\n', block)
            paragraphs = [p.strip() for p in paragraphs if p.strip()]

            current_chunk = ""
            for para in paragraphs:
                if len(para) > self.max_chunk_size:
                    # Flush current chunk
                    if current_chunk.strip():
                        chunk_text = f"{prefix}\n{current_chunk.strip()}" if prefix else current_chunk.strip()
                        all_chunks.append(chunk_text)
                        current_chunk = ""
                    # Split oversized paragraph
                    sub_chunks = self._fallback_splitter.split_text(para)
                    for sc in sub_chunks:
                        chunk_text = f"{prefix}\n{sc}" if prefix else sc
                        all_chunks.append(chunk_text)
                elif current_chunk and len(current_chunk) + len(para) + 2 > self.max_chunk_size:
                    # Flush current chunk
                    chunk_text = f"{prefix}\n{current_chunk.strip()}" if prefix else current_chunk.strip()
                    all_chunks.append(chunk_text)
                    current_chunk = para
                else:
                    current_chunk += "\n\n" + para if current_chunk else para

            if current_chunk.strip():
                chunk_text = f"{prefix}\n{current_chunk.strip()}" if prefix else current_chunk.strip()
                all_chunks.append(chunk_text)

        return all_chunks if all_chunks else [text]

    def create_documents(self, texts: List[str], metadatas: List[dict]) -> List[Document]:
        """Create Document objects from texts."""
        documents = []
        for text, metadata in zip(texts, metadatas):
            chunks = self.split_text(text)
            for i, chunk in enumerate(chunks):
                doc_metadata = {
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
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
        self._splitter = AdaptiveTextSplitter(
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
        """Add a form document to the vector store."""
        chunks = self._splitter.create_documents(
            texts=[text],
            metadatas=[metadata],
        )
        self.vectorstore.add_documents(chunks)
        return len(chunks)

    def search(self, query: str, k: int = 4, filter_metadata: Optional[dict] = None) -> list:
        """Semantic search with deduplication.

        Fetches extra results and removes near-duplicates to ensure
        diverse, relevant context.
        """
        fetch_k = min(k * 2, 20)
        kwargs = {"k": fetch_k}
        if filter_metadata:
            kwargs["filter"] = filter_metadata

        results = self.vectorstore.similarity_search(query, **kwargs)

        # Deduplicate overlapping chunks
        deduplicated = []
        seen_content = []
        for doc in results:
            content = doc.page_content
            is_duplicate = False
            for seen in seen_content:
                words_new = set(content.lower().split())
                words_seen = set(seen.lower().split())
                if len(words_new) > 0:
                    overlap = len(words_new & words_seen) / len(words_new)
                    if overlap > 0.7:
                        is_duplicate = True
                        break
            if not is_duplicate:
                deduplicated.append(doc)
                seen_content.append(content)
                if len(deduplicated) >= k:
                    break

        return deduplicated

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
