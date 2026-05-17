"""Vector store management for form document embeddings.

Uses ChromaDB for persistent vector storage and retrieval,
enabling semantic search across multiple forms.
Uses sentence-transformers for local embeddings (no API needed).
"""

from typing import Dict, List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from .config import Config


class FormVectorStore:
    """Manages vector embeddings for form documents."""

    def __init__(self):
        Config.validate()
        self._embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
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
