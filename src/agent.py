"""Intelligent Form Agent - Core QA and Summarization Pipeline.

This module implements the main agent that can:
1. Answer questions about a single form
2. Generate summaries of individual forms
3. Provide holistic insights across multiple forms
4. Extract and explain structured fields
"""

from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

from .config import Config
from .extractors import FormExtractor, StructuredFieldExtractor
from .vectorstore import FormVectorStore


class FormAgent:
    """Intelligent agent for form understanding, QA, and summarization."""

    def __init__(self):
        Config.validate()
        self._llm = ChatOpenAI(
            model=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            api_key=Config.GROQ_API_KEY,
            base_url=Config.GROQ_BASE_URL,
        )
        self._extractor = FormExtractor()
        self._field_extractor = StructuredFieldExtractor()
        self._vectorstore = FormVectorStore()
        self._loaded_forms: Dict[str, dict] = {}  # filename -> extracted data

    @property
    def loaded_forms(self) -> List[str]:
        """List of currently loaded form filenames."""
        return list(self._loaded_forms.keys())

    def load_form(self, file_path: str) -> dict:
        """Load and process a form file.

        Extracts text, identifies fields, and indexes into vector store.

        Args:
            file_path: Path to the form file.

        Returns:
            Dict with extraction results and metadata.
        """
        # Extract text
        result = self._extractor.extract(file_path)
        text = result["text"]
        metadata = result["metadata"]

        # Extract structured fields
        fields = self._field_extractor.extract_fields(text)

        # Store in vector store for retrieval
        num_chunks = self._vectorstore.add_document(text, metadata)

        # Cache locally
        form_data = {
            "text": text,
            "metadata": metadata,
            "fields": fields,
            "num_chunks": num_chunks,
        }
        self._loaded_forms[metadata["filename"]] = form_data

        return form_data

    def ask(self, question: str, form_name: Optional[str] = None) -> str:
        """Ask a question about loaded forms.

        Args:
            question: The question to answer.
            form_name: Optional specific form to query. If None, searches all.

        Returns:
            The agent's answer as a string.
        """
        # Build context from relevant chunks
        filter_meta = None
        if form_name and form_name in self._loaded_forms:
            filter_meta = {"filename": form_name}

        relevant_docs = self._vectorstore.search(
            query=question,
            k=5,
            filter_metadata=filter_meta,
        )

        if not relevant_docs:
            # Fall back to loaded form text directly
            if form_name and form_name in self._loaded_forms:
                context = self._loaded_forms[form_name]["text"]
            elif self._loaded_forms:
                context = "\n\n---\n\n".join(
                    f"[{name}]\n{data['text']}"
                    for name, data in self._loaded_forms.items()
                )
            else:
                return "No forms have been loaded. Please load a form first."
        else:
            context = "\n\n".join(doc.page_content for doc in relevant_docs)

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an intelligent form analysis agent. You help users understand "
                "form documents by answering questions accurately based on the provided context. "
                "If the answer is not found in the context, say so clearly. "
                "Be precise and cite specific values from the forms when possible."
            )),
            ("human", (
                "Context from form document(s):\n"
                "---\n"
                "{context}\n"
                "---\n\n"
                "Question: {question}\n\n"
                "Provide a clear, accurate answer based on the form content above."
            )),
        ])

        chain = prompt | self._llm
        response = chain.invoke({"context": context, "question": question})
        return response.content

    def summarize(self, form_name: Optional[str] = None) -> str:
        """Generate a summary of a form or all loaded forms.

        Args:
            form_name: Specific form to summarize. If None, summarizes all.

        Returns:
            Summary text.
        """
        if form_name and form_name in self._loaded_forms:
            forms_to_summarize = {form_name: self._loaded_forms[form_name]}
        elif form_name:
            return f"Form '{form_name}' not found. Loaded forms: {self.loaded_forms}"
        else:
            forms_to_summarize = self._loaded_forms

        if not forms_to_summarize:
            return "No forms have been loaded. Please load a form first."

        # Build context
        context_parts = []
        for name, data in forms_to_summarize.items():
            fields_str = ""
            if data["fields"]:
                fields_str = "\n\nExtracted Fields:\n" + "\n".join(
                    f"  - {k}: {v}" for k, v in data["fields"].items()
                )
            context_parts.append(f"[Form: {name}]\n{data['text']}{fields_str}")

        context = "\n\n===\n\n".join(context_parts)

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an intelligent form analysis agent. Generate a clear, concise summary "
                "of the form document(s) provided. Highlight the most important information "
                "including: form type/purpose, key fields and their values, dates, parties involved, "
                "and any notable details. Structure the summary for easy reading."
            )),
            ("human", (
                "Form content:\n"
                "---\n"
                "{context}\n"
                "---\n\n"
                "Generate a comprehensive yet concise summary of this form. "
                "Highlight the most important details that someone would need to know "
                "without reading the full form."
            )),
        ])

        chain = prompt | self._llm
        response = chain.invoke({"context": context})
        return response.content

    def holistic_analysis(self, question: str) -> str:
        """Provide holistic insights across all loaded forms.

        This analyzes patterns, commonalities, and differences across
        multiple forms to answer cross-document questions.

        Args:
            question: The analytical question spanning multiple forms.

        Returns:
            Holistic analysis response.
        """
        if len(self._loaded_forms) < 2:
            return (
                "Holistic analysis requires at least 2 loaded forms. "
                f"Currently loaded: {len(self._loaded_forms)} form(s)."
            )

        # Gather context from all forms
        context_parts = []
        for name, data in self._loaded_forms.items():
            fields_str = ""
            if data["fields"]:
                fields_str = "\nKey Fields: " + ", ".join(
                    f"{k}={v}" for k, v in list(data["fields"].items())[:15]
                )
            context_parts.append(
                f"[Form: {name}]{fields_str}\n{data['text'][:2000]}"
            )

        context = "\n\n===\n\n".join(context_parts)

        # Also get relevant chunks via semantic search
        relevant_docs = self._vectorstore.search(query=question, k=8)
        if relevant_docs:
            extra_context = "\n\n".join(
                f"[Relevant excerpt from {doc.metadata.get('filename', 'unknown')}]\n{doc.page_content}"
                for doc in relevant_docs
            )
            context += f"\n\n=== Additional Relevant Excerpts ===\n\n{extra_context}"

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an intelligent form analysis agent performing holistic analysis "
                "across multiple form documents. Your job is to identify patterns, "
                "commonalities, differences, and provide cross-document insights. "
                "Compare and contrast information across forms. "
                "Provide specific examples and data points from the forms to support your analysis."
            )),
            ("human", (
                "Multiple form documents:\n"
                "---\n"
                "{context}\n"
                "---\n\n"
                "Holistic Analysis Question: {question}\n\n"
                "Analyze across ALL the forms above and provide comprehensive insights. "
                "Reference specific forms and values in your answer."
            )),
        ])

        chain = prompt | self._llm
        response = chain.invoke({"context": context, "question": question})
        return response.content

    def explain_fields(self, form_name: str) -> str:
        """Explain the extracted fields from a specific form.

        Args:
            form_name: The form to explain fields for.

        Returns:
            Explanation of form fields and their significance.
        """
        if form_name not in self._loaded_forms:
            return f"Form '{form_name}' not found. Loaded forms: {self.loaded_forms}"

        data = self._loaded_forms[form_name]
        fields = data["fields"]

        if not fields:
            return f"No structured fields were extracted from '{form_name}'."

        fields_str = "\n".join(f"  - {k}: {v}" for k, v in fields.items())

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an intelligent form analysis agent. Explain the extracted fields "
                "from a form document. For each field, explain what it means, why it matters, "
                "and any notable observations about the value."
            )),
            ("human", (
                "Form: {form_name}\n\n"
                "Extracted Fields:\n{fields}\n\n"
                "Full form text (for context):\n{text}\n\n"
                "Explain each field: what it represents, its significance, "
                "and any observations about the values."
            )),
        ])

        chain = prompt | self._llm
        response = chain.invoke({
            "form_name": form_name,
            "fields": fields_str,
            "text": data["text"][:3000],
        })
        return response.content

    def reset(self):
        """Clear all loaded forms and vector store."""
        self._loaded_forms.clear()
        self._vectorstore.clear()
