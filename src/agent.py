"""Intelligent Form Agent - Core QA and Summarization Pipeline.

Optimized for Groq free tier (6K TPM on llama-3.1-8b-instant).
Each request is budgeted to stay under 6K tokens total:
  - System prompt: ~50 tokens
  - User question: ~50 tokens
  - Context: ~5000 tokens (~3500 chars)
  - Response: ~800 tokens (max_tokens cap)
  Total: ~5900 tokens per request
"""

from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

from .config import Config
from .extractors import FormExtractor, StructuredFieldExtractor
from .vectorstore import FormVectorStore


# Token budget for context portion only.
# 6K TPM total - ~100 tokens (prompts) - 800 tokens (response) = ~5100 tokens for context
# At ~1.3 tokens per word, ~4 chars per token: ~3500 chars is safe
MAX_CONTEXT_CHARS = 3500


class FormAgent:
    """Intelligent agent for form understanding, QA, and summarization."""

    def __init__(self):
        Config.validate()
        self._llm = ChatOpenAI(
            model=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            api_key=Config.GROQ_API_KEY,
            base_url=Config.GROQ_BASE_URL,
            max_tokens=800,
        )
        self._extractor = FormExtractor()
        self._field_extractor = StructuredFieldExtractor()
        self._vectorstore = FormVectorStore()
        self._loaded_forms: Dict[str, dict] = {}

    @property
    def loaded_forms(self) -> List[str]:
        """List of currently loaded form filenames."""
        return list(self._loaded_forms.keys())

    def load_form(self, file_path: str) -> dict:
        """Load and process a form file."""
        result = self._extractor.extract(file_path)
        text = result["text"]
        metadata = result["metadata"]

        fields = self._field_extractor.extract_fields(text)
        num_chunks = self._vectorstore.add_document(text, metadata)

        form_data = {
            "text": text,
            "metadata": metadata,
            "fields": fields,
            "num_chunks": num_chunks,
        }
        self._loaded_forms[metadata["filename"]] = form_data
        return form_data

    def _build_context(self, docs: List[Document], budget: int = MAX_CONTEXT_CHARS) -> str:
        """Assemble context from docs within character budget.

        Adds chunks one by one until budget is reached.
        Never truncates mid-chunk.
        """
        parts = []
        total = 0
        for doc in docs:
            content = doc.page_content
            if total + len(content) + 4 > budget:
                break
            parts.append(content)
            total += len(content) + 4
        return "\n\n".join(parts)

    def ask(self, question: str, form_name: Optional[str] = None) -> str:
        """Ask a question about loaded forms."""
        if not self._loaded_forms:
            return "No forms have been loaded. Please load a form first."

        filter_meta = None
        if form_name and form_name in self._loaded_forms:
            filter_meta = {"filename": form_name}

        relevant_docs = self._vectorstore.search(
            query=question, k=4, filter_metadata=filter_meta
        )

        if not relevant_docs:
            return "Could not find relevant information in the loaded forms."

        context = self._build_context(relevant_docs)

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Answer using ONLY the provided form data. Be precise, cite values. "
             "If info is missing from context, say so."),
            ("human", "{context}\n\nQ: {question}"),
        ])

        chain = prompt | self._llm
        response = chain.invoke({"context": context, "question": question})
        return response.content

    def summarize(self, form_name: Optional[str] = None) -> str:
        """Generate a summary of a form or all loaded forms."""
        if form_name and form_name in self._loaded_forms:
            forms_to_summarize = {form_name: self._loaded_forms[form_name]}
        elif form_name:
            return f"Form '{form_name}' not found. Loaded: {self.loaded_forms}"
        else:
            forms_to_summarize = self._loaded_forms

        if not forms_to_summarize:
            return "No forms loaded. Use 'load' first."

        # Use fields as compact representation + first part of text
        budget_per_form = MAX_CONTEXT_CHARS // len(forms_to_summarize)
        context_parts = []

        for name, data in forms_to_summarize.items():
            # Fields are the most token-efficient representation
            if data["fields"]:
                fields_lines = [f"  {k}: {v}" for k, v in list(data["fields"].items())[:25]]
                part = f"[{name}]\n" + "\n".join(fields_lines)
            else:
                part = f"[{name}]\n{data['text'][:budget_per_form]}"
            context_parts.append(part[:budget_per_form])

        context = "\n\n".join(context_parts)[:MAX_CONTEXT_CHARS]

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Summarize the form data: type, key people, important values, dates."),
            ("human", "{context}\n\nProvide a structured summary."),
        ])

        chain = prompt | self._llm
        response = chain.invoke({"context": context})
        return response.content

    def holistic_analysis(self, question: str) -> str:
        """Cross-form analysis using retrieval + field summaries."""
        if len(self._loaded_forms) < 2:
            return (
                f"Need at least 2 forms for holistic analysis. "
                f"Currently loaded: {len(self._loaded_forms)}."
            )

        # Compact field overview of each form
        field_parts = []
        for name, data in self._loaded_forms.items():
            if data["fields"]:
                top = ", ".join(f"{k}={v}" for k, v in list(data["fields"].items())[:8])
                field_parts.append(f"[{name}] {top}")
        fields_overview = "\n".join(field_parts)

        # Retrieval for question-specific detail
        relevant_docs = self._vectorstore.search(query=question, k=3)
        retrieved = self._build_context(
            relevant_docs, MAX_CONTEXT_CHARS - len(fields_overview) - 50
        )

        context = f"{fields_overview}\n\n{retrieved}"

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Compare and contrast across forms. Cite specific values."),
            ("human", "{context}\n\nQ: {question}"),
        ])

        chain = prompt | self._llm
        response = chain.invoke({"context": context, "question": question})
        return response.content

    def explain_fields(self, form_name: str) -> str:
        """Explain extracted fields from a form."""
        if form_name not in self._loaded_forms:
            return f"Form '{form_name}' not found. Loaded: {self.loaded_forms}"

        data = self._loaded_forms[form_name]
        fields = data["fields"]

        if not fields:
            return f"No structured fields extracted from '{form_name}'."

        # Only send fields, no raw text — saves tokens
        fields_str = "\n".join(
            f"  {k}: {v}" for k, v in list(fields.items())[:30]
        )[:MAX_CONTEXT_CHARS]

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Briefly explain each form field and note anything significant."),
            ("human", "Form: {form_name}\n\n{fields}\n\nExplain each field."),
        ])

        chain = prompt | self._llm
        response = chain.invoke({"form_name": form_name, "fields": fields_str})
        return response.content

    def reset(self):
        """Clear all loaded forms and vector store."""
        self._loaded_forms.clear()
        self._vectorstore.clear()
