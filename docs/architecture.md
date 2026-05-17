# Architecture & Design

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                       │
│              (Streamlit UI / CLI / Python API)                │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                     Form Agent Core                           │
│         (Orchestration, QA, Summarization, Analysis)         │
└──────┬──────────────────┬───────────────────┬───────────────┘
       │                  │                   │
┌──────▼──────┐  ┌───────▼───────┐  ┌───────▼───────┐
│  Extractors  │  │  Vector Store  │  │   LLM Layer   │
│  (PDF, OCR,  │  │  (ChromaDB +   │  │  (Llama 3.1   │
│   DOCX, TXT) │  │  BGE embed.)   │  │   via Groq)   │
└─────────────┘  └───────────────┘  └───────────────┘
```

## Component Details

### 1. Extractors (`src/extractors.py`)

Converts raw files into text:
- **PDF**: PyMuPDF for text-based PDFs, OCR fallback for scanned pages
- **Images**: Pytesseract OCR for PNG/JPG/TIFF
- **DOCX**: python-docx (paragraphs + tables)
- **Plain Text/CSV**: Direct file reading

`StructuredFieldExtractor` uses heuristic pattern matching to identify key-value pairs (colon-separated, underline-separated, checkboxes).

### 2. Vector Store (`src/vectorstore.py`)

**Adaptive Chunking Strategy:**
- Detects record boundaries dynamically (any repeated separator: ===, ---, etc.)
- Falls back to paragraph-based splitting for unstructured text
- Prepends context identifiers (person name, record number) to each chunk
- No hardcoded format assumptions — works with any document type

**Retrieval:**
- Embeddings via BAAI/bge-small-en-v1.5 (local, normalized)
- ChromaDB for persistent vector storage
- Deduplication: fetches 2x results, removes >70% overlapping chunks

### 3. Form Agent (`src/agent.py`)

| Capability | Method | Strategy |
|-----------|--------|----------|
| QA | `ask()` | Retrieve 4 relevant chunks → LLM answers |
| Summarize | `summarize()` | Retrieve broad chunks per form → LLM summarizes |
| Holistic | `holistic_analysis()` | Field overview + retrieval → LLM compares |
| Explain | `explain_fields()` | Send extracted fields → LLM explains |

**Token Management:**
- Context budget: ~5000 chars per request
- Compact system prompts (~20 tokens each)
- No response length cap (model generates full answers)
- Designed for Groq free tier (6K TPM)

### 4. User Interfaces

- **Streamlit** (`src/app.py`): File upload, tabbed interface, session history
- **CLI** (`src/cli.py`): Interactive REPL with command parsing
- **Python API**: Direct `FormAgent` class usage

## Design Decisions

1. **RAG over fine-tuning**: Works with any form without training. Answers are grounded in source text.

2. **Local embeddings**: BAAI/bge-small-en-v1.5 runs on CPU with no API cost. Slightly better retrieval accuracy than MiniLM.

3. **Adaptive chunking**: Detects structure dynamically rather than hardcoding delimiters. Handles PDFs, OCR output, structured forms, and free text equally.

4. **Free-tier optimized**: Groq provides fast inference at no cost. Token-efficient prompts and retrieval-based context keep usage within limits.

5. **Modular architecture**: Each component is independently replaceable (swap ChromaDB for Pinecone, Groq for OpenAI, etc.).
