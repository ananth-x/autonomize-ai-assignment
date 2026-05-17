# Architecture & Design

## System Overview

The Intelligent Form Agent is built as a modular pipeline with four main components:

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
│  (PDF, OCR,  │  │  (ChromaDB +   │  │  (OpenAI GPT  │
│   DOCX, TXT) │  │   Embeddings)  │  │   via Chain)   │
└─────────────┘  └───────────────┘  └───────────────┘
```

## Component Details

### 1. Extractors (`src/extractors.py`)

Responsible for converting raw form files into text:

- **PDF Extraction**: Uses PyMuPDF for text-based PDFs, with OCR fallback for scanned documents
- **Image OCR**: Pytesseract for PNG/JPG/TIFF form images
- **DOCX**: python-docx for Word documents (paragraphs + tables)
- **Plain Text/CSV**: Direct file reading

Additionally, `StructuredFieldExtractor` uses heuristic pattern matching to identify key-value pairs from form text (colon-separated, underline-separated, checkboxes).

### 2. Vector Store (`src/vectorstore.py`)

Manages document embeddings for semantic retrieval:

- Documents are chunked using `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap)
- Embeddings generated locally via `sentence-transformers` (all-MiniLM-L6-v2) — no API calls needed
- Stored in ChromaDB with metadata (source filename, extension, pages)
- Enables semantic search for relevant context retrieval

### 3. Form Agent (`src/agent.py`)

The core intelligence layer with four main capabilities:

| Capability | Method | Description |
|-----------|--------|-------------|
| QA | `ask()` | Answer questions about specific or all forms |
| Summarize | `summarize()` | Generate concise form summaries |
| Holistic | `holistic_analysis()` | Cross-form pattern analysis |
| Explain | `explain_fields()` | Explain extracted field meanings |

Each capability uses a tailored prompt template and retrieves relevant context from the vector store before querying the LLM.

### 4. User Interfaces

- **CLI** (`src/cli.py`): Interactive REPL with command parsing
- **Streamlit** (`src/app.py`): Web UI with file upload, tabs for each capability
- **Python API**: Direct use of `FormAgent` class

## Design Decisions

1. **RAG over fine-tuning**: Using retrieval-augmented generation allows the agent to work with any form without training, and provides traceable answers grounded in source text.

2. **Chunked indexing**: Forms are split into overlapping chunks to handle long documents while maintaining context at chunk boundaries.

3. **Dual extraction**: Both raw text and structured fields are extracted — raw text for semantic search, structured fields for precise key-value access.

4. **Modular architecture**: Each component is independently testable and replaceable (e.g., swap ChromaDB for Pinecone, or OpenAI for a local model).

5. **Graceful degradation**: OCR is optional (falls back to text extraction), and the agent works with any text-based form even without pytesseract installed.
