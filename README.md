# 🧠 Intelligent Form Agent

An AI-powered agent that processes, understands, and explains forms. It extracts information from structured and unstructured fields, answers natural language questions, generates summaries, and provides holistic insights across multiple documents.

## Features

- **Multi-format Support**: PDF (text + scanned/OCR), images, DOCX, TXT, CSV
- **Question Answering**: Ask natural language questions about any loaded form
- **Summarization**: Generate detailed summaries highlighting key information
- **Holistic Analysis**: Cross-form insights, pattern detection, and comparison
- **Field Extraction**: Automatic key-value pair detection with explanations
- **Web UI**: Streamlit-based interface with file upload and tabbed navigation
- **CLI**: Interactive command-line interface

## Architecture

```
User Interface (Streamlit / CLI)
        │
    Form Agent (QA, Summary, Analysis)
        │
   ┌────┼────┐
   │    │    │
Extractors  VectorStore     LLM
(PDF/OCR/   (ChromaDB +     (Llama 3.1 8B
 DOCX/TXT)  BGE embeddings)  via Groq)
```

See [docs/architecture.md](docs/architecture.md) for detailed design documentation.
See [docs/screenshots.pdf](docs/screenshots.pdf) for demonstration screenshots.

## Tech Stack

- **LLM**: Llama 3.1 8B Instant (via Groq API — free tier)
- **Embeddings**: BAAI/bge-small-en-v1.5 (runs locally, no API cost)
- **Vector Store**: ChromaDB (persistent, local)
- **Framework**: LangChain
- **Document Processing**: PyMuPDF, pytesseract, python-docx
- **UI**: Streamlit
- **Testing**: pytest

## Quick Start

### Prerequisites

- Python 3.9+
- Groq API key (free at https://console.groq.com)
- (Optional) Tesseract OCR for scanned PDF/image support

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd "Autonimize AI Assignment"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Run

**Web UI (Recommended):**
```bash
streamlit run run_app.py
```
Open http://localhost:8501 in your browser.

**Command Line:**
```bash
python run_cli.py
```

**Demo Script:**
```bash
python -m notebooks.demo
```

## Usage Examples

### CLI Commands

```bash
load data/sample_employment_form.txt    # Load a form
ask What is Sarah's desired salary?     # Ask a question
ask:sample_tax_form.txt What is the refund amount?  # Ask about specific form
summary:sample_medical_form.txt         # Summarize a form
holistic What personal info is common across forms?  # Cross-form analysis
fields sample_employment_form.txt       # Explain extracted fields
forms                                   # List loaded forms
reset                                   # Clear all
```

### Python API

```python
from src.agent import FormAgent

agent = FormAgent()
agent.load_form("data/sample_employment_form.txt")
agent.load_form("data/sample_tax_form.txt")

answer = agent.ask("What is Sarah's previous employer?")
summary = agent.summarize(form_name="sample_tax_form.txt")
analysis = agent.holistic_analysis("Compare salaries across forms")
```

## Demonstration

### Example 1: Single Form QA

**Q:** "What is Sarah's desired salary and current salary?"
**A:** Sarah's current salary is $145,000 at DataFlow Inc. Her desired salary for the TechCorp position is $175,000 base ($220,000 total compensation).

### Example 2: Form Summary

**Q:** Summarize the tax form
**A:** The tax form contains 3 returns: John & Jane Smith (MFJ, AGI $158,200, refund $5,507), Patricia Anderson (Single, AGI $181,868, refund $153), and Michael & Elizabeth Thompson (MFJ, AGI $320,700, refund $2,766). All prepared by Robert Thompson, CPA.

### Example 3: Holistic Analysis

**Q:** "What health conditions appear across all patients?"
**A:** All three patients have hypertension. Robert Williams and James Martinez both have hyperlipidemia. Martinez has the most complex profile with diabetes, obesity, and cardiovascular risk factors. Sarah Chen's conditions are limited to migraines and mild anxiety.

See [docs/screenshots.pdf](docs/screenshots.pdf) for visual demonstration of the UI.

## Project Structure

```
├── src/
│   ├── __init__.py          # Package init
│   ├── agent.py             # Core agent (QA, summarization, analysis)
│   ├── app.py               # Streamlit web UI
│   ├── cli.py               # Interactive CLI
│   ├── config.py            # Configuration management
│   ├── extractors.py        # Document text extraction + field detection
│   └── vectorstore.py       # Adaptive chunking + ChromaDB management
├── data/
│   ├── sample_employment_form.txt   # 3 job applicants
│   ├── sample_tax_form.txt          # 3 tax returns
│   └── sample_medical_form.txt      # 3 patient intakes
├── notebooks/
│   └── demo.py              # Demonstration script
├── tests/
│   ├── test_extractors.py   # Extractor unit tests
│   └── test_agent.py        # Agent unit + integration tests
├── docs/
│   ├── architecture.md      # Design documentation
│   └── screenshots.pdf      # UI demonstration screenshots
├── requirements.txt
├── .env.example
├── run_cli.py               # CLI entry point
├── run_app.py               # Streamlit entry point
└── README.md
```

## Running Tests

```bash
# Unit tests (no API key needed)
pytest tests/ -v -k "not Integration"

# Integration tests (requires GROQ_API_KEY)
pytest tests/ -v -k "Integration"
```

## Configuration

Environment variables (set in `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | (required) | Your Groq API key |
| `MODEL_NAME` | `llama-3.1-8b-instant` | LLM model |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local embedding model |
| `CHUNK_SIZE` | `800` | Text chunk size for indexing |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `TEMPERATURE` | `0.6` | LLM temperature |

## Design Highlights

### Adaptive Chunking
The splitter detects document structure dynamically — record boundaries (any repeated separator pattern), page markers, paragraph breaks — without hardcoding format-specific rules. Works with PDFs, OCR output, structured forms, and free text.

### Token-Efficient Pipeline
Designed for Groq's free tier (6K TPM). Compact prompts, retrieval-based context (only relevant chunks sent to LLM), and local embeddings keep API costs at zero.

### Retrieval-Augmented Generation (RAG)
Forms are chunked and embedded locally. Questions retrieve only the most relevant chunks via semantic search with deduplication, enabling accurate answers even from large multi-person documents.

### Multi-Person Form Handling
Each chunk gets a context identifier (person name, record number) prepended automatically, so the LLM always knows who the data belongs to — even when forms contain multiple people.

## Creativity Extensions

1. **Adaptive Document Chunking** — Detects structure dynamically rather than relying on fixed delimiters
2. **Streamlit Web UI** — Drag-and-drop upload, tabbed interface, session history
3. **Holistic Cross-Form Analysis** — Comparative analysis across document collections
4. **Structured Field Extraction** — Heuristic pattern matching for key:value, checkboxes, underline fields
5. **Multi-Format OCR Fallback** — Scanned PDFs automatically fall back to OCR when text extraction yields minimal content

## License

MIT
