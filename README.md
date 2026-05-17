# 🧠 Intelligent Form Agent

An AI-powered agent that can process, understand, and explain a wide variety of forms. It automatically extracts information from both structured and unstructured fields, answers questions about individual forms, generates holistic insights across multiple forms, and produces concise summaries.

## Features

- **Multi-format Support**: PDF (text + scanned), images (OCR), DOCX, TXT, CSV
- **Question Answering**: Ask natural language questions about any loaded form
- **Summarization**: Generate concise summaries highlighting key information
- **Holistic Analysis**: Cross-form insights, pattern detection, and comparison
- **Field Extraction**: Automatic key-value pair detection with explanations
- **Web UI**: Streamlit-based interface for easy interaction
- **CLI**: Interactive command-line interface for power users

## Architecture

```
User Interface (Streamlit / CLI)
        │
    Form Agent (QA, Summary, Analysis)
        │
   ┌────┼────┐
   │    │    │
Extractors  VectorStore  LLM
(PDF/OCR/   (ChromaDB)   (OpenAI)
 DOCX/TXT)
```

See [docs/architecture.md](docs/architecture.md) for detailed design documentation.

## Quick Start

### 1. Prerequisites

- Python 3.10+
- Groq API key
- (Optional) Tesseract OCR for image/scanned PDF support

### 2. Setup

```bash
# Clone the repository
git clone <repo-url>
cd "Autonimize AI Assignment"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 3. Run the Agent

**Option A: Web UI (Recommended)**
```bash
streamlit run run_app.py
```
Then open http://localhost:8501 in your browser.

**Option B: Command Line**
```bash
python run_cli.py
```

**Option C: Demo Script**
```bash
python -m notebooks.demo
```

## Usage Examples

### CLI Commands

```bash
# Load a form
🤖 > load data/sample_employment_form.txt

# Ask a question about a specific form
🤖 > ask:sample_employment_form.txt What is the applicant's desired salary?

# Ask across all forms
🤖 > ask What names appear in the loaded forms?

# Generate a summary
🤖 > summary:sample_tax_form.txt

# Holistic analysis across forms
🤖 > holistic What types of personal information are collected across all forms?

# Explain extracted fields
🤖 > fields sample_medical_form.txt

# List loaded forms
🤖 > forms
```

### Python API

```python
from src.agent import FormAgent

agent = FormAgent()

# Load forms
agent.load_form("data/sample_employment_form.txt")
agent.load_form("data/sample_tax_form.txt")

# Ask questions
answer = agent.ask("What is the applicant's name?", form_name="sample_employment_form.txt")
print(answer)

# Summarize
summary = agent.summarize(form_name="sample_tax_form.txt")
print(summary)

# Holistic analysis
analysis = agent.holistic_analysis("Compare the personal data collected in each form")
print(analysis)
```

## Example Runs

### Example 1: Single Form QA

**Input**: `ask:sample_employment_form.txt What is the applicant's desired salary and current salary?`

**Expected Output**:
> The applicant's current salary is $145,000 at DataFlow Inc., and their desired salary for the new position at TechCorp Industries is $175,000, representing a $30,000 increase.

### Example 2: Form Summary

**Input**: `summary:sample_tax_form.txt`

**Expected Output**:
> **Form 1040 - U.S. Individual Income Tax Return (2023)**
>
> This is a joint tax return filed by John A. Smith and Jane Smith from Springfield, IL. Key highlights:
> - Total Income: $106,850 (wages $85K, business income $12K, capital gains $5.2K, dividends $3.4K, interest $1.25K)
> - Adjusted Gross Income: $94,000 after $12,850 in adjustments (IRA, student loan, HSA)
> - Standard deduction of $27,700 applied
> - Tax credits of $6,000 (child tax credit $4K for 2 children, education $2K)
> - **Refund: $8,044** (withheld $9,500 vs. tax owed $1,456)

### Example 3: Holistic Analysis

**Input**: `holistic What personal information is common across all three forms?`

**Expected Output**:
> All three forms collect core personal identifiers: full name, address, and contact information. However, the depth varies:
>
> - **Employment Form** (Sarah Johnson): Name, email, phone, address, DOB — focused on professional qualifications
> - **Tax Form** (John Smith): Name, SSN, address, filing status — focused on financial identity
> - **Medical Form** (Robert Williams): Name, DOB, age, gender, address, phone, email, emergency contact — most comprehensive personal data
>
> Common fields across all: Name, Address. The medical form collects the most sensitive personal data including health history and emergency contacts.

## Project Structure

```
├── src/
│   ├── __init__.py
│   ├── agent.py          # Core agent (QA, summarization, analysis)
│   ├── app.py            # Streamlit web UI
│   ├── cli.py            # Interactive CLI
│   ├── config.py         # Configuration management
│   ├── extractors.py     # Document text extraction
│   └── vectorstore.py    # ChromaDB vector store management
├── data/
│   ├── sample_employment_form.txt
│   ├── sample_tax_form.txt
│   └── sample_medical_form.txt
├── notebooks/
│   └── demo.py           # Demonstration script
├── tests/
│   ├── test_extractors.py
│   └── test_agent.py
├── docs/
│   └── architecture.md   # Design documentation
├── requirements.txt
├── .env.example
├── run_cli.py            # CLI entry point
├── run_app.py            # Streamlit entry point
└── README.md
```

## Running Tests

```bash
# Run all tests (unit tests don't require API key)
pytest tests/ -v

# Run only unit tests
pytest tests/ -v -k "not Integration"

# Run integration tests (requires OPENAI_API_KEY)
pytest tests/ -v -k "Integration"
```

## Configuration

Environment variables (set in `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | (required) | Your Groq API key |
| `MODEL_NAME` | `deepseek-r1-distill-llama-70b` | LLM model for QA/summarization |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model |
| `CHUNK_SIZE` | `1000` | Text chunk size for indexing |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TEMPERATURE` | `0.6` | LLM temperature |

## Creativity Extensions

### 1. Structured Field Extraction
Beyond raw text, the agent uses heuristic pattern matching to identify form fields (key:value pairs, checkboxes, underline-separated fields) and can explain their significance.

### 2. Multi-Modal Input
Supports scanned PDFs and images via OCR fallback — if a PDF page has minimal text, it automatically renders the page as an image and applies Tesseract OCR.

### 3. Streamlit Web UI
Full-featured web interface with:
- Drag-and-drop file upload
- Tabbed interface for different capabilities
- Session history tracking
- Real-time processing feedback

### 4. Holistic Cross-Form Analysis
Goes beyond single-form QA to provide comparative analysis, pattern detection, and insights across document collections.

## Tech Stack

- **LLM**: DeepSeek R1 Distill Llama 70B (via Groq API)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2, runs locally)
- **Vector Store**: ChromaDB
- **Framework**: LangChain
- **Document Processing**: PyMuPDF, pytesseract, python-docx
- **UI**: Streamlit
- **Testing**: pytest

## License

MIT
