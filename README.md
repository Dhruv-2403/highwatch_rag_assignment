# Highwatch RAG — Personal ChatGPT over Google Drive

A production-ready **Retrieval-Augmented Generation (RAG)** system that connects to your Google Drive, processes your documents, and answers questions grounded strictly in your own content — no hallucination, full source attribution.

---

## What it does

Upload your company policy docs, SOPs, handbooks, or any PDF/Docs/TXT files to Google Drive. Then ask questions like:

- *"What is our refund policy?"*
- *"What are our compliance rules around gifts?"*
- *"What happens on an employee's first day?"*

The system finds the most relevant passages from your documents and generates a precise, cited answer using an LLM.

---

## Architecture

```
highwatch-rag/
├── connectors/          # Google Drive integration
│   └── gdrive.py        # OAuth 2.0 + Service Account auth, file fetching
├── processing/          # Document pipeline
│   └── pipeline.py      # Extract → clean → chunk (paragraph-aware sliding window)
├── embedding/           # Vector encoding
│   └── encoder.py       # SentenceTransformers, batch processing, L2 normalisation
├── search/              # Vector storage
│   └── store.py         # FAISS IndexFlatIP + JSON metadata, persistent to disk
├── api/                 # Web API
│   ├── app.py           # FastAPI routes: /sync-drive, /ask, /status, /health
│   └── llm.py           # OpenAI-compatible LLM with grounded prompt
├── tests/               # Automated tests
│   ├── test_processing.py
│   └── test_store.py
├── scripts/
│   └── sample_queries.py  # Demo without Google credentials
├── .github/
│   └── workflows/
│       └── ci.yml         # GitHub Actions CI pipeline
├── main.py              # Entry point
├── config.py            # Centralised settings (pydantic-settings)
├── startup.py           # Decodes service account from env on server startup
└── requirements.txt
```

### Data flow

```
─── SYNC (/sync-drive) ────────────────────────────────────────────────

  Google Drive
       │  PDF · Google Docs · TXT
       ▼
  connectors/gdrive.py       ← OAuth 2.0 or Service Account
       │  raw bytes + metadata
       ▼
  processing/pipeline.py     ← extract → clean → chunk
       │  list[Chunk]
       ▼
  embedding/encoder.py       ← SentenceTransformers, batch, L2-norm
       │  ndarray float32
       ▼
  search/store.py            ← FAISS IndexFlatIP + metadata.json (disk)

─── QUERY (/ask) ───────────────────────────────────────────────────────

  User question
       │
       ▼
  embedding/encoder.py       ← encode query to vector
       │
       ▼
  search/store.py            ← cosine similarity → top-K chunks
       │
       ▼
  api/llm.py                 ← grounded prompt → LLM
       │
       ▼
  { "answer": "...", "sources": ["doc1.pdf", "policy.docx"] }
```

---

## Tech stack

| Component | Technology |
|---|---|
| API framework | FastAPI + Uvicorn |
| Google Drive | google-api-python-client |
| Document parsing | PyPDF2, python-docx |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
| Vector store | FAISS (IndexFlatIP) |
| LLM | OpenAI-compatible (gpt-4o-mini default) |
| Config | pydantic-settings |
| Tests | pytest + pytest-cov |
| CI | GitHub Actions |

---

## Setup

### Prerequisites

- Python 3.11+
- Google Cloud account (free)
- OpenAI API key

### 1. Clone the repository

```bash
git clone https://github.com/Dhruv-2403/highwatch_rag_assignment
cd highwatch-rag
```

### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        
pip install -r requirements.txt
```

### 3. Google Drive credentials

**Option A — OAuth 2.0 (local development)**

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Create a project → Enable **Google Drive API**
3. Credentials → Create → **OAuth 2.0 Client ID** → Desktop App → Download JSON
4. Save it:

```bash
mkdir credentials
mv ~/Downloads/client_secret_*.json credentials/oauth_client_secret.json
```

**Option B — Service Account (production / server)**

1. IAM & Admin → Service Accounts → Create Service Account
2. Keys tab → Add Key → Create new key → JSON → Download
3. Share your Drive folder with the service account email (Viewer access)
4. Save the JSON to `credentials/service_account.json`

### 4. Configure environment


Edit `.env`:

```env

GOOGLE_AUTH_METHOD=oauth                          # or service_account
GOOGLE_OAUTH_CLIENT_SECRET_FILE=credentials/oauth_client_secret.json

# LLM
OPENAI_API_KEY=sk-your-key-here
LLM_MODEL=gpt-4o-mini
```

### 5. Run the server

```bash
python main.py
```


---

## API reference

### `POST /sync-drive`

Fetches files from Google Drive, processes, embeds, and stores them.

**Request:**
```json
{
  "folder_ids": ["1abc...", "2def..."]
}
```
Omit `folder_ids` to sync the entire Drive.

**Response:**
```json
{
  "status": "completed",
  "message": "Sync finished in 23.4s",
  "documents_processed": 8,
  "chunks_stored": 112,
  "errors": [],
  "duration_seconds": 23.4
}
```

---

### `POST /ask`

Answers a question using RAG over synced documents.

**Request:**
```json
{
  "query": "What is our refund policy?",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "Customers are eligible for a full refund within 30 days of purchase if the product is unused and in its original packaging...",
  "sources": ["refund_policy.pdf", "company_handbook.docx"],
  "source_details": [
    {
      "file_name": "refund_policy.pdf",
      "doc_id": "1abc...",
      "relevance_score": 0.9124,
      "web_view_link": "https://docs.google.com/...",
      "chunk_preview": "Customers are eligible for a full refund within 30 days..."
    }
  ],
  "query": "What is our refund policy?"
}
```

---

### `GET /status`

Returns index statistics and last sync time.

```json
{
  "total_chunks": 112,
  "unique_documents": 8,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "llm_model": "gpt-4o-mini",
  "sync_status": "completed",
  "last_sync": "2026-04-25T18:30:00Z"
}
```

---

### `GET /health`

Liveness check.

```json
{ "status": "ok" }
```

---

## Sample queries and outputs

Run the demo without any Google credentials:

```bash
python scripts/sample_queries.py
```

This creates three synthetic policy documents and runs questions against them.

**Sample output:**

```
Q: What is our refund policy?
A: Customers are eligible for a full refund within 30 days of purchase if
   the product is unused and in its original packaging. After 30 days,
   refunds are only available for defective products.
Sources: ['refund_policy.txt']

Q: What are our company policies on compliance?
A: The compliance policy covers four areas: anti-bribery (gifts capped at
   $50, exceptions require Legal approval), data privacy (GDPR compliance,
   no third-party sharing without consent), conflict of interest disclosure
   within 30 days, and incident reporting via compliance@acme.com or the
   anonymous whistleblower hotline.
Sources: ['compliance_policy.txt']

Q: What happens on an employee's first day?
A: On day 1, HR sends a welcome email with system credentials by 8 AM,
   IT provisions the laptop and access to Slack, Jira, and Confluence,
   and the manager schedules an introductory 1:1 meeting.
Sources: ['onboarding_sop.txt']

Q: Who do I report a compliance violation to?
A: Suspected violations should be reported to compliance@acme.com or
   through the anonymous whistleblower hotline at 1-800-ACM-SAFE.
Sources: ['compliance_policy.txt']
```

---

## Chunking strategy

The pipeline uses **paragraph-aware sliding-window chunking** instead of naive fixed-width splitting:

1. Text is split on blank lines to respect semantic paragraph boundaries.
2. Paragraphs are accumulated into a window until `CHUNK_SIZE` tokens (~512) is reached.
3. Each new chunk carries `CHUNK_OVERLAP` tokens (64) from the tail of the previous chunk — preserving cross-boundary context.
4. Paragraphs individually exceeding the size limit are hard-split by character with the same overlap logic.

This ensures semantic units stay together and retrieval boundaries don't silently drop context.

---

```bash
pip install pytest pytest-cov
pytest tests/ -v
```

**With coverage report:**

```bash
pytest tests/ --cov=. --cov-report=term-missing
```

Tests cover:
- Text cleaning (unicode normalisation, control characters, whitespace)
- Chunking (overlap, hard-split, empty input, single chunk)
- Vector store (upsert, deduplication, removal, search, stats)

---
### Production note

The FAISS index is stored on the local filesystem. For persistent deployments, mount a persistent volume or replace FAISS with a hosted vector database (Pinecone, Weaviate, or Qdrant free tiers). On free-tier platforms (Render, Railway), call `POST /sync-drive` once after each deploy.

---

## Design decisions

| Decision | Rationale |
|---|---|
| FAISS `IndexFlatIP` | Exact cosine search, no approximation error, no external service needed |
| L2-normalised embeddings | Inner product becomes cosine similarity — simpler index, same quality |
| JSON metadata sidecar | Parallel array to FAISS rows — zero-dependency, human-readable, inspectable |
| Upsert = delete + re-add | FAISS flat index has no in-place deletion; rebuilding kept rows is fast for typical corpus sizes |
| Paragraph-aware chunking | Semantic units stay whole — significantly better retrieval quality vs fixed-width splits |
| OpenAI-compatible client | Works with OpenAI, Azure OpenAI, Ollama, LM Studio, Groq — swap via `OPENAI_BASE_URL` |
| Strict grounding prompt | LLM answers only from context — prevents hallucination, forces source attribution |
| Pydantic settings | All config in one place, type-validated, loaded from `.env` — no hardcoded values |

---

## Project structure — assignment mapping

| Assignment requirement | Implementation |
|---|---|
| `connectors/` → Google Drive | `connectors/gdrive.py` |
| `processing/` → parsing and chunking | `processing/pipeline.py` |
| `embedding/` | `embedding/encoder.py` |
| `search/` | `search/store.py` |
| `api/` | `api/app.py`, `api/llm.py` |
| `POST /sync-drive` | `api/app.py` → `_run_sync()` |
| `POST /ask` | `api/app.py` → `ask()` |
| Answer with sources | `AskResponse.sources` + `AskResponse.source_details` |
| Good chunking strategy | Paragraph-aware sliding window with overlap |
| Clean API design | Pydantic models, OpenAPI docs at `/docs` |
| Incremental sync | Upsert in `store.py` — re-syncing a file removes old chunks first |

---

