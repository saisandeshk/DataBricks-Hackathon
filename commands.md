# Nyaya Sahayak — Deployment Guide

## Architecture

```
                        ┌─────────────────────────┐
   Legal PDFs ────────→ │ pymupdf4llm (PDF → MD)  │
   BNS CSV ───────────→ │ Ingestion Notebook       │ → Delta Table → FAISS + Parent Store
   IndicLegalQA JSON ─→ │ MarkdownHeaderSplitter   │
                        └─────────────────────────┘

User → Gradio UI (voice + text + 13 languages)
        ↓
    LangGraph Agent (query rewrite → orchestrator → tools → compress → aggregate)
        ↓
    FAISS Index (child chunks) + Parent Store (full sections)
        ↓
    Sarvam AI (sarvam-m model, OpenAI-compatible)
```

**Data flow:**
1. User asks a legal question (text or voice, any of 13 Indian languages)
2. Sarvam AI translates to English (if needed) / transcribes voice
3. LangGraph agent rewrites query, searches FAISS, retrieves parent context
4. Agent may compress context and iterate (up to 8 iterations)
5. Final answer aggregated, translated back, optionally read aloud via TTS

---

## Prerequisites

- Databricks workspace (Free Edition works)
- Python 3.10+
- Sarvam AI API key (free tier at https://sarvam.ai — used for LLM + voice + translation)

---

## Step 1: Create Databricks Workspace

1. Go to https://www.databricks.com/try-databricks
2. Sign up for Community/Free Edition
3. Note your workspace URL (e.g., `https://adb-1234567890.12.azuredatabricks.net`)

---

## Step 2: Set Up Unity Catalog

Run in a Databricks SQL editor or notebook:

```sql
-- Create schema and volume
CREATE DATABASE IF NOT EXISTS workspace.default;
CREATE VOLUME IF NOT EXISTS workspace.default.bharat_bricks_hacks;
```

---

## Step 3: Upload Datasets

**Required:**
1. Upload `datasets/BNS/bns_sections.csv` to the Volume
   - In Databricks: **Catalog → workspace → default → bharat_bricks_hacks**
   - Click **Upload to Volume** → upload `bns_sections.csv`

**Legal PDFs (Constitution, SC Judgments, etc.):**
2. Create a `pdfs/` subfolder in the Volume
3. Upload any legal PDF files to `pdfs/`:
   - Constitution of India PDF
   - Supreme Court judgment compilations
   - BNS gazette PDF
   - Any other legal documents
   - The ingestion notebook will automatically convert them to Markdown and chunk them

**Recommended (for richer corpus):**
4. Upload `datasets/IndicLegalQA Dataset_10K_Revised.json` to the same Volume
   - This adds ~10,000 Supreme Court QA pairs to the corpus

---

## Step 4: Create Secrets

```bash
# In Databricks CLI (or notebook with %sh)
databricks secrets create-scope nyaya-sahayak

# Sarvam API key (used for LLM, voice, translation — all in one key)
databricks secrets put-secret nyaya-sahayak sarvam-api-key
```

**Alternative (environment variables in notebook):**
```python
import os
os.environ["SARVAM_API_KEY"] = dbutils.secrets.get("nyaya-sahayak", "sarvam-api-key")
```

---

## Step 5: Run Ingestion Notebook

1. Import `notebooks/ingest_corpus.py` into your Databricks workspace
2. Attach to a cluster (any size — single node works)
3. Update `CATALOG`, `SCHEMA`, `VOLUME` if different from defaults
4. Run all cells

**Expected output:**
```
BNS: 358 sections
Found N PDF(s) in /Volumes/.../pdfs
📄 Processing: constitution_of_india.pdf → X parent chunks, Y child chunks
📄 Processing: sc_judgments.pdf → X parent chunks, Y child chunks
IndicLegalQA: ~X case summaries + ~10000 individual QA chunks
Total corpus: 10000+ chunks
✅ Saved to workspace.default.legal_rag_corpus
```

---

## Step 6: Build FAISS Index

1. Import `notebooks/build_index.py` into your Databricks workspace
2. Attach to a cluster with ≥4GB RAM
3. Run all cells

**Expected output:**
```
Parents: 358+
Children: 1000+ (depends on chunk splitting)
✅ FAISS index saved to /Volumes/workspace/default/bharat_bricks_hacks/nyaya_index/
✅ Parent store saved to /Volumes/workspace/default/bharat_bricks_hacks/nyaya_index/
```

---

## Step 7: Deploy as Databricks App

### Option A: Databricks Apps (recommended)

1. Upload the entire `project/` folder to your workspace:
   ```
   /Workspace/Users/<your-email>/nyaya-sahayak/
   ```

2. In Databricks: **Compute → Apps → Create App**
3. Set:
   - **Name:** `nyaya-sahayak`
   - **Source:** `/Workspace/Users/<your-email>/nyaya-sahayak/`
   - **Config:** `app.yaml`

4. The app will install dependencies from `requirements.txt` and start

5. Environment variables from `app.yaml`:
   - `SARVAM_API_KEY` — from secret scope (used for LLM + voice + translation)
   - `LLM_OPENAI_BASE_URL` — `https://api.sarvam.ai/v1`
   - `LLM_MODEL` — `sarvam-m`
   - `FAISS_INDEX_DIR` — Volume path for FAISS index
   - `PARENT_STORE_DIR` — Volume path for parent chunks

### Option B: Run in Notebook

```python
# In a Databricks notebook cell:
import os, sys

os.environ["SARVAM_API_KEY"] = dbutils.secrets.get("nyaya-sahayak", "sarvam-api-key")
os.environ["LLM_OPENAI_BASE_URL"] = "https://api.sarvam.ai/v1"
os.environ["LLM_MODEL"] = "sarvam-m"
os.environ["FAISS_INDEX_DIR"] = "/Volumes/workspace/default/bharat_bricks_hacks/nyaya_index"
os.environ["PARENT_STORE_DIR"] = "/Volumes/workspace/default/bharat_bricks_hacks/nyaya_index"

# Add project to path
REPO_ROOT = "/Workspace/Users/<your-email>/nyaya-sahayak"
sys.path.insert(0, REPO_ROOT)

from src.ui.gradio_app import create_gradio_ui
demo = create_gradio_ui()
demo.launch(share=True)
```

### Option C: Run Locally

```bash
cd project/

# Option 1: Edit .env file (recommended)
cp .env .env.local
# Edit .env.local — set SARVAM_API_KEY and paths

# Option 2: Export vars manually
export SARVAM_API_KEY="your-sarvam-key"
export LLM_OPENAI_BASE_URL="https://api.sarvam.ai/v1"
export LLM_MODEL="sarvam-m"
export FAISS_INDEX_DIR="./faiss_index"      # copy from Volume
export PARENT_STORE_DIR="./parent_store"    # copy from Volume

pip install -r requirements.txt
python app/main.py
```

---

## Step 8: Verify

1. Open the app URL (shown after deployment)
2. Try these test queries:
   - "What is the punishment for theft under BNS?"
   - "Explain Article 21 of the Constitution"
   - "What did the Supreme Court say about right to privacy?"
3. Test voice: click the microphone, speak in Hindi or English
4. Test translation: switch language dropdown to Hindi, ask in English

---

## Project Structure

```
project/
├── app.yaml                 # Databricks Apps config
├── requirements.txt         # Python dependencies
├── .env                     # Local env vars (edit with your keys)
├── commands.md              # This file
├── app/
│   ├── __init__.py
│   └── main.py              # Entry point (loads .env)
├── src/
│   ├── __init__.py
│   ├── config.py            # All configuration constants
│   ├── llm_client.py        # Sarvam AI / OpenAI-compatible wrapper
│   ├── sarvam_client.py     # Sarvam STT/TTS/Translation
│   ├── document_processor.py # PDF → Markdown → Parent/Child chunks
│   ├── embedder.py          # SentenceTransformer wrapper
│   ├── query_logger.py      # Delta Lake / CSV logging
│   ├── utils.py             # Token estimation utilities
│   ├── db/
│   │   ├── faiss_manager.py # FAISS vector store (LangChain)
│   │   └── parent_store.py  # Parent chunk JSON store
│   ├── rag_agent/
│   │   ├── graph.py         # LangGraph main + agent subgraph
│   │   ├── graph_state.py   # State definitions
│   │   ├── nodes.py         # All graph node implementations
│   │   ├── edges.py         # Routing logic
│   │   ├── prompts.py       # Legal-domain prompt templates
│   │   ├── schemas.py       # Pydantic models (QueryAnalysis)
│   │   └── tools.py         # search_child_chunks + retrieve_parent_chunks
│   ├── core/
│   │   ├── rag_system.py    # System initialization
│   │   └── chat_interface.py # Streaming chat handler
│   └── ui/
│       └── gradio_app.py    # Full Gradio UI with voice
└── notebooks/
    ├── ingest_corpus.py     # BNS CSV + Legal PDFs + IndicLegalQA → Delta table
    └── build_index.py       # Delta → FAISS index + parent store
```

---

## Agent Architecture

The system uses a **two-level LangGraph** agent:

### Main Graph
```
START → summarize_history → rewrite_query → [clarification | agent(s)] → aggregate_answers → END
```

- **summarize_history**: Compresses conversation context (last 6 exchanges)
- **rewrite_query**: Rewrites query for optimal legal document retrieval; splits multi-part queries
- **request_clarification**: Asks user for details if query is ambiguous (interrupt_before)
- **agent**: Dispatches to agent subgraph (parallel if multiple sub-queries)
- **aggregate_answers**: Combines sub-answers into a single coherent legal response

### Agent Subgraph (per sub-query)
```
START → orchestrator → [tools | fallback_response | collect_answer]
         ↑                ↓
         ← should_compress_context ← tools
         ← compress_context ←
```

- **orchestrator**: Decides what to search/retrieve next using the LLM with tools
- **tools**: Executes search_child_chunks (FAISS) or retrieve_parent_chunks (JSON store)
- **should_compress_context**: Checks if context exceeds token threshold
- **compress_context**: Summarizes accumulated context to fit token budget
- **fallback_response**: Generates best answer from available context when iteration limit reached
- **collect_answer**: Captures final agent answer

### Tools
- `search_child_chunks(query, limit)` — FAISS similarity search over child chunks
- `retrieve_parent_chunks(parent_id)` — Full parent text lookup from JSON store

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `SARVAM_API_KEY not set` | Set in `.env` file, env var, or Databricks secret scope |
| `FAISS index not found` | Run `build_index.py` notebook first |
| `LLM returns empty` | Check SARVAM_API_KEY is valid at https://dashboard.sarvam.ai |
| `Sarvam voice not working` | Same key is used for LLM + voice — verify it's valid |
| `Import errors` | Run `pip install -r requirements.txt` |
| `Out of memory` | Use a cluster with ≥4GB RAM for index building |
| `structured_output fails` | Ensure LLM supports tool_calling (sarvam-m does) |
