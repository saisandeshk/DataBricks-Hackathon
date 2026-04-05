# Nyaya Sahayak — Deployment Guide

## Architecture

```
User → Gradio UI (voice + text + 13 languages)
        ↓
    LangGraph Agent (query rewrite → orchestrator → tools → compress → aggregate)
        ↓
    FAISS Index (child chunks) + Parent Store (full sections)
        ↓
    Databricks AI Gateway (Llama 4 Maverick)
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
- Sarvam AI API key (free tier at https://sarvam.ai — for voice/translation)

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

## Step 3: Upload BNS Dataset

1. Download `bns_sections.csv` from the `datasets/BNS/` folder in this repo
2. In Databricks: **Catalog → workspace → default → bharat_bricks_hacks**
3. Click **Upload to Volume** → upload `bns_sections.csv`

Optional (for richer corpus):
- Upload `constitution_articles.csv` (Constitution of India articles)
- Upload `sc_judgments.csv` (Supreme Court judgment excerpts)
- Upload `indiclegalqa.csv` (IndicLegalQA dataset)

---

## Step 4: Create Secrets

```bash
# In Databricks CLI (or notebook with %sh)
databricks secrets create-scope nyaya-sahayak

# Databricks PAT token
databricks secrets put-secret nyaya-sahayak databricks-token

# Sarvam API key (optional — for voice/translation)
databricks secrets put-secret nyaya-sahayak sarvam-api-key
```

**Alternative (environment variables in notebook):**
```python
import os
os.environ["DATABRICKS_TOKEN"] = dbutils.secrets.get("nyaya-sahayak", "databricks-token")
os.environ["SARVAM_API_KEY"] = dbutils.secrets.get("nyaya-sahayak", "sarvam-api-key")
```

---

## Step 5: Set Up AI Gateway

1. In Databricks: **Serving → AI Gateway → Create endpoint**
2. Or use an existing endpoint. Note the base URL:
   ```
   https://<workspace-id>.ai-gateway.cloud.databricks.com/mlflow/v1
   ```
3. Ensure the model `databricks-llama-4-maverick` is available

---

## Step 6: Run Ingestion Notebook

1. Import `notebooks/ingest_corpus.py` into your Databricks workspace
2. Attach to a cluster (any size — single node works)
3. Update `CATALOG`, `SCHEMA`, `VOLUME` if different from defaults
4. Run all cells

**Expected output:**
```
BNS: 358 sections
Total corpus: 358+ chunks (more with Constitution/SC/IndicLegalQA)
✅ Saved to workspace.default.legal_rag_corpus
```

---

## Step 7: Build FAISS Index

1. Import `notebooks/build_index.py` into your Databricks workspace
2. Attach to a cluster with ≥4GB RAM
3. Run all cells

**Expected output:**
```
Parents: 358+
Children: 1000+ (depends on chunk splitting)
✅ FAISS index saved to /Volumes/workspace/default/bharat_bricks_hacks/faiss_index/
✅ Parent store saved to /Volumes/workspace/default/bharat_bricks_hacks/parent_store/
```

---

## Step 8: Deploy as Databricks App

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
   - `DATABRICKS_TOKEN` — from secret scope
   - `LLM_OPENAI_BASE_URL` — your AI Gateway URL
   - `LLM_MODEL` — `databricks-llama-4-maverick`
   - `SARVAM_API_KEY` — from secret scope
   - `FAISS_INDEX_DIR` — Volume path for FAISS index
   - `PARENT_STORE_DIR` — Volume path for parent chunks

### Option B: Run in Notebook

```python
# In a Databricks notebook cell:
import os, sys

os.environ["DATABRICKS_TOKEN"] = dbutils.secrets.get("nyaya-sahayak", "databricks-token")
os.environ["LLM_OPENAI_BASE_URL"] = "https://<workspace-id>.ai-gateway.cloud.databricks.com/mlflow/v1"
os.environ["LLM_MODEL"] = "databricks-llama-4-maverick"
os.environ["SARVAM_API_KEY"] = dbutils.secrets.get("nyaya-sahayak", "sarvam-api-key")
os.environ["FAISS_INDEX_DIR"] = "/Volumes/workspace/default/bharat_bricks_hacks/faiss_index"
os.environ["PARENT_STORE_DIR"] = "/Volumes/workspace/default/bharat_bricks_hacks/parent_store"

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

# Set environment variables
export DATABRICKS_TOKEN="dapi..."
export LLM_OPENAI_BASE_URL="https://<workspace-id>.ai-gateway.cloud.databricks.com/mlflow/v1"
export LLM_MODEL="databricks-llama-4-maverick"
export SARVAM_API_KEY="your-sarvam-key"
export FAISS_INDEX_DIR="./faiss_index"      # copy from Volume
export PARENT_STORE_DIR="./parent_store"    # copy from Volume

pip install -r requirements.txt
python app/main.py
```

---

## Step 9: Verify

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
├── commands.md              # This file
├── app/
│   ├── __init__.py
│   └── main.py              # Entry point
├── src/
│   ├── __init__.py
│   ├── config.py            # All configuration constants
│   ├── llm_client.py        # Databricks AI Gateway wrapper
│   ├── sarvam_client.py     # Sarvam STT/TTS/Translation
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
    ├── ingest_corpus.py     # BNS + Constitution + SC → Delta table
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
| `DATABRICKS_TOKEN not set` | Set via env var or Databricks secret scope |
| `FAISS index not found` | Run `build_index.py` notebook first |
| `LLM returns empty` | Check AI Gateway URL and model availability |
| `Sarvam voice not working` | Verify SARVAM_API_KEY is set and valid |
| `Import errors` | Run `pip install -r requirements.txt` |
| `Out of memory` | Use a cluster with ≥4GB RAM for index building |
| `structured_output fails` | Ensure LLM supports tool_calling (Llama 4 Maverick does) |
