# Databricks notebook source
# MAGIC %md
# MAGIC # Build FAISS Index & Parent Store for Nyaya Sahayak
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Table `workspace.default.legal_rag_corpus` populated (or CSV files available)
# MAGIC - Cluster with `sentence-transformers`, `faiss-cpu`, `langchain-community`, `langchain-huggingface`
# MAGIC
# MAGIC **Outputs:**
# MAGIC - FAISS index at `/Volumes/workspace/default/bharat_bricks_hacks/faiss_index/`
# MAGIC - Parent store at `/Volumes/workspace/default/bharat_bricks_hacks/parent_store/`

# COMMAND ----------

# MAGIC %pip install -q faiss-cpu>=1.7 sentence-transformers>=2.2 langchain-community langchain-huggingface numpy"<2" tiktoken
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os, sys, json, hashlib
import pandas as pd
import numpy as np
from pathlib import Path

# Config
CATALOG = "workspace"
SCHEMA = "default"
TABLE = "legal_rag_corpus"
VOLUME = "bharat_bricks_hacks"

VOL_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
FAISS_DIR = f"{VOL_PATH}/faiss_index"
PARENT_DIR = f"{VOL_PATH}/parent_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 100

os.makedirs(FAISS_DIR, exist_ok=True)
os.makedirs(PARENT_DIR, exist_ok=True)
print(f"FAISS → {FAISS_DIR}")
print(f"Parent store → {PARENT_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load corpus from Delta table

# COMMAND ----------

try:
    pdf = spark.table(f"{CATALOG}.{SCHEMA}.{TABLE}").select(
        "chunk_id", "source", "doc_type", "title", "text"
    ).toPandas()
    print(f"Loaded {len(pdf)} rows from Delta table")
except Exception as e:
    print(f"Delta table not found ({e}), trying CSV fallback …")
    # Fallback: load BNS sections CSV
    csv_path = f"{VOL_PATH}/bns_sections.csv"
    if not os.path.exists(csv_path):
        csv_path = "/Workspace/datasets/BNS/bns_sections.csv"
    df_bns = pd.read_csv(csv_path)
    rows = []
    for _, r in df_bns.iterrows():
        sec = str(r.get("Section", "")).strip()
        title = str(r.get("Section _name", r.get("Section_name", ""))).strip()
        desc = str(r.get("Description", "")).strip()
        text = f"BNS Section {sec} — {title}\n\n{desc}" if desc else f"BNS Section {sec} — {title}"
        rows.append({
            "chunk_id": f"BNS_S{sec}",
            "source": f"BNS Section {sec}",
            "doc_type": "bns",
            "title": title,
            "text": text,
        })
    pdf = pd.DataFrame(rows)
    print(f"Loaded {len(pdf)} BNS sections from CSV fallback")

display(pdf.head(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create parent and child chunks

# COMMAND ----------

from langchain.text_splitter import RecursiveCharacterTextSplitter

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHILD_CHUNK_SIZE,
    chunk_overlap=CHILD_CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

parent_chunks = []  # (parent_id, content, metadata)
child_chunks = []   # (text, metadata_with_parent_id)

for _, row in pdf.iterrows():
    parent_id = str(row["chunk_id"])
    text = str(row["text"]).strip()
    if not text:
        continue

    metadata = {
        "source": str(row.get("source", "")),
        "doc_type": str(row.get("doc_type", "")),
        "title": str(row.get("title", "")),
    }

    # Save parent (full text)
    parent_chunks.append((parent_id, text, metadata))

    # Split into children
    splits = child_splitter.split_text(text)
    if not splits:
        splits = [text]

    for i, chunk_text in enumerate(splits):
        child_meta = {**metadata, "parent_id": parent_id, "child_index": i}
        child_chunks.append((chunk_text, child_meta))

print(f"Parents: {len(parent_chunks)}")
print(f"Children: {len(child_chunks)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Save parent store (JSON files)

# COMMAND ----------

saved = 0
for parent_id, content, metadata in parent_chunks:
    fp = os.path.join(PARENT_DIR, f"{parent_id}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump({"page_content": content, "metadata": metadata}, f, ensure_ascii=False, indent=2)
    saved += 1

print(f"✅ Saved {saved} parent chunks to {PARENT_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Build & save LangChain FAISS index

# COMMAND ----------

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    encode_kwargs={"normalize_embeddings": True},
)

docs = [Document(page_content=t, metadata=m) for t, m in child_chunks]
print(f"Embedding {len(docs)} child chunks …")

store = FAISS.from_documents(docs, embeddings)
store.save_local(FAISS_DIR)
print(f"✅ FAISS index saved to {FAISS_DIR} ({len(docs)} vectors)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Smoke test

# COMMAND ----------

store2 = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
results = store2.similarity_search("What is theft under BNS?", k=5)
for i, doc in enumerate(results, 1):
    print(f"\n--- Result {i} ---")
    print(f"Source: {doc.metadata.get('source', '?')}")
    print(f"Parent: {doc.metadata.get('parent_id', '?')}")
    print(f"Text: {doc.page_content[:200]}…")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Summary manifest

# COMMAND ----------

manifest = {
    "embedding_model": EMBED_MODEL,
    "total_parents": len(parent_chunks),
    "total_children": len(child_chunks),
    "child_chunk_size": CHILD_CHUNK_SIZE,
    "child_chunk_overlap": CHILD_CHUNK_OVERLAP,
    "faiss_dir": FAISS_DIR,
    "parent_dir": PARENT_DIR,
    "source_table": f"{CATALOG}.{SCHEMA}.{TABLE}",
    "doc_types": list(pdf["doc_type"].unique()),
}
manifest_path = os.path.join(FAISS_DIR, "manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(json.dumps(manifest, indent=2))
print(f"\n✅ Done! Manifest saved to {manifest_path}")
