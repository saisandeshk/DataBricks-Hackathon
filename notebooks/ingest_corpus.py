# Databricks notebook source
# MAGIC %md
# MAGIC # Ingest Legal Corpus → Delta Table
# MAGIC
# MAGIC Assembles **BNS sections**, **Constitution articles**, and **SC judgments**
# MAGIC into `workspace.default.legal_rag_corpus`.
# MAGIC
# MAGIC Run this BEFORE `build_index.py`.

# COMMAND ----------

# MAGIC %pip install -q pandas requests beautifulsoup4 lxml openpyxl
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os, re, json
import pandas as pd

CATALOG = "workspace"
SCHEMA = "default"
VOLUME = "bharat_bricks_hacks"
TABLE = "legal_rag_corpus"

VOL_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

spark.sql(f"CREATE DATABASE IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
os.makedirs(VOL_PATH, exist_ok=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. BNS Sections

# COMMAND ----------

# Try Volume first, then fallback paths
csv_candidates = [
    f"{VOL_PATH}/bns_sections.csv",
    f"{VOL_PATH}/bns_sections/bns_sections.csv",
    "/Workspace/datasets/BNS/bns_sections.csv",
]

df_bns = None
for p in csv_candidates:
    if os.path.exists(p):
        df_bns = pd.read_csv(p)
        print(f"Loaded BNS from {p}: {len(df_bns)} rows")
        break

if df_bns is None:
    # GitHub fallback
    import requests
    urls = [
        "https://raw.githubusercontent.com/OpenNyAI/Opennyai/main/datasets/bns_sections.csv",
        "https://raw.githubusercontent.com/nandr39/bns-dataset/main/bns_sections.csv",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            from io import StringIO
            df_bns = pd.read_csv(StringIO(r.text))
            # Save to Volume for future runs
            csv_dest = f"{VOL_PATH}/bns_sections.csv"
            df_bns.to_csv(csv_dest, index=False)
            print(f"Downloaded BNS from GitHub: {len(df_bns)} rows → {csv_dest}")
            break
        except Exception as e:
            print(f"  {url} failed: {e}")

if df_bns is None:
    raise RuntimeError("Could not load BNS sections from any source.")

corpus_rows = []
for _, r in df_bns.iterrows():
    sec = str(r.get("Section", "")).strip()
    # Handle both "Section _name" and "Section_name" column variants
    title = str(r.get("Section _name", r.get("Section_name", ""))).strip()
    desc = str(r.get("Description", "")).strip()
    chapter = str(r.get("Chapter_name", "")).strip()

    text = f"BNS Section {sec} — {title}"
    if chapter:
        text += f" (Chapter: {chapter})"
    text += f"\n\n{desc}" if desc else ""

    corpus_rows.append({
        "chunk_id": f"BNS_S{sec}",
        "source": f"BNS Section {sec}",
        "doc_type": "bns",
        "title": title,
        "text": text.strip(),
    })

print(f"BNS: {len(corpus_rows)} sections")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Constitution Articles (if available)

# COMMAND ----------

const_path = f"{VOL_PATH}/constitution_articles.csv"
if os.path.exists(const_path):
    df_const = pd.read_csv(const_path)
    for _, r in df_const.iterrows():
        art = str(r.get("article_number", r.get("Article", ""))).strip()
        title = str(r.get("title", r.get("Title", ""))).strip()
        text = str(r.get("text", r.get("Text", r.get("content", "")))).strip()
        full_text = f"Constitution Article {art} — {title}\n\n{text}" if text else f"Constitution Article {art} — {title}"
        corpus_rows.append({
            "chunk_id": f"CONST_A{art}",
            "source": f"Constitution Article {art}",
            "doc_type": "constitution",
            "title": title,
            "text": full_text.strip(),
        })
    print(f"Constitution: {len(df_const)} articles added")
else:
    print(f"No Constitution CSV at {const_path} — skipping")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. SC Judgments (if available)

# COMMAND ----------

sc_path = f"{VOL_PATH}/sc_judgments.csv"
if os.path.exists(sc_path):
    df_sc = pd.read_csv(sc_path)
    for idx, r in df_sc.iterrows():
        case_id = str(r.get("case_id", f"SC_{idx}")).strip()
        case_name = str(r.get("case_name", r.get("title", ""))).strip()
        text = str(r.get("text", r.get("content", r.get("judgment", "")))).strip()
        full_text = f"Supreme Court: {case_name}\n\n{text}" if text else f"Supreme Court: {case_name}"
        corpus_rows.append({
            "chunk_id": case_id,
            "source": case_name,
            "doc_type": "sc_judgment",
            "title": case_name,
            "text": full_text.strip(),
        })
    print(f"SC Judgments: {len(df_sc)} cases added")
else:
    print(f"No SC judgments CSV at {sc_path} — skipping")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. IndicLegalQA (if available)

# COMMAND ----------

ilqa_path = f"{VOL_PATH}/indiclegalqa.csv"
if os.path.exists(ilqa_path):
    df_qa = pd.read_csv(ilqa_path)
    count = 0
    for idx, r in df_qa.iterrows():
        q = str(r.get("question", "")).strip()
        a = str(r.get("answer", "")).strip()
        if not q or not a:
            continue
        text = f"Legal Q&A:\nQ: {q}\nA: {a}"
        corpus_rows.append({
            "chunk_id": f"ILQA_{idx}",
            "source": "IndicLegalQA",
            "doc_type": "indiclegalqa",
            "title": q[:100],
            "text": text,
        })
        count += 1
    print(f"IndicLegalQA: {count} QA pairs added")
else:
    print(f"No IndicLegalQA CSV at {ilqa_path} — skipping")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Save to Delta table

# COMMAND ----------

df_corpus = pd.DataFrame(corpus_rows)
print(f"\nTotal corpus: {len(df_corpus)} chunks")
print(f"Doc types: {df_corpus['doc_type'].value_counts().to_dict()}")

sdf = spark.createDataFrame(df_corpus)
sdf.write.format("delta").mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.{TABLE}")
print(f"\n✅ Saved to {CATALOG}.{SCHEMA}.{TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Verify

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT doc_type, COUNT(*) as cnt FROM workspace.default.legal_rag_corpus GROUP BY doc_type ORDER BY cnt DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT chunk_id, source, doc_type, LEFT(text, 150) AS preview FROM workspace.default.legal_rag_corpus LIMIT 10
