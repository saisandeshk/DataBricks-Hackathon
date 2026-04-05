# Databricks notebook source
# MAGIC %md
# MAGIC # Ingest Legal Corpus → Delta Table
# MAGIC
# MAGIC Assembles the legal knowledge base from **three sources**:
# MAGIC 1. **BNS sections** — `bns_sections.csv` (structured CSV)
# MAGIC 2. **Legal PDFs** — Any PDF in `pdfs/` folder (Constitution, SC judgments, etc.)
# MAGIC 3. **IndicLegalQA** — `IndicLegalQA Dataset_10K_Revised.json` (QA pairs)
# MAGIC
# MAGIC **PDF pipeline** (from agentic-rag-for-dummies):
# MAGIC ```
# MAGIC PDF → pymupdf4llm → Markdown → MarkdownHeaderTextSplitter → Parent chunks
# MAGIC                                                              → RecursiveCharacterTextSplitter → Child chunks
# MAGIC ```
# MAGIC
# MAGIC Run this BEFORE `build_index.py`.

# COMMAND ----------

# MAGIC %pip install -q pandas requests pymupdf pymupdf4llm langchain-text-splitters
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os, sys, json, glob
import pandas as pd
from pathlib import Path

CATALOG = "workspace"
SCHEMA = "default"
VOLUME = "bharat_bricks_hacks"
TABLE = "legal_rag_corpus"

VOL_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
PDF_DIR = f"{VOL_PATH}/pdfs"
MARKDOWN_DIR = f"{VOL_PATH}/markdown_docs"

spark.sql(f"CREATE DATABASE IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
os.makedirs(VOL_PATH, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(MARKDOWN_DIR, exist_ok=True)

print(f"Volume:   {VOL_PATH}")
print(f"PDFs:     {PDF_DIR}")
print(f"Markdown: {MARKDOWN_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. BNS Sections (CSV)

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
# MAGIC ## 2. Legal PDFs → Markdown → Chunks
# MAGIC
# MAGIC Upload any legal PDFs to `pdfs/` folder on the Volume, e.g.:
# MAGIC - `constitution_of_india.pdf`
# MAGIC - `sc_judgments_compilation.pdf`
# MAGIC - `bns_2023_gazette.pdf`
# MAGIC - Any other legal document PDF
# MAGIC
# MAGIC The pipeline uses `pymupdf4llm` to convert PDF → Markdown, then splits
# MAGIC into parent/child chunks using the agentic-rag approach.

# COMMAND ----------

import pymupdf
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Chunking parameters (same as agentic-rag-for-dummies)
CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 100
MIN_PARENT_SIZE = 2000
MAX_PARENT_SIZE = 4000
HEADERS_TO_SPLIT_ON = [("#", "H1"), ("##", "H2"), ("###", "H3")]

parent_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=HEADERS_TO_SPLIT_ON,
    strip_headers=False,
)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHILD_CHUNK_SIZE,
    chunk_overlap=CHILD_CHUNK_OVERLAP,
)

def pdf_to_markdown(pdf_path, output_dir):
    """Convert a single PDF to Markdown using pymupdf4llm."""
    doc = pymupdf.open(str(pdf_path))
    md = pymupdf4llm.to_markdown(
        doc, header=False, footer=False,
        page_separators=True, ignore_images=True,
        write_images=False, image_path=None,
    )
    md_cleaned = md.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="ignore")
    md_path = Path(output_dir) / f"{Path(pdf_path).stem}.md"
    md_path.write_text(md_cleaned, encoding="utf-8")
    return md_path

def merge_small_parents(chunks, min_size):
    """Merge parent chunks smaller than min_size with neighbors."""
    if not chunks:
        return []
    merged, current = [], None
    for chunk in chunks:
        if current is None:
            current = chunk
        else:
            current.page_content += "\n\n" + chunk.page_content
            for k, v in chunk.metadata.items():
                if k in current.metadata:
                    current.metadata[k] = f"{current.metadata[k]} -> {v}"
                else:
                    current.metadata[k] = v
        if len(current.page_content) >= min_size:
            merged.append(current)
            current = None
    if current:
        if merged:
            merged[-1].page_content += "\n\n" + current.page_content
        else:
            merged.append(current)
    return merged

def split_large_parents(chunks, max_size):
    """Split parent chunks larger than max_size."""
    result = []
    for chunk in chunks:
        if len(chunk.page_content) <= max_size:
            result.append(chunk)
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_size, chunk_overlap=CHILD_CHUNK_OVERLAP
            )
            result.extend(splitter.split_documents([chunk]))
    return result

def process_markdown(md_path):
    """Split a Markdown file into parent + child chunks."""
    text = Path(md_path).read_text(encoding="utf-8")
    if not text.strip():
        return [], []
    raw_parents = parent_splitter.split_text(text)
    merged = merge_small_parents(raw_parents, MIN_PARENT_SIZE)
    split = split_large_parents(merged, MAX_PARENT_SIZE)

    parents_out, children_out = [], []
    stem = Path(md_path).stem
    for i, p_chunk in enumerate(split):
        parent_id = f"{stem}_parent_{i}"
        p_chunk.metadata.update({"source": f"{stem}.pdf", "parent_id": parent_id})
        parents_out.append((parent_id, p_chunk))
        children_out.extend(child_splitter.split_documents([p_chunk]))
    return parents_out, children_out

# Process all PDFs
pdf_files = sorted(Path(PDF_DIR).glob("*.pdf"))
print(f"Found {len(pdf_files)} PDF(s) in {PDF_DIR}")

all_pdf_parents = []
all_pdf_children = []

for pdf_path in pdf_files:
    print(f"\n📄 Processing: {pdf_path.name}")
    md_path = pdf_to_markdown(pdf_path, MARKDOWN_DIR)
    print(f"   → Markdown: {md_path.name} ({md_path.stat().st_size / 1024:.1f} KB)")
    parents, children = process_markdown(md_path)
    all_pdf_parents.extend(parents)
    all_pdf_children.extend(children)
    print(f"   → {len(parents)} parent chunks, {len(children)} child chunks")

# Add PDF-derived chunks to corpus
for parent_id, p_doc in all_pdf_parents:
    # Determine doc_type from filename
    stem = Path(p_doc.metadata.get("source", "")).stem.lower()
    if "constitution" in stem:
        doc_type = "constitution"
    elif "judgment" in stem or "sc_" in stem or "supreme" in stem:
        doc_type = "sc_judgment"
    elif "bns" in stem:
        doc_type = "bns_pdf"
    else:
        doc_type = "legal_pdf"

    corpus_rows.append({
        "chunk_id": parent_id,
        "source": p_doc.metadata.get("source", stem),
        "doc_type": doc_type,
        "title": p_doc.metadata.get("H1", p_doc.metadata.get("H2", stem)),
        "text": p_doc.page_content,
    })

print(f"\nPDF totals: {len(all_pdf_parents)} parents, {len(all_pdf_children)} children")
if not pdf_files:
    print("⚠️  No PDFs found. Upload Constitution/SC judgment PDFs to the pdfs/ folder on Volume.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. IndicLegalQA — Supreme Court QA pairs (JSON)
# MAGIC
# MAGIC Loads `IndicLegalQA Dataset_10K_Revised.json` from the Volume.
# MAGIC Creates two kinds of chunks per case:
# MAGIC - **Case summary** — groups all QA pairs for a case into one rich chunk
# MAGIC - **Individual QA** — each question-answer as a standalone chunk

# COMMAND ----------

# Look for the JSON file in several possible locations
ilqa_candidates = [
    f"{VOL_PATH}/IndicLegalQA Dataset_10K_Revised.json",
    f"{VOL_PATH}/IndicLegalQA_Dataset_10K_Revised.json",
    f"{VOL_PATH}/indiclegalqa.json",
    "/Workspace/datasets/IndicLegalQA Dataset_10K_Revised.json",
]

ilqa_data = None
for p in ilqa_candidates:
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            ilqa_data = json.load(f)
        print(f"Loaded IndicLegalQA from {p}: {len(ilqa_data)} entries")
        break

if ilqa_data is not None:
    # Group by case_name for case-summary chunks
    from collections import defaultdict
    cases = defaultdict(list)
    for entry in ilqa_data:
        cn = entry.get("case_name", "Unknown Case").strip()
        cases[cn].append(entry)

    # Strategy 1: Case summary chunks (one per case, all QAs combined)
    case_count = 0
    for case_name, entries in cases.items():
        date = entries[0].get("judgement_date", "")
        qa_parts = []
        for e in entries:
            q = e.get("question", "").strip()
            a = e.get("answer", "").strip()
            if q and a:
                qa_parts.append(f"Q: {q}\nA: {a}")
        if not qa_parts:
            continue
        summary_text = f"Supreme Court Case: {case_name}"
        if date:
            summary_text += f" (Date: {date})"
        summary_text += "\n\n" + "\n\n".join(qa_parts)
        # Truncate very long summaries
        if len(summary_text) > 4000:
            summary_text = summary_text[:4000] + "..."
        corpus_rows.append({
            "chunk_id": f"ILQA_CASE_{case_count}",
            "source": case_name,
            "doc_type": "sc_judgment",
            "title": case_name,
            "text": summary_text,
        })
        case_count += 1

    # Strategy 2: Individual QA chunks
    qa_count = 0
    for idx, entry in enumerate(ilqa_data):
        q = entry.get("question", "").strip()
        a = entry.get("answer", "").strip()
        cn = entry.get("case_name", "").strip()
        if not q or not a:
            continue
        text = f"Legal Q&A — {cn}\nQ: {q}\nA: {a}"
        corpus_rows.append({
            "chunk_id": f"ILQA_{idx}",
            "source": "IndicLegalQA",
            "doc_type": "indiclegalqa",
            "title": q[:100],
            "text": text,
        })
        qa_count += 1

    print(f"IndicLegalQA: {case_count} case summaries + {qa_count} individual QA chunks")
else:
    print("⚠️  IndicLegalQA JSON not found — skipping.")
    print("  Upload 'IndicLegalQA Dataset_10K_Revised.json' to your Volume.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Save to Delta table

# COMMAND ----------

df_corpus = pd.DataFrame(corpus_rows)
print(f"\nTotal corpus: {len(df_corpus)} chunks")
print(f"Doc types: {df_corpus['doc_type'].value_counts().to_dict()}")

sdf = spark.createDataFrame(df_corpus)
sdf.write.format("delta").mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.{TABLE}")
print(f"\n✅ Saved to {CATALOG}.{SCHEMA}.{TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Verify

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT doc_type, COUNT(*) as cnt FROM workspace.default.legal_rag_corpus GROUP BY doc_type ORDER BY cnt DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT chunk_id, source, doc_type, LEFT(text, 150) AS preview FROM workspace.default.legal_rag_corpus LIMIT 10
