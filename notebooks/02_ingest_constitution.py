# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 02: Constitution of India Ingestion
# MAGIC
# MAGIC Ingests the Constitution of India (article-by-article) into Unity Catalog using PySpark,
# MAGIC then appends article chunks to `legal_rag_corpus` via Delta MERGE.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Notebook 01 (`india_legal_policy_ingest.py`) already run (catalog/schema/volume exists).
# MAGIC - Cluster with internet access (downloads HuggingFace dataset).
# MAGIC
# MAGIC **Output:**
# MAGIC - Delta table: `workspace.india_legal.constitution_articles`
# MAGIC - Appended rows in: `workspace.india_legal.legal_rag_corpus` (doc_type=`constitution`)

# COMMAND ----------

import os
import subprocess
import sys

REPO_ROOT = "/Workspace/Users/11avanshdhar@gmail.com/nyaya-dhwani-hackathon"  # Edit me

_src = os.path.join(REPO_ROOT, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "datasets>=2.14.0",
    "pandas>=2.0,<3",
])
print("✅ datasets installed")

# COMMAND ----------

CATALOG = "workspace"
SCHEMA = "india_legal"
CORPUS_TABLE = f"{CATALOG}.{SCHEMA}.legal_rag_corpus"
CONST_TABLE = f"{CATALOG}.{SCHEMA}.constitution_articles"

# COMMAND ----------

# MAGIC %md ## Step 1: Download Constitution of India from HuggingFace

# COMMAND ----------

from datasets import load_dataset
import pandas as pd

# Primary dataset: article-level Constitution text
# Try multiple HuggingFace sources with fallback
_HF_DATASETS = [
    ("pbeukema/indian_constitution", "train"),
    ("Hamzeha/Constitution-of-India", "train"),
]

const_df = None
for hf_name, split in _HF_DATASETS:
    try:
        ds = load_dataset(hf_name, split=split, trust_remote_code=True)
        const_df = ds.to_pandas()
        print(f"✅ Loaded {hf_name}: {len(const_df)} rows — columns: {list(const_df.columns)}")
        break
    except Exception as e:
        print(f"⚠️  {hf_name} failed: {e}")

if const_df is None:
    # Minimal fallback: hardcode a stub of key articles so the pipeline doesn't break
    print("⚠️  All HuggingFace sources failed — using hardcoded stub (key articles only)")
    _STUB_ARTICLES = [
        (1, "I", "Name and territory of the Union",
         "India, that is Bharat, shall be a Union of States."),
        (12, "III", "Definition of State",
         "In this Part, unless the context otherwise requires, 'the State' includes the Government and Parliament of India..."),
        (13, "III", "Laws inconsistent with or in derogation of the fundamental rights",
         "All laws in force in the territory of India immediately before the commencement of this Constitution..."),
        (14, "III", "Equality before law",
         "The State shall not deny to any person equality before the law or the equal protection of the laws within the territory of India."),
        (19, "III", "Protection of certain rights regarding freedom of speech etc.",
         "All citizens shall have the right to freedom of speech and expression..."),
        (20, "III", "Protection in respect of conviction for offences",
         "No person shall be convicted of any offence except for violation of a law in force at the time of the act..."),
        (21, "III", "Protection of life and personal liberty",
         "No person shall be deprived of his life or personal liberty except according to procedure established by law."),
        (21, "III", "Right to Education",  # Note: 21A
         "The State shall provide free and compulsory education to all children of the age of six to fourteen years..."),
        (22, "III", "Protection against arrest and detention in certain cases",
         "No person who is arrested shall be detained in custody without being informed, as soon as may be..."),
        (32, "III", "Remedies for enforcement of rights conferred by this Part",
         "The right to move the Supreme Court by appropriate proceedings for the enforcement of the rights conferred by this Part is guaranteed."),
        (226, "V", "Power of High Courts to issue certain writs",
         "Every High Court shall have power, throughout the territories in relation to which it exercises jurisdiction..."),
        (300, "XII", "Suits and proceedings",
         "The Government of India may sue or be sued by the name of the Union of India..."),
        (370, "XXI", "Temporary provisions with respect to the State of Jammu and Kashmir",
         "Notwithstanding anything in this Constitution, the provisions of this article shall apply in relation to the State of Jammu and Kashmir..."),
    ]
    const_df = pd.DataFrame(
        _STUB_ARTICLES,
        columns=["article_number", "part", "article_title", "article_text"],
    )

# Normalise columns
def _normalise_const_df(df: pd.DataFrame) -> pd.DataFrame:
    """Map various HuggingFace column names to a standard schema."""
    col_map = {
        # pbeukema/indian_constitution
        "article_number": "article_number",
        "part": "part",
        "article_title": "article_title",
        "article_text": "article_text",
        # Hamzeha/Constitution-of-India (may differ)
        "Article Number": "article_number",
        "Part": "part",
        "Title": "article_title",
        "Text": "article_text",
        "Content": "article_text",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    for col in ("article_number", "part", "article_title", "article_text"):
        if col not in df.columns:
            df[col] = ""
    return df[["article_number", "part", "article_title", "article_text"]]

const_df = _normalise_const_df(const_df)
const_df["article_number"] = const_df["article_number"].astype(str).str.strip()
const_df["article_text"] = const_df["article_text"].fillna("").str.strip()
const_df = const_df[const_df["article_text"].str.len() > 10].reset_index(drop=True)
print(f"✅ {len(const_df)} constitution articles ready")
display(const_df.head(3))

# COMMAND ----------

# MAGIC %md ## Step 2: Write to Delta table (PySpark)

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType

schema = StructType([
    StructField("article_number", StringType(), True),
    StructField("part", StringType(), True),
    StructField("article_title", StringType(), True),
    StructField("article_text", StringType(), True),
])

sdf = spark.createDataFrame(const_df, schema=schema)

(
    sdf.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(CONST_TABLE)
)
print(f"✅ Wrote {sdf.count()} rows to {CONST_TABLE}")

# Enable Change Data Feed for Vector Search sync
spark.sql(f"""
    ALTER TABLE {CONST_TABLE}
    SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
""")
print("✅ CDF enabled")

# COMMAND ----------

# MAGIC %md ## Step 3: Build corpus chunks and MERGE into legal_rag_corpus

# COMMAND ----------

corpus_rows = []
for _, row in const_df.iterrows():
    art_num = str(row["article_number"]).strip()
    title = str(row["article_title"]).strip()
    text = str(row["article_text"]).strip()

    # Chunk: single article as one chunk (truncate at 2000 chars)
    chunk_id = f"CONST_A{art_num}"
    chunk_text = f"Article {art_num} of the Constitution of India: {title}\n\n{text}"[:2000]

    corpus_rows.append({
        "chunk_id": chunk_id,
        "source": "CONSTITUTION_OF_INDIA",
        "doc_type": "constitution",
        "title": f"Article {art_num}: {title}"[:200],
        "text": chunk_text,
    })

from pyspark.sql.types import StructType, StructField, StringType

corpus_schema = StructType([
    StructField("chunk_id", StringType(), False),
    StructField("source", StringType(), True),
    StructField("doc_type", StringType(), True),
    StructField("title", StringType(), True),
    StructField("text", StringType(), True),
])

import pandas as pd
corpus_pdf = pd.DataFrame(corpus_rows)
stage_sdf = spark.createDataFrame(corpus_pdf, schema=corpus_schema)
stage_sdf.createOrReplaceTempView("const_stage")

print(f"✅ Prepared {len(corpus_rows)} constitution corpus chunks")

# COMMAND ----------

# MERGE into legal_rag_corpus (upsert — preserves existing BNS/IPC/scheme chunks)
spark.sql(f"""
    MERGE INTO {CORPUS_TABLE} AS target
    USING const_stage AS source
    ON target.chunk_id = source.chunk_id
    WHEN MATCHED THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
""")
print(f"✅ MERGE complete into {CORPUS_TABLE}")

# COMMAND ----------

# MAGIC %md ## Step 4: Verify

# COMMAND ----------

verify = spark.sql(f"""
    SELECT doc_type, COUNT(*) AS cnt
    FROM {CORPUS_TABLE}
    GROUP BY doc_type
    ORDER BY cnt DESC
""")
display(verify)

spark.sql(f"""
    SELECT chunk_id, title, LEFT(text, 150) AS text_preview
    FROM {CORPUS_TABLE}
    WHERE doc_type = 'constitution'
    LIMIT 10
""").display()
