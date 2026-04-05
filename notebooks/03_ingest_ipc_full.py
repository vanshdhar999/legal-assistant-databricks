# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 03: IPC Full Text Ingestion
# MAGIC
# MAGIC Loads Indian Penal Code (IPC 1860) section text from local CSV sources,
# MAGIC cross-references with the existing `bns_ipc_mapping` table, and upserts into
# MAGIC `legal_rag_corpus` with `doc_type = "criminal_law_ipc"`.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Notebook 01 run (bns_ipc_mapping table exists).
# MAGIC
# MAGIC **Output:**
# MAGIC - Delta table: `workspace.india_legal.ipc_sections`
# MAGIC - Appended rows in: `workspace.india_legal.legal_rag_corpus` (doc_type=`criminal_law_ipc`)

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
    "pandas>=2.0,<3",
])
print("✅ dependencies installed")

# COMMAND ----------

CATALOG = "workspace"
SCHEMA = "india_legal"
IPC_TABLE = f"{CATALOG}.{SCHEMA}.ipc_sections"
MAPPING_TABLE = f"{CATALOG}.{SCHEMA}.bns_ipc_mapping"
CORPUS_TABLE = f"{CATALOG}.{SCHEMA}.legal_rag_corpus"

# COMMAND ----------

# MAGIC %md ## Step 1: Load IPC section text from CSV (local-first)

# COMMAND ----------

import pandas as pd
from pathlib import Path

ipc_df = None

# Prefer local assets in repo/volume/workspace over remote loaders.
_CSV_CANDIDATES = [
    Path("/Volumes/workspace/india_legal/legal_files/my_gov_schemes.csv"),
    Path(REPO_ROOT) / "IPC.csv",
    Path("/Workspace") / "IPC.csv",
    Path("IPC.csv"),
]

def _load_ipc_from_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded IPC CSV: {csv_path} ({len(df)} rows)")

    if "URL" in df.columns:
        extracted = (
            df["URL"]
            .astype(str)
            .str.extract(r"section-([0-9A-Za-z]+)")[0]
            .str.strip()
        )
        if "section_number" not in df.columns:
            df["section_number"] = extracted
        else:
            df["section_number"] = (
                df["section_number"]
                .astype(str)
                .str.strip()
                .mask(lambda s: s.eq("") | s.eq("nan"), extracted)
            )

    if "section_title" not in df.columns:
        if "Offense" in df.columns:
            df["section_title"] = df["Offense"].astype(str)
        else:
            df["section_title"] = ""

    if "section_text" not in df.columns:
        if "Description" in df.columns:
            df["section_text"] = df["Description"].astype(str)
        else:
            df["section_text"] = ""

    return df

for c in _CSV_CANDIDATES:
    if c.exists():
        try:
            ipc_df = _load_ipc_from_csv(c)
            break
        except Exception as e:
            print(f"⚠️  CSV load failed for {c}: {e}")

if ipc_df is None:
    # Final fallback: use the mapping table and a small set of high-frequency IPC sections.
    print("⚠️  No usable IPC CSV source found — building minimal stub dataset")
    try:
        mapping_pdf = spark.table(MAPPING_TABLE).toPandas()
        ipc_sections_in_mapping = mapping_pdf["ipc_section"].dropna().unique().tolist()
        print(f"  Found {len(ipc_sections_in_mapping)} IPC section numbers from mapping table")
    except Exception as e:
        print(f"  Could not load mapping table: {e}")
        ipc_sections_in_mapping = []

    # Hardcoded stubs for the most commonly referenced IPC sections
    _IPC_STUBS = {
        "302": ("Murder", "Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine."),
        "304": ("Culpable homicide not amounting to murder", "Whoever commits culpable homicide not amounting to murder shall be punished..."),
        "307": ("Attempt to murder", "Whoever does any act with such intention or knowledge, and under such circumstances that, if he by that act caused death, he would be guilty of murder..."),
        "376": ("Punishment for rape", "Whoever commits rape shall be punished with rigorous imprisonment of either description for a term which shall not be less than seven years..."),
        "378": ("Theft", "Whoever, intending to take dishonestly any moveable property out of the possession of any person without that person's consent..."),
        "379": ("Punishment for theft", "Whoever commits theft shall be punished with imprisonment of either description for a term which may extend to three years, or with fine, or with both."),
        "392": ("Robbery", "Whoever commits robbery shall be punished with rigorous imprisonment for a term which may extend to ten years, and shall also be liable to fine."),
        "395": ("Dacoity", "Whoever commits dacoity shall be punished with imprisonment for life, or with rigorous imprisonment for a term which may extend to ten years, and shall also be liable to fine."),
        "406": ("Punishment for criminal breach of trust", "Whoever commits criminal breach of trust shall be punished with imprisonment of either description for a term which may extend to three years..."),
        "420": ("Cheating and dishonestly inducing delivery of property", "Whoever cheats and thereby dishonestly induces the person deceived to deliver any property to any person..."),
        "498A": ("Husband or relative of husband of a woman subjecting her to cruelty", "Whoever, being the husband or the relative of the husband of a woman, subjects such woman to cruelty shall be punished..."),
    }

    rows = []
    for num, (title, text) in _IPC_STUBS.items():
        rows.append({"section_number": num, "section_title": title, "section_text": text})
    ipc_df = pd.DataFrame(rows)
    print(f"  Using {len(ipc_df)} hardcoded IPC section stubs")

# Normalise column names from heterogeneous CSV representations.
def _normalise_ipc_df(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {
        "section_number": "section_number",
        "Section": "section_number",
        "section": "section_number",
        "IPC Section": "section_number",
        "section_title": "section_title",
        "Title": "section_title",
        "title": "section_title",
        "Offense": "section_title",
        "offense": "section_title",
        "section_text": "section_text",
        "Description": "section_text",
        "description": "section_text",
        "text": "section_text",
        "Content": "section_text",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    for col in ("section_number", "section_title", "section_text"):
        if col not in df.columns:
            df[col] = ""
    return df[["section_number", "section_title", "section_text"]]

ipc_df = _normalise_ipc_df(ipc_df)
ipc_df["section_number"] = ipc_df["section_number"].astype(str).str.strip()
ipc_df["section_text"] = ipc_df["section_text"].fillna("").str.strip()
ipc_df = ipc_df[ipc_df["section_number"].str.len() > 0]
ipc_df = ipc_df[ipc_df["section_text"].str.len() > 5].reset_index(drop=True)
ipc_df = ipc_df.drop_duplicates(subset=["section_number"], keep="first").reset_index(drop=True)
print(f"✅ {len(ipc_df)} IPC sections ready")
display(ipc_df.head(3))

# COMMAND ----------

# MAGIC %md ## Step 2: Cross-reference with bns_ipc_mapping

# COMMAND ----------

try:
    mapping_pdf = spark.table(MAPPING_TABLE).toPandas()
    mapping_pdf["ipc_section"] = mapping_pdf["ipc_section"].astype(str).str.strip()
    ipc_df["section_number"] = ipc_df["section_number"].astype(str).str.strip()
    ipc_df = ipc_df.merge(
        mapping_pdf[["ipc_section", "bns_section", "status"]].rename(
            columns={"ipc_section": "section_number", "bns_section": "bns_equivalent", "status": "mapping_status"}
        ),
        on="section_number",
        how="left",
    )
    print(f"✅ Cross-referenced: {ipc_df['bns_equivalent'].notna().sum()} sections have BNS mapping")
except Exception as e:
    print(f"⚠️  Could not join with bns_ipc_mapping: {e}")
    ipc_df["bns_equivalent"] = None
    ipc_df["mapping_status"] = None

display(ipc_df.head(3))

# COMMAND ----------

# MAGIC %md ## Step 3: Write ipc_sections Delta table

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType

ipc_schema = StructType([
    StructField("section_number", StringType(), True),
    StructField("section_title", StringType(), True),
    StructField("section_text", StringType(), True),
    StructField("bns_equivalent", StringType(), True),
    StructField("mapping_status", StringType(), True),
])

ipc_sdf = spark.createDataFrame(ipc_df.astype(str).fillna(""), schema=ipc_schema)
(
    ipc_sdf.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(IPC_TABLE)
)
spark.sql(f"ALTER TABLE {IPC_TABLE} SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')")
print(f"✅ Wrote {ipc_sdf.count()} rows to {IPC_TABLE}")

# COMMAND ----------

# MAGIC %md ## Step 4: Build corpus chunks with BNS cross-reference text

# COMMAND ----------

corpus_rows = []
for _, row in ipc_df.iterrows():
    sec_num = str(row["section_number"]).strip()
    title = str(row.get("section_title", "")).strip()
    text = str(row.get("section_text", "")).strip()
    bns_eq = str(row.get("bns_equivalent", "")).strip()
    status = str(row.get("mapping_status", "")).strip()

    # Enrich text with BNS cross-reference information
    bns_note = ""
    if bns_eq and bns_eq not in ("nan", "None", ""):
        bns_note = f"\n\nBNS Equivalent: Section {bns_eq} of Bharatiya Nyaya Sanhita 2023"
        if status and status not in ("nan", "None", ""):
            bns_note += f" (Status: {status})"
    elif status and "repeal" in status.lower():
        bns_note = "\n\nThis IPC section has been repealed under the Bharatiya Nyaya Sanhita (BNS) 2023."

    chunk_text = (
        f"IPC Section {sec_num} – {title}\n\n{text}{bns_note}"
    )[:2000]

    corpus_rows.append({
        "chunk_id": f"IPC_S{sec_num}",
        "source": "IPC_1860",
        "doc_type": "criminal_law_ipc",
        "title": f"IPC Section {sec_num}: {title}"[:200],
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

stage_sdf = spark.createDataFrame(pd.DataFrame(corpus_rows), schema=corpus_schema)
stage_sdf.createOrReplaceTempView("ipc_corpus_stage")
print(f"✅ Prepared {len(corpus_rows)} IPC corpus chunks")

# COMMAND ----------

spark.sql(f"""
    MERGE INTO {CORPUS_TABLE} AS target
    USING ipc_corpus_stage AS source
    ON target.chunk_id = source.chunk_id
    WHEN MATCHED THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
""")
print(f"✅ MERGE complete into {CORPUS_TABLE}")

# COMMAND ----------

# MAGIC %md ## Step 5: Spark analysis — IPC coverage statistics

# COMMAND ----------

from pyspark.sql import functions as F

ipc_sdf_loaded = spark.table(IPC_TABLE)

# Coverage: how many IPC sections have BNS mappings?
coverage = ipc_sdf_loaded.agg(
    F.count("*").alias("total_ipc_sections"),
    F.count(F.when(F.col("bns_equivalent").isNotNull() & (F.col("bns_equivalent") != ""), "x")).alias("mapped_to_bns"),
    F.count(F.when(F.col("mapping_status").contains("Repealed"), "x")).alias("repealed_in_bns"),
).collect()[0]

print(f"Total IPC sections ingested : {coverage['total_ipc_sections']}")
print(f"Mapped to BNS equivalent   : {coverage['mapped_to_bns']}")
print(f"Repealed in BNS            : {coverage['repealed_in_bns']}")

# Corpus table doc_type distribution
display(spark.sql(f"""
    SELECT doc_type, COUNT(*) AS cnt
    FROM {CORPUS_TABLE}
    GROUP BY doc_type ORDER BY cnt DESC
"""))
