# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 04: Government Schemes Ingestion + SparkML TF-IDF
# MAGIC
# MAGIC Fetches Indian government welfare schemes (MyScheme / HuggingFace), writes a structured
# MAGIC Delta table with parsed eligibility columns, trains a **SparkML TF-IDF** model for
# MAGIC keyword-based scheme matching, and upserts scheme chunks into `legal_rag_corpus`.
# MAGIC
# MAGIC **Prerequisites:** Notebook 01 run (catalog/schema/volume exist).
# MAGIC
# MAGIC **Output tables:**
# MAGIC - `workspace.india_legal.gov_welfare_schemes` — structured eligibility columns
# MAGIC - `workspace.india_legal.scheme_tfidf_scores` — TF-IDF feature vectors
# MAGIC - Appended rows in `workspace.india_legal.legal_rag_corpus` (doc_type=`government_scheme`)
# MAGIC
# MAGIC **Saved models:**
# MAGIC - `/Volumes/workspace/india_legal/legal_files/scheme_tfidf_model` — fitted SparkML Pipeline

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
    "datasets>=2.14.0", "pandas>=2.0,<3",
])
print("✅ dependencies installed")

# COMMAND ----------

CATALOG = "workspace"
SCHEMA = "india_legal"
SCHEMES_TABLE = f"{CATALOG}.{SCHEMA}.gov_welfare_schemes"
TFIDF_TABLE = f"{CATALOG}.{SCHEMA}.scheme_tfidf_scores"
CORPUS_TABLE = f"{CATALOG}.{SCHEMA}.legal_rag_corpus"
MODEL_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/legal_files/scheme_tfidf_model"

# COMMAND ----------

# MAGIC %md ## Step 1: Load Government Schemes Dataset

# COMMAND ----------

import pandas as pd

_HF_SCHEME_DATASETS = [
    ("nannanav/gov_myscheme", "train"),
    ("psmathur/india_government_schemes", "train"),
    ("vaibhavs10/myscheme", "train"),
]

schemes_df = None
for hf_name, split in _HF_SCHEME_DATASETS:
    try:
        from datasets import load_dataset
        ds = load_dataset(hf_name, split=split, trust_remote_code=True)
        schemes_df = ds.to_pandas()
        print(f"✅ Loaded {hf_name}: {len(schemes_df)} rows, columns: {list(schemes_df.columns)}")
        break
    except Exception as e:
        print(f"⚠️  {hf_name}: {e}")

if schemes_df is None:
    # Fallback: hardcoded stubs covering major central government schemes
    print("⚠️  HuggingFace datasets unavailable — using hardcoded scheme stubs")
    _SCHEME_STUBS = [
        {
            "scheme_id": "PMAY001", "scheme_name": "Pradhan Mantri Awas Yojana (PMAY)",
            "ministry": "Ministry of Housing and Urban Affairs",
            "state": "ALL",
            "description": "Affordable housing scheme for urban and rural poor. Provides financial assistance for construction or purchase of a pucca house.",
            "eligibility": "Families with annual income below Rs 18 lakh (Urban) or Rs 2 lakh (Rural). No pucca house in family name.",
            "benefits": "Subsidy up to Rs 2.67 lakh on home loans; direct benefit transfer for rural beneficiaries.",
            "min_age": "18", "max_age": "60", "gender": "ALL",
            "income_limit_inr": "1800000", "caste_category": "ALL",
            "occupation_tags": "BPL,Labour/Worker,Self-employed",
        },
        {
            "scheme_id": "PM_KISAN001", "scheme_name": "PM-KISAN (Pradhan Mantri Kisan Samman Nidhi)",
            "ministry": "Ministry of Agriculture & Farmers Welfare",
            "state": "ALL",
            "description": "Direct income support of Rs 6000 per year to farmer families.",
            "eligibility": "Small and marginal farmers with cultivable land up to 2 hectares.",
            "benefits": "Rs 6000 per year in three equal installments directly to bank account.",
            "min_age": "18", "max_age": "None", "gender": "ALL",
            "income_limit_inr": "None", "caste_category": "ALL",
            "occupation_tags": "Farmer",
        },
        {
            "scheme_id": "MUDRA001", "scheme_name": "Pradhan Mantri MUDRA Yojana (PMMY)",
            "ministry": "Ministry of Finance",
            "state": "ALL",
            "description": "Collateral-free loans to micro/small businesses.",
            "eligibility": "Non-corporate, non-farm small/micro enterprises. No existing bank default.",
            "benefits": "Loans up to Rs 10 lakh: Shishu (up to 50K), Kishor (50K-5L), Tarun (5L-10L).",
            "min_age": "18", "max_age": "None", "gender": "ALL",
            "income_limit_inr": "None", "caste_category": "ALL",
            "occupation_tags": "Self-employed,Labour/Worker",
        },
        {
            "scheme_id": "BETI001", "scheme_name": "Beti Bachao Beti Padhao",
            "ministry": "Ministry of Women and Child Development",
            "state": "ALL",
            "description": "Scheme to address declining child sex ratio and promote welfare of girl child.",
            "eligibility": "Girl children from birth. Special focus on districts with low sex ratio.",
            "benefits": "Conditional cash transfers, scholarship support, awareness campaigns.",
            "min_age": "0", "max_age": "21", "gender": "F",
            "income_limit_inr": "None", "caste_category": "ALL",
            "occupation_tags": "Student",
        },
        {
            "scheme_id": "PMJAY001", "scheme_name": "Pradhan Mantri Jan Arogya Yojana (PM-JAY) / Ayushman Bharat",
            "ministry": "Ministry of Health and Family Welfare",
            "state": "ALL",
            "description": "Health insurance scheme providing Rs 5 lakh per family per year.",
            "eligibility": "Families listed in SECC 2011 database. BPL families, SC/ST households, casual labourers.",
            "benefits": "Health cover of Rs 5 lakh per family per year for secondary and tertiary hospitalization.",
            "min_age": "0", "max_age": "None", "gender": "ALL",
            "income_limit_inr": "300000", "caste_category": "ALL",
            "occupation_tags": "BPL,Farmer,Labour/Worker",
        },
        {
            "scheme_id": "SC_SCHOLAR001", "scheme_name": "Post Matric Scholarship for SC Students",
            "ministry": "Ministry of Social Justice and Empowerment",
            "state": "ALL",
            "description": "Scholarships for Scheduled Caste students pursuing post-matric education.",
            "eligibility": "SC students with family income below Rs 2.5 lakh per annum.",
            "benefits": "Maintenance allowance, tuition fee reimbursement, study allowance.",
            "min_age": "15", "max_age": "35", "gender": "ALL",
            "income_limit_inr": "250000", "caste_category": "SC",
            "occupation_tags": "Student",
        },
        {
            "scheme_id": "OBC_SCHOLAR001", "scheme_name": "Post Matric Scholarship for OBC Students",
            "ministry": "Ministry of Social Justice and Empowerment",
            "state": "ALL",
            "description": "Scholarships for Other Backward Class students pursuing post-matric education.",
            "eligibility": "OBC students with family income below Rs 1 lakh per annum.",
            "benefits": "Maintenance allowance, tuition fee support.",
            "min_age": "15", "max_age": "35", "gender": "ALL",
            "income_limit_inr": "100000", "caste_category": "OBC",
            "occupation_tags": "Student",
        },
        {
            "scheme_id": "NREGA001", "scheme_name": "Mahatma Gandhi National Rural Employment Guarantee Act (MGNREGA)",
            "ministry": "Ministry of Rural Development",
            "state": "ALL",
            "description": "Guarantees 100 days of wage employment per year to rural households.",
            "eligibility": "Any adult rural household member willing to do unskilled manual work.",
            "benefits": "Minimum wage for 100 days per year; work within 5 km radius.",
            "min_age": "18", "max_age": "None", "gender": "ALL",
            "income_limit_inr": "None", "caste_category": "ALL",
            "occupation_tags": "Labour/Worker,BPL,Farmer",
        },
        {
            "scheme_id": "JJM001", "scheme_name": "Jal Jeevan Mission",
            "ministry": "Ministry of Jal Shakti",
            "state": "ALL",
            "description": "Tap water connection to every rural household by 2024.",
            "eligibility": "Rural households without piped water supply.",
            "benefits": "Functional household tap connection at minimum 55 LPCD.",
            "min_age": "None", "max_age": "None", "gender": "ALL",
            "income_limit_inr": "None", "caste_category": "ALL",
            "occupation_tags": "BPL,Farmer,Labour/Worker",
        },
        {
            "scheme_id": "PM_SVANidhi001", "scheme_name": "PM SVANidhi (PM Street Vendor's AtmaNirbhar Nidhi)",
            "ministry": "Ministry of Housing and Urban Affairs",
            "state": "ALL",
            "description": "Micro-credit scheme for street vendors affected by COVID-19.",
            "eligibility": "Urban street vendors with Certificate of Vending or Letter of Recommendation.",
            "benefits": "Working capital loans: Rs 10,000 initially, up to Rs 50,000 on repayment.",
            "min_age": "18", "max_age": "None", "gender": "ALL",
            "income_limit_inr": "None", "caste_category": "ALL",
            "occupation_tags": "Self-employed,Labour/Worker",
        },
    ]
    schemes_df = pd.DataFrame(_SCHEME_STUBS)

# Normalise columns
def _normalise_schemes_df(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {
        "Scheme Name": "scheme_name", "name": "scheme_name",
        "Ministry": "ministry", "Department": "ministry",
        "State": "state",
        "Description": "description", "About": "description",
        "Eligibility": "eligibility", "Eligibility Criteria": "eligibility",
        "Benefits": "benefits", "Benefit": "benefits",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    required = ["scheme_id", "scheme_name", "ministry", "state", "description",
                "eligibility", "benefits", "min_age", "max_age", "gender",
                "income_limit_inr", "caste_category", "occupation_tags"]
    for col in required:
        if col not in df.columns:
            df[col] = None

    if "scheme_id" not in df.columns or df["scheme_id"].isna().all():
        df["scheme_id"] = ["SCH_" + str(i).zfill(4) for i in range(len(df))]

    return df[required]

schemes_df = _normalise_schemes_df(schemes_df)
schemes_df = schemes_df.fillna("")
print(f"✅ {len(schemes_df)} schemes ready")
display(schemes_df.head(3))

# COMMAND ----------

# MAGIC %md ## Step 2: Write structured gov_welfare_schemes Delta table

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType

schemes_schema = StructType([StructField(c, StringType(), True) for c in [
    "scheme_id", "scheme_name", "ministry", "state", "description",
    "eligibility_raw", "benefits", "min_age", "max_age", "gender",
    "income_limit_inr", "caste_category", "occupation_tags",
]])

schemes_df_out = schemes_df.rename(columns={"eligibility": "eligibility_raw"})
for col in [c.name for c in schemes_schema.fields]:
    if col not in schemes_df_out.columns:
        schemes_df_out[col] = ""

schemes_sdf = spark.createDataFrame(
    schemes_df_out[[c.name for c in schemes_schema.fields]].astype(str),
    schema=schemes_schema,
)
(
    schemes_sdf.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(SCHEMES_TABLE)
)
spark.sql(f"ALTER TABLE {SCHEMES_TABLE} SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')")
print(f"✅ {schemes_sdf.count()} schemes written to {SCHEMES_TABLE}")

# COMMAND ----------

# MAGIC %md ## Step 3: SparkML TF-IDF Pipeline

# COMMAND ----------

from pyspark.ml.feature import Tokenizer, HashingTF, IDF, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql import functions as F

# Build a combined text column for TF-IDF
schemes_sdf_loaded = spark.table(SCHEMES_TABLE)
schemes_sdf_text = schemes_sdf_loaded.withColumn(
    "scheme_text",
    F.concat_ws(" ",
        F.col("scheme_name"),
        F.col("description"),
        F.col("eligibility_raw"),
        F.col("benefits"),
        F.col("occupation_tags"),
        F.col("state"),
        F.col("caste_category"),
    )
)

# SparkML Pipeline: Tokenizer → HashingTF → IDF
tokenizer = Tokenizer(inputCol="scheme_text", outputCol="tokens")
hashing_tf = HashingTF(inputCol="tokens", outputCol="raw_features", numFeatures=1024)
idf = IDF(inputCol="raw_features", outputCol="tfidf_features", minDocFreq=1)

pipeline = Pipeline(stages=[tokenizer, hashing_tf, idf])
model = pipeline.fit(schemes_sdf_text)
print("✅ SparkML TF-IDF pipeline fitted")

# Transform and save feature vectors
transformed = model.transform(schemes_sdf_text)

# Save TF-IDF scores table (drop the vector column — store as string for Delta compatibility)
(
    transformed.select("scheme_id", "scheme_name", "scheme_text")
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TFIDF_TABLE)
)
print(f"✅ TF-IDF metadata saved to {TFIDF_TABLE}")

# Save the Pipeline model to UC Volume
try:
    model.write().overwrite().save(MODEL_PATH)
    print(f"✅ SparkML Pipeline model saved to {MODEL_PATH}")
except Exception as e:
    print(f"⚠️  Could not save model to Volume: {e}")

# COMMAND ----------

# MAGIC %md ## Step 4: MLflow — Log TF-IDF model

# COMMAND ----------

try:
    import mlflow
    import mlflow.spark

    with mlflow.start_run(run_name="scheme_tfidf_training"):
        mlflow.log_param("num_features", 1024)
        mlflow.log_param("num_schemes", schemes_sdf.count())
        mlflow.log_param("min_doc_freq", 1)
        mlflow.spark.log_model(model, "scheme-tfidf-pipeline")
        run_id = mlflow.active_run().info.run_id
        print(f"✅ MLflow: logged TF-IDF model (run_id={run_id})")

        try:
            mlflow.register_model(
                f"runs:/{run_id}/scheme-tfidf-pipeline",
                "nyaya-sahayak-scheme-tfidf",
            )
            print("✅ MLflow: model registered as 'nyaya-sahayak-scheme-tfidf'")
        except Exception as e:
            print(f"⚠️  Model registration failed: {e}")
except Exception as e:
    print(f"⚠️  MLflow logging failed (non-fatal): {e}")

# COMMAND ----------

# MAGIC %md ## Step 5: Build corpus chunks and MERGE into legal_rag_corpus

# COMMAND ----------

corpus_rows = []
for _, row in schemes_df.iterrows():
    sid = str(row.get("scheme_id", "")).strip()
    name = str(row.get("scheme_name", "")).strip()
    ministry = str(row.get("ministry", "")).strip()
    state = str(row.get("state", "ALL")).strip()
    desc = str(row.get("description", "")).strip()
    eligibility = str(row.get("eligibility", row.get("eligibility_raw", ""))).strip()
    benefits = str(row.get("benefits", "")).strip()

    chunk_text = (
        f"Government Scheme: {name}\n"
        f"Ministry: {ministry} | Applicable in: {state}\n\n"
        f"Description: {desc}\n\n"
        f"Eligibility: {eligibility}\n\n"
        f"Benefits: {benefits}"
    )[:2000]

    corpus_rows.append({
        "chunk_id": f"SCH_{sid}" if sid else f"SCH_{hash(name) % 100000}",
        "source": "MYSCHEME_GOV_IN",
        "doc_type": "government_scheme",
        "title": name[:200],
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
stage_sdf.createOrReplaceTempView("scheme_corpus_stage")
print(f"✅ Prepared {len(corpus_rows)} scheme corpus chunks")

spark.sql(f"""
    MERGE INTO {CORPUS_TABLE} AS target
    USING scheme_corpus_stage AS source
    ON target.chunk_id = source.chunk_id
    WHEN MATCHED THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
""")
print(f"✅ MERGE complete into {CORPUS_TABLE}")

# COMMAND ----------

# MAGIC %md ## Step 6: Verify corpus distribution

# COMMAND ----------

display(spark.sql(f"""
    SELECT doc_type, COUNT(*) AS cnt
    FROM {CORPUS_TABLE}
    GROUP BY doc_type
    ORDER BY cnt DESC
"""))

display(spark.sql(f"""
    SELECT scheme_id, scheme_name, ministry, state
    FROM {SCHEMES_TABLE}
    LIMIT 10
"""))
