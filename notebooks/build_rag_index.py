# Databricks notebook source
# MAGIC %md
# MAGIC # Build RAG index (`corpus.faiss` + `chunks.parquet` + `manifest.json`)
# MAGIC
# MAGIC Run on a **cluster or serverless** notebook attached to your workspace. Prerequisites:
# MAGIC
# MAGIC - Table `main.india_legal.legal_rag_corpus` populated (ingestion notebook).
# MAGIC - **Run the next code cell first** — it `pip install`s this repo in editable mode so `import nyaya_dhwani` works. Edit `REPO_ROOT` to match your clone (Workspace sidebar → right-click repo → **Copy path**).
# MAGIC
# MAGIC Do **not** put `%restart_python` in the install cell — it restarts the kernel before later cells run and often breaks the flow. If `pip install -e` alone fails on your runtime, the next cell adds `src/` to `sys.path` so imports still work.
# MAGIC
# MAGIC Output default: `/Volumes/main/india_legal/legal_files/nyaya_index/`
# MAGIC
# MAGIC See [docs/WORKSPACE_SETUP.md](../docs/WORKSPACE_SETUP.md) for secrets and Git.

# COMMAND ----------

# --- Edit REPO_ROOT, then run this whole cell once before any `import nyaya_dhwani` ---
# Copy path from Workspace sidebar (right-click repo folder → Copy path). Examples:
#   /Workspace/Repos/you@domain.com/nyaya-dhwani-hackathon
#   /Workspace/Users/you@domain.com/nyaya-dhwani-hackathon
import os
import subprocess
import sys

REPO_ROOT = "/Workspace/Users/11avanshdhar@gmail.com/nyaya-dhwani-hackathon"  # noqa: E501 — edit me

_src = os.path.join(REPO_ROOT, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

# RAG stack in *this* kernel. Pin faiss-cpu<1.8 + numpy<2 so databricks-connect stays happy
# (faiss 1.8+ needs NumPy 2's numpy._core; databricks-connect requires numpy<2).
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "numpy>=1.24,<2",
        "pandas>=2.0,<3",
        "faiss-cpu>=1.7.0,<1.8",
        "pyarrow>=14",
        "sentence-transformers>=2.2.0",
    ]
)
print("✅ pip: numpy 1.x, pandas<3, faiss-cpu 1.7.x, pyarrow, sentence-transformers")

try:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-e",
            f"{REPO_ROOT}[rag,rag_embed]",
        ]
    )
    print(f"✅ pip install -e {REPO_ROOT}[rag,rag_embed]")
except subprocess.CalledProcessError as e:
    print(f"⚠️  pip install -e failed ({e}); using sys.path only → {_src}")

try:
    import nyaya_dhwani
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        f"Cannot import nyaya_dhwani — check REPO_ROOT exists and contains src/nyaya_dhwani. "
        f"Tried sys.path[0]={_src!r}. Original error: {e}"
    ) from e

import importlib

_faiss = importlib.import_module("faiss")
print("✅ import faiss OK")

print("✅ import nyaya_dhwani →", os.path.dirname(nyaya_dhwani.__file__))

# COMMAND ----------

CATALOG = "workspace"  # Changed from 'main' to match available catalogs
SCHEMA = "india_legal"
TABLE = "legal_rag_corpus"
OUT_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/legal_files/nyaya_index"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

pdf = spark.table(f"{CATALOG}.{SCHEMA}.{TABLE}").select(
    "chunk_id", "source", "doc_type", "title", "text"
).toPandas()

print(pdf.shape)
pdf.head(3)

# COMMAND ----------

import importlib.util

if importlib.util.find_spec("nyaya_dhwani") is None:
    raise RuntimeError(
        "Run the install cell above first and set REPO_ROOT to your repo path "
        "(Workspace sidebar → right-click nyaya-dhwani-hackathon → Copy path)."
    )

from nyaya_dhwani.embedder import SentenceEmbedder
from nyaya_dhwani.index_builder import save_rag_artifacts

texts = pdf["text"].fillna("").astype(str).tolist()
embedder = SentenceEmbedder(model_name=EMBED_MODEL, normalize=True)
emb = embedder.encode(texts)
print(emb.shape)

import os
os.makedirs(OUT_DIR, exist_ok=True)

manifest = save_rag_artifacts(
    OUT_DIR,
    embeddings=emb,
    chunks_df=pdf,
    embedding_model=EMBED_MODEL,
    catalog=CATALOG,
    schema=SCHEMA,
    source_table=TABLE,
)
print(manifest.to_json())

# COMMAND ----------

# Smoke test load
from nyaya_dhwani.retrieval import CorpusIndex

ci = CorpusIndex.load(OUT_DIR)
q = embedder.encode(["What is theft under BNS?"])
display(ci.search(q, k=5))