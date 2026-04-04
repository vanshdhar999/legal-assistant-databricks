# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 05: MLflow RAG Evaluation & Model Registry
# MAGIC
# MAGIC Runs benchmark questions through the full RAG pipeline, logs metrics per question and
# MAGIC language to MLflow, and registers the embedding model in the MLflow Model Registry.
# MAGIC
# MAGIC **Prerequisites:** Notebooks 01–04 run; FAISS index built (notebook `build_rag_index.py`).
# MAGIC
# MAGIC **What this does:**
# MAGIC 1. Sets up MLflow experiment `/nyaya-dhwani/rag-evaluation`
# MAGIC 2. Registers `all-MiniLM-L6-v2` embedder in Model Registry as `nyaya-dhwani-embedder`
# MAGIC 3. Runs benchmark questions from `tests/benchmark_questions.json`
# MAGIC 4. Logs per-question: retrieval latency, LLM latency, num chunks retrieved, language
# MAGIC 5. Computes aggregate metrics per language

# COMMAND ----------

import os
import subprocess
import sys
import json
import time
import logging

logging.basicConfig(level=logging.INFO)

REPO_ROOT = "/Workspace/Users/11avanshdhar@gmail.com/nyaya-dhwani-hackathon"  # Edit me

_src = os.path.join(REPO_ROOT, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "numpy>=1.24,<2", "pandas>=2.0,<3",
    "faiss-cpu>=1.7.0,<1.8", "pyarrow>=14",
    "sentence-transformers>=2.2.0",
    f"{REPO_ROOT}[rag,rag_embed]",
])
print("✅ RAG stack ready")

# COMMAND ----------

# MAGIC %md ## Step 1: Setup MLflow Experiment

# COMMAND ----------

import mlflow

EXPERIMENT_NAME = "/nyaya-dhwani/rag-evaluation"
mlflow.set_experiment(EXPERIMENT_NAME)
exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
print(f"✅ MLflow experiment: {EXPERIMENT_NAME} (id={exp.experiment_id if exp else 'new'})")

# COMMAND ----------

# MAGIC %md ## Step 2: Register Embedding Model in Model Registry

# COMMAND ----------

import mlflow.sentence_transformers
from nyaya_dhwani.embedder import SentenceEmbedder

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

with mlflow.start_run(run_name="embedder-registration") as run:
    embedder = SentenceEmbedder(model_name=EMBED_MODEL, normalize=True)

    # Log model metadata
    mlflow.log_param("model_name", EMBED_MODEL)
    mlflow.log_param("embedding_dim", 384)
    mlflow.log_param("normalize", True)

    # Smoke test embed
    test_emb = embedder.encode(["IPC Section 302"])
    mlflow.log_metric("embedding_dim_actual", test_emb.shape[1])

    # Log as sentence_transformers model
    try:
        mlflow.sentence_transformers.log_model(
            embedder._model,
            artifact_path="embedder",
            registered_model_name="nyaya-dhwani-embedder",
        )
        print(f"✅ Embedder logged and registered (run_id={run.info.run_id})")
    except Exception as e:
        print(f"⚠️  sentence_transformers log_model failed: {e}")
        # Fallback: log as pyfunc
        mlflow.log_artifact(
            os.path.join(_src, "nyaya_dhwani", "embedder.py"),
            artifact_path="embedder_source",
        )
        print("  Fallback: logged embedder source as artifact")

# COMMAND ----------

# MAGIC %md ## Step 3: Load RAG Retriever + LLM

# COMMAND ----------

from nyaya_dhwani.retriever import get_retriever
from nyaya_dhwani.llm_client import chat_completions, extract_assistant_text, rag_user_message

INDEX_DIR = os.environ.get(
    "NYAYA_INDEX_DIR",
    f"/Volumes/workspace/india_legal/legal_files/nyaya_index",
)
os.environ.setdefault("NYAYA_RETRIEVAL_BACKEND", "faiss")
os.environ.setdefault("NYAYA_INDEX_DIR", INDEX_DIR)

retriever = get_retriever()
print(f"✅ Retriever: {type(retriever).__name__}")

SYSTEM_PROMPT = (
    "You are Nyaya Dhwani, an Indian legal information assistant. "
    "Answer using the Context when relevant. Cite Acts or sections. "
    "Respond in English."
)

# COMMAND ----------

# MAGIC %md ## Step 4: Load Benchmark Questions

# COMMAND ----------

BENCHMARK_PATH = os.path.join(REPO_ROOT, "tests", "benchmark_questions.json")

with open(BENCHMARK_PATH) as f:
    benchmark = json.load(f)

# benchmark is a list of dicts with keys: question, language, expected_keywords (optional)
if isinstance(benchmark, dict):
    # Handle both list and dict formats
    questions = benchmark.get("questions", list(benchmark.values()))
else:
    questions = benchmark

print(f"✅ Loaded {len(questions)} benchmark questions")
print("Sample:", questions[0] if questions else "(none)")

# COMMAND ----------

# MAGIC %md ## Step 5: Run Evaluation Loop

# COMMAND ----------

EVAL_K = 7  # chunks to retrieve per question
results = []

with mlflow.start_run(run_name="rag-benchmark-eval") as parent_run:
    mlflow.log_param("benchmark_file", BENCHMARK_PATH)
    mlflow.log_param("retrieval_k", EVAL_K)
    mlflow.log_param("num_questions", len(questions))

    for i, item in enumerate(questions):
        if isinstance(item, str):
            q_text = item
            q_lang = "en"
            expected_keywords = []
        elif isinstance(item, dict):
            q_text = item.get("question", item.get("query", str(item)))
            q_lang = item.get("language", "en")
            expected_keywords = item.get("expected_keywords", [])
        else:
            continue

        with mlflow.start_run(run_name=f"q_{i:03d}_{q_lang}", nested=True):
            mlflow.log_param("question_idx", i)
            mlflow.log_param("language", q_lang)
            mlflow.log_param("question_preview", q_text[:100])

            # Retrieval
            t0 = time.monotonic()
            try:
                chunks_df = retriever.search(q_text, k=EVAL_K)
                retrieval_ms = int((time.monotonic() - t0) * 1000)
                num_chunks = len(chunks_df)
                top_score = float(chunks_df["score"].iloc[0]) if "score" in chunks_df.columns and not chunks_df.empty else 0.0
                retrieval_ok = True
            except Exception as e:
                retrieval_ms = int((time.monotonic() - t0) * 1000)
                num_chunks = 0
                top_score = 0.0
                retrieval_ok = False
                print(f"  ⚠️  Q{i}: retrieval failed: {e}")

            mlflow.log_metric("retrieval_latency_ms", retrieval_ms)
            mlflow.log_metric("num_chunks_retrieved", num_chunks)
            mlflow.log_metric("top_retrieval_score", top_score)

            # LLM generation
            llm_ok = False
            llm_ms = 0
            answer = ""
            keyword_hits = 0

            if retrieval_ok and num_chunks > 0:
                texts = chunks_df["text"].tolist()
                user_msg = rag_user_message([str(t) for t in texts], q_text)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ]
                t1 = time.monotonic()
                try:
                    raw = chat_completions(messages, max_tokens=512, temperature=0.2)
                    answer = extract_assistant_text(raw)
                    llm_ms = int((time.monotonic() - t1) * 1000)
                    llm_ok = True
                    # Keyword hit rate
                    answer_lower = answer.lower()
                    keyword_hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
                except Exception as e:
                    llm_ms = int((time.monotonic() - t1) * 1000)
                    print(f"  ⚠️  Q{i}: LLM failed: {e}")

            mlflow.log_metric("llm_latency_ms", llm_ms)
            mlflow.log_metric("total_latency_ms", retrieval_ms + llm_ms)
            mlflow.log_metric("llm_ok", int(llm_ok))
            if expected_keywords:
                kw_rate = keyword_hits / len(expected_keywords)
                mlflow.log_metric("keyword_hit_rate", kw_rate)

            results.append({
                "idx": i,
                "language": q_lang,
                "question": q_text[:80],
                "retrieval_ms": retrieval_ms,
                "llm_ms": llm_ms,
                "num_chunks": num_chunks,
                "top_score": top_score,
                "llm_ok": llm_ok,
                "keyword_hit_rate": keyword_hits / max(len(expected_keywords), 1),
            })

        if (i + 1) % 5 == 0:
            print(f"  Progress: {i+1}/{len(questions)} questions evaluated")

    # Aggregate metrics
    import pandas as pd
    results_df = pd.DataFrame(results)

    mlflow.log_metric("avg_retrieval_latency_ms", results_df["retrieval_ms"].mean())
    mlflow.log_metric("avg_llm_latency_ms", results_df["llm_ms"].mean())
    mlflow.log_metric("avg_total_latency_ms", (results_df["retrieval_ms"] + results_df["llm_ms"]).mean())
    mlflow.log_metric("avg_chunks_retrieved", results_df["num_chunks"].mean())
    mlflow.log_metric("llm_success_rate", results_df["llm_ok"].mean())
    mlflow.log_metric("avg_keyword_hit_rate", results_df["keyword_hit_rate"].mean())

    print(f"\n✅ Evaluation complete — {len(results)} questions")
    print(f"   Avg retrieval latency : {results_df['retrieval_ms'].mean():.0f} ms")
    print(f"   Avg LLM latency       : {results_df['llm_ms'].mean():.0f} ms")
    print(f"   LLM success rate      : {results_df['llm_ok'].mean():.1%}")
    print(f"   Avg keyword hit rate  : {results_df['keyword_hit_rate'].mean():.1%}")

# COMMAND ----------

# MAGIC %md ## Step 6: Per-language Breakdown

# COMMAND ----------

display(
    results_df.groupby("language").agg(
        count=("idx", "count"),
        avg_retrieval_ms=("retrieval_ms", "mean"),
        avg_llm_ms=("llm_ms", "mean"),
        avg_chunks=("num_chunks", "mean"),
        avg_kw_hit_rate=("keyword_hit_rate", "mean"),
    ).reset_index().round(1)
)

# COMMAND ----------

# MAGIC %md ## Step 7: Save results to Delta for analysis

# COMMAND ----------

results_sdf = spark.createDataFrame(results_df)
(
    results_sdf.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.india_legal.rag_eval_results")
)
print("✅ Results saved to workspace.india_legal.rag_eval_results")
display(results_sdf)
