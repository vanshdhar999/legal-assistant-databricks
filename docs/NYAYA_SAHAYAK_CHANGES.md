# Nyaya Sahayak — Feature Expansion Summary

> **Original project:** Nyaya Dhwani (multilingual BNS/IPC RAG chatbot)
> **Expanded to:** Nyaya Sahayak — Governance & Access to Justice

---

## What Changed at a Glance

| Category | Before | After |
|----------|--------|-------|
| App tabs | 1 (Legal Q&A chat) | 5 (Legal Q&A, BNS Explainer, Scheme Finder, IPC↔BNS, About) |
| Datasets | BNS 2023, BNS↔IPC mapping | + Constitution of India, IPC 1860 full text, Government Schemes |
| Databricks components | Delta, Vector Search, AI Gateway, Apps | + SparkML TF-IDF, MLflow Experiment Tracking, MLflow Model Registry, Delta MERGE upserts, CDF on all tables |
| Source modules | 10 | 14 (+4 new, 3 modified) |
| Notebooks | 4 | 8 (+4 new) |

---

## New Files

### Source Modules (`src/nyaya_dhwani/`)

#### `mlflow_logger.py` — MLflow Experiment Tracking
- `RAGQueryLogger` context manager that wraps every query turn as a nested MLflow run
- Logs: query preview, language, retrieval latency, LLM latency, number of chunks retrieved, top retrieval score
- **Graceful degradation**: all methods are no-ops when `MLFLOW_TRACKING_URI` is not set or MLflow is not installed — the app never crashes
- Used by all four feature tabs

#### `bns_explainer.py` — BNS Explainer
- Specialised RAG pipeline for the Bharatiya Nyaya Sanhita (BNS) 2023
- Re-ranks retrieved chunks to prioritise `criminal_law` and `law_mapping` doc types
- Custom system prompt that forces structured output: plain English explanation → Hindi key terms (Devanagari) → IPC predecessor → punishment/penalty → practical impact
- `explain_bns_section(query, retriever)` → dict with answer, citations, detected sections, latency metrics
- `format_bns_response(result)` → Gradio-compatible markdown

#### `scheme_checker.py` — Government Scheme Eligibility Checker
- `UserProfile` dataclass: state, age, gender, annual income (INR), caste category, occupation
- **Hard filter**: Databricks SQL query on `gov_welfare_schemes` Delta table via `WorkspaceClient().statement_execution` — filters on age range, income cap, gender, caste, state
- **Soft match**: RAG retrieval on `government_scheme` corpus chunks when SQL warehouse not available
- **LLM explanation**: Llama Maverick explains matched schemes in plain language for rural users
- Falls back to RAG-only if `NYAYA_SQL_WAREHOUSE_ID` env var not set
- `check_eligibility(profile, retriever)` → `(candidates_df, explanation_markdown)`

#### `ipc_bns_compare.py` — IPC → BNS Comparison Tool
- Parses IPC section input (`"302"`, `"IPC 302"`, `"Section 302"`)
- **Priority 1**: Databricks SQL JOIN across `ipc_sections` + `bns_ipc_mapping` + `bns_sections` tables
- **Priority 2**: RAG fallback using FAISS/Vector Search
- **Priority 3**: LLM-only answer with RAG context
- `compute_text_diff(ipc_text, bns_text)` → `difflib.unified_diff` output (≤50 lines)
- `llm_comparative_analysis(data)` → structured analysis: what changed, what stayed, punishment changes, practical impact
- `format_comparison_response(result)` → four strings for the Gradio UI (IPC markdown, BNS markdown, analysis markdown, diff text)

---

### Notebooks

#### `02_ingest_constitution.py` — Constitution of India Ingestion
- Downloads Constitution from HuggingFace (`pbeukema/indian_constitution`, with fallback sources)
- Converts to Spark DataFrame and writes `workspace.india_legal.constitution_articles` Delta table
- Enables Change Data Feed (CDF) on the table for Vector Search sync
- Builds article-level corpus chunks (`chunk_id = CONST_A{num}`, `doc_type = "constitution"`)
- Upserts into `legal_rag_corpus` using **Delta MERGE** (preserves existing BNS/IPC rows)

#### `03_ingest_ipc_full.py` — IPC Full Text Ingestion
- Downloads IPC 1860 section text from HuggingFace (multiple fallback sources + hardcoded stubs)
- Cross-references with existing `bns_ipc_mapping` table to add BNS equivalent to each IPC chunk
- Writes `workspace.india_legal.ipc_sections` Delta table with CDF enabled
- Corpus chunks enriched with BNS cross-reference note (`doc_type = "criminal_law_ipc"`)
- Spark analysis: prints IPC coverage statistics (mapped vs. repealed sections)
- Delta MERGE upsert into `legal_rag_corpus`

#### `04_ingest_gov_schemes.py` — Government Schemes + SparkML TF-IDF
- Fetches government welfare schemes from HuggingFace (with hardcoded stubs as fallback: PM-KISAN, PMAY, MUDRA, Beti Bachao, Ayushman Bharat, MGNREGA, etc.)
- Writes `workspace.india_legal.gov_welfare_schemes` Delta table with structured eligibility columns
- **SparkML Pipeline**: `Tokenizer → HashingTF → IDF` fitted on scheme text corpus
  - Saves model to `/Volumes/workspace/india_legal/legal_files/scheme_tfidf_model`
  - Saves `scheme_tfidf_scores` Delta table
  - **MLflow**: logs and registers model as `nyaya-sahayak-scheme-tfidf` in Model Registry
- Corpus chunks with `doc_type = "government_scheme"` merged into `legal_rag_corpus`

#### `05_mlflow_rag_evaluation.py` — RAG Benchmark Evaluation
- Sets up MLflow experiment `/nyaya-dhwani/rag-evaluation`
- **Registers embedding model** (`all-MiniLM-L6-v2`) in MLflow Model Registry as `nyaya-dhwani-embedder`
- Loads benchmark questions from `tests/benchmark_questions.json`
- Runs each question through full RAG pipeline, logging per-question metrics as nested MLflow runs:
  - retrieval latency, LLM latency, total latency, num chunks, top retrieval score, keyword hit rate
- Computes and logs aggregate metrics by language
- Saves results to `workspace.india_legal.rag_eval_results` Delta table

---

## Modified Files

### `src/nyaya_dhwani/keyword_boost.py`
- Added `_ARTICLE_RE` regex pattern for Constitution article references (`"Article 21"`, `"Art. 14"`, `"Article 370"`)
- Added `detect_article_references(query)` function
- `boost_with_keywords()` now also boosts Constitution chunks when article references are detected in the query

### `src/nyaya_dhwani/retriever.py`
- `Retriever` Protocol: added `doc_type_filter: str | None = None` parameter to `search()`
- `FaissRetriever.search()`: post-filters results by `doc_type` when `doc_type_filter` is provided
- `FallbackRetriever.search()`: passes `doc_type_filter` through to both primary and fallback retrievers
- Fully backward-compatible (default is `None`, existing call sites unchanged)

### `src/nyaya_dhwani/vs_retriever.py`
- `VectorSearchRetriever.search()`: added `doc_type_filter` parameter
- When `doc_type_filter` is set, uses it as the VS filter instead of the automatic `law_mapping` filter for section references
- Main query now passes `filters_json` when `doc_type_filter` is provided
- The law_mapping pre-filter for section references is skipped when `doc_type_filter` is explicitly set

### `app/main.py`
- Renamed app: **Nyaya Dhwani → Nyaya Sahayak**
- Restructured from single chat column to `gr.Tabs` with 5 tabs
- Added graceful `try/except` imports for all new feature modules (app works even if modules fail to load)
- Added `RAGQueryLogger` wrapping around `run_turn()` and all new turn handlers
- New turn functions:
  - `run_bns_turn(query, history, lang)` — calls BNS explainer, returns updated chatbot history
  - `run_scheme_check(state, age, gender, income, caste, occupation, lang)` — returns markdown explanation + status
  - `run_ipc_bns_compare(ipc_input, lang)` — returns IPC text, BNS text, analysis, diff
- Added `BNS_TOPIC_SEEDS` quick-pick topics for BNS Explainer tab
- `on_begin()` now reveals `main_col` (with tabs) instead of `chat_col`

### `pyproject.toml`
- Added `[mlflow]` optional extra: `mlflow>=2.12.0`
- Added `[ingest]` optional extra: `datasets>=2.14.0` + `mlflow>=2.12.0` (for ingestion notebooks)

### `requirements.txt`
- Added `mlflow>=2.12.0`
- Pinned `databricks-sdk>=0.20` (required for `statement_execution` API)

---

## New Databricks Components (vs. original)

| Component | How it's used |
|-----------|--------------|
| **SparkML** | `HashingTF + IDF` Pipeline trained on scheme corpus in notebook 04; saved to UC Volume |
| **MLflow Experiments** | Every query turn logged as a nested run under `/nyaya-dhwani/rag-queries` |
| **MLflow Model Registry** | Embedder (`nyaya-dhwani-embedder`) and TF-IDF Pipeline (`nyaya-sahayak-scheme-tfidf`) registered |
| **Delta MERGE** | All new ingest notebooks use `MERGE INTO ... WHEN MATCHED / WHEN NOT MATCHED` for safe upserts |
| **Change Data Feed** | Enabled on `constitution_articles`, `gov_welfare_schemes`, `ipc_sections` for future VS sync |
| **Databricks SQL** | `statement_execution.execute_statement()` for scheme eligibility hard filter (SQL warehouse) |

---

## New Environment Variables

| Variable | Purpose | Required? |
|----------|---------|-----------|
| `NYAYA_SQL_WAREHOUSE_ID` | SQL warehouse ID for scheme eligibility hard filter | Optional — falls back to RAG-only |
| `MLFLOW_TRACKING_URI` | MLflow tracking server (auto-configured on Databricks) | Optional — logging is no-op if unset |
| `NYAYA_MLFLOW_EXPERIMENT` | Override default experiment name `/nyaya-dhwani/rag-queries` | Optional |

---

## New Delta Tables

| Table | Source | Key Columns |
|-------|--------|-------------|
| `workspace.india_legal.constitution_articles` | HuggingFace / stub | `article_number`, `part`, `article_title`, `article_text` |
| `workspace.india_legal.ipc_sections` | HuggingFace / stub | `section_number`, `section_title`, `section_text`, `bns_equivalent`, `mapping_status` |
| `workspace.india_legal.gov_welfare_schemes` | HuggingFace / stub | `scheme_id`, `scheme_name`, `ministry`, `state`, `eligibility_raw`, `benefits`, `min_age`, `max_age`, `gender`, `income_limit_inr`, `caste_category`, `occupation_tags` |
| `workspace.india_legal.scheme_tfidf_scores` | SparkML transform | `scheme_id`, `scheme_name`, `scheme_text` |
| `workspace.india_legal.rag_eval_results` | Notebook 05 | `idx`, `language`, `question`, `retrieval_ms`, `llm_ms`, `num_chunks`, `llm_ok`, `keyword_hit_rate` |

---

---

# Deployment Workflow on Databricks

## Prerequisites

Before running anything, ensure:
- Databricks workspace is accessible at `https://dbc-6651e87a-25a5.cloud.databricks.com`
- You have authenticated: `databricks auth login --host https://dbc-6651e87a-25a5.cloud.databricks.com --profile free-aws`
- Secret scope `nyaya-dhwani` exists with `sarvam_api_key` set
- The repo is cloned into your Databricks Workspace (Repos)
- A cluster (or serverless) is available for running notebooks

---

## Step 1: One-Time Secret & Scope Setup

```bash
# Authenticate locally
databricks auth login --host https://dbc-6651e87a-25a5.cloud.databricks.com --profile free-aws
export DATABRICKS_CONFIG_PROFILE=free-aws

# Create secret scope (skip if already exists)
databricks secrets create-scope nyaya-dhwani

# Add required secrets
databricks secrets put-secret nyaya-dhwani sarvam_api_key    # Sarvam API key
databricks secrets put-secret nyaya-dhwani hf_token          # HuggingFace token (for dataset downloads)
```

---

## Step 2: Run Ingestion Notebooks (in order)

Open each notebook in the Databricks Workspace UI and attach it to a cluster.
**Edit `REPO_ROOT`** at the top of each notebook to match your Workspace Repos path.

### 2a. Original BNS + IPC Mapping Ingestion

```
Notebook: notebooks/india_legal_policy_ingest.py
```

- Creates schema `workspace.india_legal` and volume `legal_files`
- Ingests BNS sections and BNS↔IPC mapping
- Writes `workspace.india_legal.legal_rag_corpus` (baseline corpus)

### 2b. Constitution of India

```
Notebook: notebooks/02_ingest_constitution.py
```

- Downloads Constitution articles from HuggingFace
- Writes `workspace.india_legal.constitution_articles`
- Merges article chunks into `legal_rag_corpus`

### 2c. IPC Full Text

```
Notebook: notebooks/03_ingest_ipc_full.py
```

- Downloads IPC 1860 section text
- Cross-references with existing BNS mapping
- Writes `workspace.india_legal.ipc_sections`
- Merges IPC chunks into `legal_rag_corpus`

### 2d. Government Schemes + SparkML TF-IDF

```
Notebook: notebooks/04_ingest_gov_schemes.py
```

- Downloads government welfare scheme data
- Trains and saves SparkML TF-IDF Pipeline
- Writes `workspace.india_legal.gov_welfare_schemes`
- Logs TF-IDF model to MLflow Model Registry
- Merges scheme chunks into `legal_rag_corpus`

After all four notebooks complete, verify the corpus distribution:

```sql
SELECT doc_type, COUNT(*) AS cnt
FROM workspace.india_legal.legal_rag_corpus
GROUP BY doc_type
ORDER BY cnt DESC;
```

Expected doc types: `criminal_law`, `law_mapping`, `constitution`, `criminal_law_ipc`, `government_scheme`

---

## Step 3: Build the FAISS Index

```
Notebook: notebooks/build_rag_index.py
```

- Reads full `legal_rag_corpus` table
- Embeds all chunks with `sentence-transformers/all-MiniLM-L6-v2`
- Saves `corpus.faiss` + `chunks.parquet` + `manifest.json` to:
  `/Volumes/workspace/india_legal/legal_files/nyaya_index/`

---

## Step 4: Set Up Databricks Vector Search (Primary Retrieval Backend)

```
Notebook: notebooks/setup_vector_search.py
```

- Creates Vector Search endpoint `nyaya_vs_endpoint` (if not exists)
- Creates Delta Sync index on `legal_rag_corpus` (embedding model: `databricks-bge-large-en`)
- **Trigger a sync** to pick up all new doc types (constitution, IPC, schemes) added in Step 2

To trigger a manual sync from the UI:
> **Machine Learning** → **Vector Search** → `nyaya_vs_endpoint` → `legal_rag_corpus_index` → **Sync**

---

## Step 5: Run MLflow Evaluation (Optional but Recommended)

```
Notebook: notebooks/05_mlflow_rag_evaluation.py
```

- Registers the embedding model in MLflow Model Registry
- Runs benchmark questions through the full pipeline
- Logs per-question metrics to experiment `/nyaya-dhwani/rag-evaluation`
- Saves results to `workspace.india_legal.rag_eval_results`

View results in: **Experiments** → `/nyaya-dhwani/rag-evaluation`

---

## Step 6: Configure the SQL Warehouse (Optional — for Scheme Finder)

The Scheme Finder tab uses a direct SQL query for hard eligibility filtering.
To enable it:

1. Go to **SQL** → **SQL Warehouses** in your workspace
2. Start a warehouse (Serverless recommended) and copy its ID
3. Add it to your App environment:

```yaml
# In app.yaml, add:
env:
  - name: NYAYA_SQL_WAREHOUSE_ID
    value: "<your-warehouse-id>"
```

> Without this, the Scheme Finder still works via RAG-only matching.

---

## Step 7: Deploy via Databricks Apps

### Option A: Via the UI

1. Go to **Compute** → **Apps** → **Create App**
2. Connect your Git repo (or Workspace Repo path)
3. Set entrypoint: `python app/main.py`
4. The `app.yaml` file configures all environment variables automatically
5. Add a secret resource: scope `nyaya-dhwani`, key `sarvam_api_key`, env var `SARVAM_API_KEY`
6. Click **Deploy**

### Option B: Via CLI

```bash
# From your local machine with the CLI authenticated
databricks apps deploy nyaya-sahayak \
  --source-code-path /Workspace/Users/<you>/nyaya-dhwani-hackathon
```

### Required App Environment Variables (`app.yaml`)

```yaml
env:
  - name: LLM_OPENAI_BASE_URL
    value: "https://dbc-6651e87a-25a5.cloud.databricks.com/serving-endpoints"
  - name: LLM_MODEL
    value: "databricks-llama-4-maverick"
  - name: NYAYA_RETRIEVAL_BACKEND
    value: "vector_search"
  - name: NYAYA_VS_ENDPOINT_NAME
    value: "nyaya_vs_endpoint"
  - name: NYAYA_VS_INDEX_NAME
    value: "workspace.india_legal.legal_rag_corpus_index"
  - name: NYAYA_INDEX_DIR
    value: "/Volumes/workspace/india_legal/legal_files/nyaya_index"
  - name: NYAYA_SQL_WAREHOUSE_ID
    value: "<your-sql-warehouse-id>"   # Optional
```

---

## Step 8: Verify the Deployed App

Once the app is running, check each tab:

| Tab | Smoke Test |
|-----|-----------|
| ⚖️ Legal Q&A | Ask: *"What is the procedure to file an FIR?"* |
| 📖 BNS Explainer | Ask: *"Explain BNS Section 303 on theft"* |
| 🏛️ Scheme Finder | Set: State=Maharashtra, Age=28, Gender=Female, Income=180000, Caste=OBC → Find Schemes |
| ⚡ IPC ↔ BNS | Enter: `302` → Compare |

---

## Full Run Order Summary

```
[One-time setup]
  databricks auth login
  databricks secrets create-scope nyaya-dhwani
  databricks secrets put-secret nyaya-dhwani sarvam_api_key
  databricks secrets put-secret nyaya-dhwani hf_token

[Ingestion — run notebooks in order on a cluster]
  1. india_legal_policy_ingest.py       ← BNS + mapping (existing)
  2. 02_ingest_constitution.py          ← Constitution of India (new)
  3. 03_ingest_ipc_full.py             ← IPC 1860 full text (new)
  4. 04_ingest_gov_schemes.py           ← Government schemes + SparkML (new)

[Index building]
  5. build_rag_index.py                 ← Rebuild FAISS index
  6. setup_vector_search.py             ← Trigger VS sync (or manual sync from UI)

[Evaluation — optional]
  7. 05_mlflow_rag_evaluation.py        ← Benchmark + model registry

[Deploy]
  8. Databricks Apps → Create → connect repo → deploy
```

---

## Local Development

```bash
cd /path/to/nyaya-dhwani-hackathon

# Install all extras
pip install -e ".[dev,rag,rag_embed,app,mlflow]"

# Set environment variables
cp .env.example .env
# Fill in: SARVAM_API_KEY, LLM_OPENAI_BASE_URL, LLM_MODEL, DATABRICKS_TOKEN
export $(grep -v '^#' .env | xargs)

# Run the app locally (FAISS backend, no Vector Search needed)
python app/main.py
```

For local development, set `NYAYA_RETRIEVAL_BACKEND=faiss` and ensure the FAISS index is built first.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Scheme Finder shows RAG-only results | Set `NYAYA_SQL_WAREHOUSE_ID` in app.yaml |
| MLflow logging silent | Set `MLFLOW_TRACKING_URI` or run on Databricks (auto-configured) |
| BNS Explainer tab shows "module unavailable" | Run `pip install -e ".[rag,rag_embed,app]"` |
| IPC→BNS diff is empty | Run notebook 03 to ingest IPC full text first |
| Constitution articles not retrieved | Run notebook 02, then rebuild FAISS index |
| Vector Search returns empty | Trigger a manual sync after ingesting new doc types |
