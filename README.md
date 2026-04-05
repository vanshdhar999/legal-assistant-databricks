# MULIA

Multilingual Legal Information Assistant for Indian law.

MULIA is a Databricks-native legal assistant that combines:
- curated legal corpora (BNS, IPC, Constitution, Government Schemes),
- retrieval-augmented generation (RAG),
- multilingual translation and speech support,
- and a Gradio web app deployed on Databricks Apps.

Not legal advice. This project provides general legal information only.

## What You Can Do

- Ask legal questions in multiple Indian languages.
- Get grounded answers with retrieved context citations.
- Use dedicated tools for:
  - BNS section explanation,
  - IPC to BNS comparison,
  - scheme eligibility checks.
- Ingest and refresh legal datasets through reproducible notebooks.

## Architecture Overview

User query flow:

1. User asks a question (text or voice).
2. Optional translation to English for retrieval.
3. Retriever fetches top legal chunks from corpus.
4. LLM generates grounded answer.
5. Optional translation back to selected language.
6. Optional TTS playback.

Retrieval backend:

- Primary: Databricks Vector Search (if configured).
- Fallback: FAISS index stored in Unity Catalog Volume.

## Repository Structure

| Path | Purpose |
|---|---|
| [app/main.py](app/main.py) | Gradio frontend and app orchestration |
| [app.yaml](app.yaml) | Databricks App runtime command + env vars |
| [src/nyaya_dhwani](src/nyaya_dhwani) | Core package (retrieval, LLM client, translators, feature modules) |
| [notebooks](notebooks) | Data ingestion, indexing, vector search setup, evaluation |
| [tests](tests) | Unit tests |
| [docs](docs) | Deep-dive docs, deployment details, user/developer guides |

## Deployment and Reproducibility (End-to-End)

This section is the canonical runbook for recreating the project from scratch.

### 1. Prerequisites

- Databricks workspace.
- Databricks CLI installed and authenticated.
- Permissions to create/read secret scope and Unity Catalog objects.
- Sarvam API key (optional but recommended for multilingual voice/text UX).

### 2. Authenticate Databricks CLI

```bash
databricks auth login --host https://<your-workspace>.cloud.databricks.com --profile mulia
export DATABRICKS_CONFIG_PROFILE=mulia
databricks current-user me
```

### 3. Create and Populate Secrets

```bash
databricks secrets create-scope nyaya-dhwani --profile mulia
databricks secrets put-secret nyaya-dhwani sarvam_api_key --profile mulia
```

Optional benchmark/data keys can be added similarly.

### 4. Run Data Ingestion Notebooks

Run these notebooks in order on Databricks compute:

1. [notebooks/india_legal_policy_ingest.py](notebooks/india_legal_policy_ingest.py)
2. [notebooks/02_ingest_constitution.py](notebooks/02_ingest_constitution.py)
3. [notebooks/03_ingest_ipc_full.py](notebooks/03_ingest_ipc_full.py)
4. [notebooks/04_ingest_gov_schemes.py](notebooks/04_ingest_gov_schemes.py)

Expected outcome:

- populated legal corpus table,
- domain-specific Delta tables,
- governed data in Unity Catalog.

### 5. Build FAISS Index (Fallback Retrieval)

Run:

- [notebooks/build_rag_index.py](notebooks/build_rag_index.py)

Expected artifacts in configured volume path:

- manifest.json
- corpus.faiss
- chunks.parquet

### 6. (Recommended) Setup Databricks Vector Search

Run:

- [notebooks/setup_vector_search.py](notebooks/setup_vector_search.py)

This creates endpoint/index and syncs from the Delta corpus table.

### 7. Configure App Runtime

Verify [app.yaml](app.yaml):

- command points to app/main.py,
- LLM base/model values are correct for your workspace,
- retrieval backend/env vars are set.

Current defaults include:

- NYAYA_RETRIEVAL_BACKEND=vector_search,
- FAISS path fallback via NYAYA_INDEX_DIR.

### 8. Create and Deploy Databricks App

In Databricks UI:

1. Compute -> Apps -> Create
2. Connect this Git repository
3. Confirm environment variables from app.yaml
4. Deploy

### 9. Grant Required Permissions to App Service Principal

At minimum:

- CAN_QUERY on serving endpoint (LLM).
- READ on Unity Catalog volume containing FAISS index.
- READ on secret scope nyaya-dhwani.

If using Vector Search, also grant:

- USE CATALOG, USE SCHEMA, SELECT on source table,
- access to Vector Search endpoint and index entity.

### 10. Smoke Test After Deployment

Run these checks in the app:

1. Ask a legal question in English.
2. Ask the same in one non-English language.
3. Test BNS Explainer tab.
4. Test IPC to BNS tab.
5. Test Scheme Finder tab.

If any retrieval issue occurs, app should still serve answers via FAISS fallback.

## Local Development

Install project with extras:

```bash
pip install -e ".[dev,rag,rag_embed,app,llm_openai,mlflow]"
```

Run app:

```bash
python app/main.py
```

One-command bootstrap (recommended):

```bash
./run_local.sh
```

Optional fast restart without reinstall:

```bash
./run_local.sh --no-install
```

The script auto-loads `.env.local` if present and validates required variables.

For local LLM calls, configure environment values matching your Databricks serving endpoint.

## Reproduce Exactly (Checklist)

Use this as a release checklist:

1. Pin branch/commit.
2. Create fresh cluster/runtime.
3. Re-run ingestion notebooks in order.
4. Rebuild FAISS index.
5. Recreate/sync vector index.
6. Redeploy app from Git commit.
7. Run smoke tests.
8. Run benchmark notebook and save metrics.

Optional evaluation notebook:

- [notebooks/05_mlflow_rag_evaluation.py](notebooks/05_mlflow_rag_evaluation.py)

## Testing

```bash
pip install -e ".[dev,rag]"
pytest tests -v
```

## Key Environment Variables

| Variable | Purpose |
|---|---|
| LLM_OPENAI_BASE_URL | Databricks serving/AI gateway base URL |
| LLM_MODEL | LLM model id |
| NYAYA_RETRIEVAL_BACKEND | retrieval mode: vector_search or faiss |
| NYAYA_VS_ENDPOINT_NAME | vector search endpoint name |
| NYAYA_VS_INDEX_NAME | vector search index name |
| NYAYA_INDEX_DIR | FAISS index path (volume or local) |
| SARVAM_API_KEY | translation/STT/TTS key |

## Product and Technical Documentation

- [docs/APP_USER_GUIDE.md](docs/APP_USER_GUIDE.md)
- [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md)
- [docs/WORKSPACE_SETUP.md](docs/WORKSPACE_SETUP.md)
- [docs/PLAYGROUND_TO_APP.md](docs/PLAYGROUND_TO_APP.md)
- [docs/BENCHMARK_EVALUATION.md](docs/BENCHMARK_EVALUATION.md)
- [docs/PLAN.md](docs/PLAN.md)

## Current Branding

App name: MULIA (Multilingual Legal Information Assistant).

If you are migrating from older docs/scripts mentioning Nyaya Dhwani or Nyaya Sahayak, treat MULIA as the canonical name.
