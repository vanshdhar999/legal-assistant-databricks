#!/usr/bin/env bash
set -euo pipefail

# MULIA local bootstrap runner
# Usage:
#   ./run_local.sh
#   ./run_local.sh --no-install
#
# Optional: create .env.local in repo root; it will be auto-loaded.
# Required env vars (via shell or .env.local):
#   LLM_OPENAI_BASE_URL
#   LLM_MODEL
#   one of: DATABRICKS_TOKEN | LLM_API_KEY | OPENAI_API_KEY
#
# Retrieval defaults are aligned with app.yaml and can be overridden.

NO_INSTALL=0
if [[ "${1:-}" == "--no-install" ]]; then
  NO_INSTALL=1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -f ".env.local" ]]; then
  echo "[mulia] loading .env.local"
  set -a
  # shellcheck source=/dev/null
  source .env.local
  set +a
fi

# Defaults (can be overridden in env/.env.local)
: "${NYAYA_RETRIEVAL_BACKEND:=vector_search}"
: "${NYAYA_VS_ENDPOINT_NAME:=nyaya_vs_endpoint}"
: "${NYAYA_VS_INDEX_NAME:=workspace.india_legal.legal_rag_corpus_index}"
: "${NYAYA_INDEX_DIR:=/Volumes/workspace/india_legal/legal_files/nyaya_index}"
: "${LLM_MODEL:=databricks-llama-4-maverick}"

export NYAYA_RETRIEVAL_BACKEND NYAYA_VS_ENDPOINT_NAME NYAYA_VS_INDEX_NAME NYAYA_INDEX_DIR LLM_MODEL

fail=0

if [[ -z "${LLM_OPENAI_BASE_URL:-}" ]]; then
  echo "[mulia][error] LLM_OPENAI_BASE_URL is required."
  fail=1
fi

if [[ -z "${DATABRICKS_TOKEN:-}" && -z "${LLM_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[mulia][error] Set one of DATABRICKS_TOKEN, LLM_API_KEY, or OPENAI_API_KEY."
  fail=1
fi

# Databricks SDK calls (Vector Search, secret scopes) require a workspace host.
# If only LLM_OPENAI_BASE_URL is provided and it points to a workspace endpoint,
# derive DATABRICKS_HOST automatically.
if [[ -z "${DATABRICKS_HOST:-}" && -n "${LLM_OPENAI_BASE_URL:-}" ]]; then
  DERIVED_HOST="$(python - <<'PY'
from urllib.parse import urlparse
import os

base = os.environ.get("LLM_OPENAI_BASE_URL", "").strip()
if not base:
    raise SystemExit(0)

u = urlparse(base)
host = (u.hostname or "").lower()
if not host:
    raise SystemExit(0)

# AI Gateway host is not a workspace host for WorkspaceClient.
if host.endswith("ai-gateway.cloud.databricks.com"):
    raise SystemExit(0)

if u.scheme and u.netloc:
    print(f"{u.scheme}://{u.netloc}")
PY
)"
  if [[ -n "${DERIVED_HOST}" ]]; then
    export DATABRICKS_HOST="${DERIVED_HOST}"
    echo "[mulia] derived DATABRICKS_HOST from LLM_OPENAI_BASE_URL"
  fi
fi

if [[ -n "${DATABRICKS_TOKEN:-}" && -z "${DATABRICKS_HOST:-}" ]]; then
  echo "[mulia][error] DATABRICKS_TOKEN is set but DATABRICKS_HOST is missing."
  echo "[mulia][error] Set DATABRICKS_HOST=https://<your-workspace-host> for Databricks SDK auth."
  fail=1
fi

if [[ "${NYAYA_RETRIEVAL_BACKEND}" == "vector_search" ]]; then
  if [[ -z "${NYAYA_VS_ENDPOINT_NAME:-}" || -z "${NYAYA_VS_INDEX_NAME:-}" ]]; then
    echo "[mulia][error] vector_search selected but NYAYA_VS_ENDPOINT_NAME/NYAYA_VS_INDEX_NAME missing."
    fail=1
  fi
fi

if [[ $fail -ne 0 ]]; then
  cat <<'EOF'

Example .env.local
------------------
LLM_OPENAI_BASE_URL=https://<workspace-or-ai-gateway>/serving-endpoints
LLM_MODEL=databricks-llama-4-maverick
DATABRICKS_TOKEN=<your_pat>
DATABRICKS_HOST=https://<your-workspace-host>

# Optional Sarvam
SARVAM_API_KEY=<your_sarvam_key>

# Optional retrieval override
# NYAYA_RETRIEVAL_BACKEND=faiss
# NYAYA_INDEX_DIR=/absolute/path/to/nyaya_index
EOF
  exit 1
fi

if [[ ! -d ".venv" ]]; then
  echo "[mulia] creating .venv"
  python3 -m venv .venv
fi

# shellcheck source=/dev/null
source .venv/bin/activate

PY_MM="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PY_MAJ="${PY_MM%%.*}"
PY_MIN="${PY_MM##*.}"

INSTALL_EXTRAS=".[dev,rag,rag_embed,app,llm_openai,mlflow]"
if [[ "$PY_MAJ" -ge 3 && "$PY_MIN" -ge 13 ]]; then
  # faiss-cpu<1.8 has no wheels for Python 3.13; keep local run usable via Vector Search.
  INSTALL_EXTRAS=".[dev,app,llm_openai,mlflow]"
  echo "[mulia] Python ${PY_MM} detected: skipping rag extras (faiss-cpu<1.8 unavailable on 3.13)."
  echo "[mulia] Using vector_search backend for local run."
  if [[ "${NYAYA_RETRIEVAL_BACKEND}" == "faiss" ]]; then
    echo "[mulia][error] NYAYA_RETRIEVAL_BACKEND=faiss is incompatible with Python ${PY_MM} in this setup."
    echo "[mulia][error] Use NYAYA_RETRIEVAL_BACKEND=vector_search or run with Python 3.11/3.12."
    exit 1
  fi
fi

if [[ $NO_INSTALL -eq 0 ]]; then
  echo "[mulia] installing dependencies"
  python -m pip install --upgrade pip
  python -m pip install -e "$INSTALL_EXTRAS"
else
  echo "[mulia] --no-install set; skipping pip install"
fi

if [[ "$PY_MAJ" -ge 3 && "$PY_MIN" -ge 13 ]]; then
  if ! python -c "import audioop" >/dev/null 2>&1; then
    echo "[mulia] Python ${PY_MM}: installing audioop-lts compatibility package"
    python -m pip install audioop-lts
  fi
fi

echo "[mulia] starting app on local Gradio server"
python app/main.py
