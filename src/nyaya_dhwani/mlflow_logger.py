"""MLflow experiment logging for RAG queries.

Graceful degradation: all methods are no-ops when MLflow is unavailable or
MLFLOW_TRACKING_URI is not set.  Import is always safe.

Usage::

    from nyaya_dhwani.mlflow_logger import RAGQueryLogger

    with RAGQueryLogger(feature="bns_explainer") as tracker:
        # ... retrieval ...
        tracker.log_retrieval(query, chunk_ids, latency_ms=42)
        # ... LLM call ...
        tracker.log_llm(latency_ms=310, model_name="databricks-llama-4-maverick")
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy MLflow import so the app never crashes if mlflow isn't installed.
try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    mlflow = None  # type: ignore[assignment]
    _MLFLOW_AVAILABLE = False

EXPERIMENT_NAME = os.environ.get(
    "NYAYA_MLFLOW_EXPERIMENT", "/nyaya-dhwani/rag-queries"
)


def _is_configured() -> bool:
    """Return True only when MLflow is installed and a tracking URI is configured."""
    return _MLFLOW_AVAILABLE and bool(
        os.environ.get("MLFLOW_TRACKING_URI", "").strip()
        or os.environ.get("DATABRICKS_HOST", "").strip()  # Databricks auto-configures MLflow
    )


class RAGQueryLogger:
    """Context manager that wraps one RAG query turn as a nested MLflow run.

    Example::

        with RAGQueryLogger("legal_chat") as log:
            chunks = retriever.search(query, k=7)
            log.log_retrieval(query, [r["chunk_id"] for _, r in chunks.iterrows()], latency_ms=35)
            answer = llm_call(...)
            log.log_llm(latency_ms=280)
            log.log_language("hi")
    """

    def __init__(self, feature: str = "legal_chat") -> None:
        self.feature = feature
        self._run = None
        self._start: float = 0.0

    def __enter__(self) -> "RAGQueryLogger":
        self._start = time.monotonic()
        if _is_configured():
            try:
                mlflow.set_experiment(EXPERIMENT_NAME)
                self._run = mlflow.start_run(
                    nested=True,
                    run_name=f"{self.feature}_query",
                    tags={"feature": self.feature},
                )
                mlflow.log_param("feature", self.feature)
            except Exception as exc:
                logger.debug("MLflow start_run failed (non-fatal): %s", exc)
                self._run = None
        return self

    def __exit__(self, *_args) -> None:
        if self._run is not None:
            try:
                elapsed = int((time.monotonic() - self._start) * 1000)
                mlflow.log_metric("total_latency_ms", elapsed)
                mlflow.end_run()
            except Exception as exc:
                logger.debug("MLflow end_run failed (non-fatal): %s", exc)
        self._run = None

    def log_retrieval(
        self,
        query: str,
        chunk_ids: list[str],
        latency_ms: int,
    ) -> None:
        """Log retrieval parameters and metrics."""
        if self._run is None:
            return
        try:
            mlflow.log_param("query_preview", query[:200])
            mlflow.log_metric("retrieval_latency_ms", latency_ms)
            mlflow.log_metric("num_chunks_retrieved", len(chunk_ids))
        except Exception as exc:
            logger.debug("MLflow log_retrieval failed: %s", exc)

    def log_llm(
        self,
        latency_ms: int,
        model_name: str = "",
        tokens_approx: int = 0,
    ) -> None:
        """Log LLM call metrics."""
        if self._run is None:
            return
        try:
            mlflow.log_metric("llm_latency_ms", latency_ms)
            if model_name:
                mlflow.log_param("llm_model", model_name)
            if tokens_approx > 0:
                mlflow.log_metric("tokens_approx", tokens_approx)
        except Exception as exc:
            logger.debug("MLflow log_llm failed: %s", exc)

    def log_language(self, lang: str) -> None:
        """Log user language."""
        if self._run is None:
            return
        try:
            mlflow.log_param("language", lang)
        except Exception as exc:
            logger.debug("MLflow log_language failed: %s", exc)

    def log_result_quality(self, num_results: int, top_score: Optional[float] = None) -> None:
        """Log retrieval quality signals."""
        if self._run is None:
            return
        try:
            mlflow.log_metric("num_results", num_results)
            if top_score is not None:
                mlflow.log_metric("top_retrieval_score", float(top_score))
        except Exception as exc:
            logger.debug("MLflow log_result_quality failed: %s", exc)


def setup_parent_experiment() -> Optional[str]:
    """Set up the parent MLflow experiment and return its ID (if MLflow available).

    Call once from a notebook before starting your app or evaluation run.
    """
    if not _is_configured():
        logger.info("MLflow not configured — skipping experiment setup")
        return None
    try:
        mlflow.set_experiment(EXPERIMENT_NAME)
        exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        experiment_id = exp.experiment_id if exp else None
        logger.info("MLflow experiment ready: %s (id=%s)", EXPERIMENT_NAME, experiment_id)
        return experiment_id
    except Exception as exc:
        logger.warning("MLflow experiment setup failed: %s", exc)
        return None
