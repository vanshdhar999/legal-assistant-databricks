"""Databricks Vector Search retriever backend.

Uses managed embeddings (Databricks computes embeddings from the ``text`` column)
and supports hybrid search + metadata filters for IPC/BNS section lookups.
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from nyaya_dhwani.keyword_boost import detect_section_references

logger = logging.getLogger(__name__)

_RESULT_COLUMNS = ["chunk_id", "text", "title", "source", "doc_type"]


class VectorSearchRetriever:
    """Query a Databricks Vector Search index via the SDK."""

    def __init__(self, endpoint_name: str, index_name: str) -> None:
        self._endpoint_name = endpoint_name
        self._index_name = index_name
        self._index = None

    def _get_client(self):
        """Return the vector_search_indexes API client (not the index object)."""
        if self._index is not None:
            return self._index
        from databricks.sdk import WorkspaceClient
        try:
            self._index = WorkspaceClient().vector_search_indexes
        except Exception as e:
            raise RuntimeError(
                "Failed to initialize Databricks Vector Search client. "
                "Set DATABRICKS_HOST=https://<workspace-host> and DATABRICKS_TOKEN=<pat>, "
                "or configure Databricks unified authentication. "
                f"Original error: {e}"
            ) from e
        logger.info("VectorSearchRetriever: connected to endpoint %s, index %s",
                     self._endpoint_name, self._index_name)
        return self._index

    def search(
        self,
        query: str,
        k: int = 7,
        doc_type_filter: str | None = None,
    ) -> pd.DataFrame:
        """Similarity search with optional metadata filters for section references.

        Args:
            doc_type_filter: When provided, restricts ALL results to this doc_type
                (overrides the automatic law_mapping filter for section references).
        """
        client = self._get_client()
        index_name = self._index_name

        # If caller requested a specific doc_type, use that directly.
        if doc_type_filter:
            filters = {"doc_type": doc_type_filter}
        else:
            # Detect IPC/BNS section references for targeted filtering.
            refs = detect_section_references(query)
            filters = {"doc_type": "law_mapping"} if refs else None

        try:
            # First: if we have section references (and no explicit doc_type_filter),
            # do a filtered search for law_mapping chunks.
            mapping_rows = []
            if filters and not doc_type_filter:
                try:
                    mapping_resp = client.query_index(
                        index_name=index_name,
                        columns=_RESULT_COLUMNS,
                        query_text=query,
                        num_results=3,
                        filters_json=json.dumps(filters),
                    )
                    mapping_rows = _response_to_rows(mapping_resp)
                except Exception:
                    logger.debug("Filtered VS search failed, continuing with unfiltered", exc_info=True)

            # Main search (filtered by doc_type if requested, otherwise unfiltered).
            main_query_kwargs: dict = dict(
                index_name=index_name,
                columns=_RESULT_COLUMNS,
                query_text=query,
                num_results=k,
            )
            if doc_type_filter:
                main_query_kwargs["filters_json"] = json.dumps({"doc_type": doc_type_filter})
            resp = client.query_index(**main_query_kwargs)
            main_rows = _response_to_rows(resp)

            # Merge: mapping results first, then main results (deduplicated).
            seen_ids = set()
            combined = []
            for row in mapping_rows + main_rows:
                cid = row.get("chunk_id")
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)
                combined.append(row)

            combined = combined[:k]
            for i, row in enumerate(combined):
                row["rank"] = i

            if not combined:
                return pd.DataFrame(columns=_RESULT_COLUMNS + ["score", "rank"])
            return pd.DataFrame(combined)

        except Exception as e:
            logger.warning("VectorSearchRetriever.search failed: %s", e)
            return pd.DataFrame(columns=_RESULT_COLUMNS + ["score", "rank"])


def _response_to_rows(resp) -> list[dict]:
    """Convert a VS similarity_search/query_index response to a list of dicts."""
    rows = []
    # The SDK may return a dataclass or a dict depending on version.
    # Convert to dict first if possible.
    if hasattr(resp, "as_dict"):
        resp = resp.as_dict()
    try:
        manifest = resp.get("manifest", {}) if isinstance(resp, dict) else getattr(resp, "manifest", {})
        result = resp.get("result", {}) if isinstance(resp, dict) else getattr(resp, "result", {})

        if isinstance(manifest, dict):
            col_names = [c.get("name", f"col_{i}") for i, c in enumerate(manifest.get("columns", []))]
        else:
            col_names = [c.name for c in (manifest.columns or [])]

        data_array = result.get("data_array", []) if isinstance(result, dict) else (result.data_array or [])

        for row_data in data_array:
            row = dict(zip(col_names, row_data))
            # VS returns a score column (typically named "score" or the last column).
            if "score" not in row:
                row["score"] = row.pop(col_names[-1], 0.0) if len(col_names) > len(_RESULT_COLUMNS) else 0.0
            rows.append(row)
    except Exception:
        logger.warning("Failed to parse VS response", exc_info=True)
    return rows
