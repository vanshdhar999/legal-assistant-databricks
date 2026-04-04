"""IPC → BNS section comparison tool.

Pipeline:
1. Parse the section number from user input (e.g. "302", "IPC 302", "Section 302").
2. Hard lookup: Databricks SQL JOIN across ``ipc_sections``, ``bns_sections``, and
   ``bns_ipc_mapping`` Delta tables.
3. RAG fallback: if SQL unavailable, search FAISS/VS index for both texts.
4. Compute unified diff between IPC and BNS texts (difflib).
5. LLM comparative analysis: what changed, why, and practical impact.

Usage::

    from nyaya_dhwani.ipc_bns_compare import compare_ipc_to_bns
    from nyaya_dhwani.retriever import get_retriever

    retriever = get_retriever()
    result = compare_ipc_to_bns("302", retriever)
    print(result["analysis"])
"""

from __future__ import annotations

import difflib
import logging
import os
import re
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from nyaya_dhwani.retriever import Retriever

logger = logging.getLogger(__name__)

_SECTION_NUM_RE = re.compile(
    r"(?:IPC\s+)?(?:Section\s+)?(\d+(?:\([a-z0-9]+\))?)",
    re.IGNORECASE,
)

_COMPARE_SYSTEM_PROMPT = """You are a senior Indian legal expert specialising in criminal law.
Compare the IPC and BNS provisions below and provide a structured analysis.

Structure your response as:
1. **What Changed**: Key differences between IPC and BNS versions
2. **What Stayed the Same**: Unchanged provisions
3. **New Provisions**: Additions in BNS not present in IPC (or repealed content)
4. **Punishment / Penalty Changes**: Any changes in punishments, fines, or imprisonment terms
5. **Practical Impact**: What this change means for citizens, victims, and the accused

Rules:
- Be specific — cite section numbers.
- If a section was repealed in BNS, explain the intent.
- Use plain language alongside technical terms.
- Do NOT speculate beyond the provided texts."""

DISCLAIMER = (
    "This comparison is for informational purposes only. "
    "Consult the official Gazette or a qualified lawyer for authoritative interpretation."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_section_num(text: str) -> str | None:
    """Extract numeric section from inputs like '302', 'IPC 302', 'Section 302A'."""
    m = _SECTION_NUM_RE.search(text.strip())
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Delta SQL lookup
# ---------------------------------------------------------------------------

def lookup_ipc_section(
    ipc_section_num: str,
    *,
    catalog: str = "workspace",
    schema: str = "india_legal",
) -> dict | None:
    """Join ipc_sections + bns_ipc_mapping + bns_sections to get both texts.

    Returns a dict with keys:
        ipc_num, ipc_title, ipc_text, bns_num, bns_title, bns_text, mapping_status.
    Returns None if SQL warehouse not configured or section not found.
    """
    num = _parse_section_num(ipc_section_num)
    if not num:
        return None

    warehouse_id = os.environ.get("NYAYA_SQL_WAREHOUSE_ID", "").strip()
    if not warehouse_id:
        return None

    sql = f"""
        SELECT
            i.section_number   AS ipc_num,
            i.section_title    AS ipc_title,
            i.section_text     AS ipc_text,
            m.bns_section      AS bns_num,
            b.section_title    AS bns_title,
            b.section_text     AS bns_text,
            m.status           AS mapping_status
        FROM `{catalog}`.`{schema}`.`ipc_sections` i
        LEFT JOIN `{catalog}`.`{schema}`.`bns_ipc_mapping` m
               ON CAST(i.section_number AS STRING) = m.ipc_section
        LEFT JOIN `{catalog}`.`{schema}`.`bns_sections` b
               ON m.bns_section = CAST(b.section_number AS STRING)
        WHERE CAST(i.section_number AS STRING) = '{num}'
        LIMIT 1
    """

    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        resp = w.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            statement=sql,
            wait_timeout="30s",
        )
        if (
            hasattr(resp, "result") and resp.result
            and hasattr(resp.result, "data_array")
            and resp.result.data_array
            and resp.result.data_array[0]
        ):
            cols = [c.name for c in resp.manifest.schema.columns]
            return dict(zip(cols, resp.result.data_array[0]))
    except Exception as exc:
        logger.warning("SQL IPC lookup failed: %s", exc)

    return None


# ---------------------------------------------------------------------------
# RAG fallback
# ---------------------------------------------------------------------------

def _rag_fallback_lookup(ipc_section_num: str, retriever: "Retriever") -> dict | None:
    """Use FAISS/VS retriever to find IPC and BNS text when SQL is unavailable."""
    num = _parse_section_num(ipc_section_num)
    if not num:
        return None

    query = f"IPC Section {num} BNS equivalent punishment"
    chunks_df = retriever.search(query, k=8)
    if chunks_df.empty:
        return None

    ipc_text, bns_text, bns_num = "", "", ""

    for _, row in chunks_df.iterrows():
        text = str(row.get("text", ""))
        chunk_id = str(row.get("chunk_id", ""))
        doc_type = str(row.get("doc_type", ""))

        if chunk_id == f"IPC_S{num}" or (
            doc_type == "criminal_law_ipc" and f"Section {num}" in text
        ):
            ipc_text = text[:2000]
        elif chunk_id.startswith("MAP_") and (
            f"IPC {num}" in text or f"IPC Section {num}" in text
        ):
            bns_text = text[:2000]
            m = re.search(r"BNS\s+(?:Section\s+)?(\d+\w*)", text, re.IGNORECASE)
            if m:
                bns_num = m.group(1)

    if not ipc_text and not bns_text:
        return None

    return {
        "ipc_num": num,
        "ipc_title": f"IPC Section {num}",
        "ipc_text": ipc_text,
        "bns_num": bns_num or "—",
        "bns_title": f"BNS Section {bns_num}" if bns_num else "No direct equivalent found",
        "bns_text": bns_text,
        "mapping_status": "RAG-retrieved (approximate)",
    }


# ---------------------------------------------------------------------------
# Text diff
# ---------------------------------------------------------------------------

def compute_text_diff(
    ipc_text: str,
    bns_text: str,
    max_lines: int = 50,
) -> str:
    """Unified diff between IPC and BNS section texts."""
    if not ipc_text.strip() and not bns_text.strip():
        return "(Both texts unavailable)"
    if not ipc_text.strip():
        return "(IPC text not available for diff)"
    if not bns_text.strip():
        return "(BNS equivalent text not available for diff — section may be repealed)"

    ipc_lines = ipc_text.splitlines(keepends=True)
    bns_lines = bns_text.splitlines(keepends=True)

    diff = list(difflib.unified_diff(
        ipc_lines, bns_lines,
        fromfile="IPC", tofile="BNS (Bharatiya Nyaya Sanhita)",
        lineterm="",
    ))

    if not diff:
        return "(Texts are identical — no changes detected)"

    result = "".join(diff[:max_lines])
    if len(diff) > max_lines:
        result += f"\n... ({len(diff) - max_lines} more lines truncated)"
    return result


# ---------------------------------------------------------------------------
# LLM analysis
# ---------------------------------------------------------------------------

def llm_comparative_analysis(ipc_data: dict) -> str:
    """Generate structured LLM comparative analysis between IPC and BNS versions."""
    from nyaya_dhwani.llm_client import chat_completions, extract_assistant_text

    ipc_num = ipc_data.get("ipc_num", "?")
    bns_num = ipc_data.get("bns_num", "—")
    ipc_title = ipc_data.get("ipc_title", f"IPC Section {ipc_num}")
    bns_title = ipc_data.get("bns_title", f"BNS Section {bns_num}")
    ipc_text = (ipc_data.get("ipc_text") or "Not available")[:1500]
    bns_text = (ipc_data.get("bns_text") or "")[:1500]
    status = ipc_data.get("mapping_status", "unknown")

    bns_block = (
        f"{bns_title}:\n{bns_text}"
        if bns_text.strip()
        else f"This IPC section has been repealed or has no direct BNS equivalent (status: {status})."
    )

    user_content = (
        f"Compare {ipc_title} with BNS Section {bns_num} (Mapping status: {status}).\n\n"
        f"{ipc_title}:\n{ipc_text}\n\n"
        f"{bns_block}\n\n"
        "Provide a detailed comparative legal analysis."
    )

    messages = [
        {"role": "system", "content": _COMPARE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    raw = chat_completions(messages, max_tokens=2048, temperature=0.1)
    return extract_assistant_text(raw)


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def compare_ipc_to_bns(
    ipc_section_input: str,
    retriever: "Retriever",
) -> dict:
    """Full IPC → BNS comparison pipeline.

    Returns a dict with keys:
    - ``found``: bool — True if section data was located
    - ``data``: dict with ipc_num, ipc_title, ipc_text, bns_num, bns_title, bns_text, mapping_status
    - ``diff``: unified diff string
    - ``analysis``: LLM comparative analysis markdown
    - ``error``: error/warning message or None
    """
    num = _parse_section_num(ipc_section_input)
    if not num:
        return {
            "found": False,
            "data": None,
            "diff": "",
            "analysis": "",
            "error": (
                f"Could not parse a section number from '{ipc_section_input}'. "
                "Try entering just the number (e.g. 302) or 'IPC 302'."
            ),
        }

    # 1. SQL lookup
    data = lookup_ipc_section(num)

    # 2. RAG fallback
    if not data:
        data = _rag_fallback_lookup(num, retriever)

    # 3. Last resort: LLM + generic RAG context
    if not data:
        query = f"IPC Section {num} Bharatiya Nyaya Sanhita equivalent"
        chunks_df = retriever.search(query, k=6)
        texts = chunks_df["text"].tolist() if not chunks_df.empty else []

        from nyaya_dhwani.llm_client import chat_completions, extract_assistant_text, rag_user_message
        ctx = "\n\n".join(str(t) for t in texts[:5])
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a legal expert on Indian criminal law. "
                    "Explain the IPC section and its BNS equivalent if any."
                ),
            },
            {"role": "user", "content": rag_user_message([ctx], query)},
        ]
        raw = chat_completions(messages, max_tokens=1024, temperature=0.2)
        analysis = extract_assistant_text(raw)

        return {
            "found": False,
            "data": None,
            "diff": "",
            "analysis": analysis,
            "error": (
                f"IPC Section {num} not found in database. "
                "Showing RAG-retrieved information (may be incomplete)."
            ),
        }

    diff_text = compute_text_diff(
        data.get("ipc_text") or "",
        data.get("bns_text") or "",
    )
    analysis = llm_comparative_analysis(data)

    return {
        "found": True,
        "data": data,
        "diff": diff_text,
        "analysis": analysis,
        "error": None,
    }


def format_comparison_response(result: dict) -> tuple[str, str, str, str]:
    """Format comparison result into four markdown/text strings for the Gradio UI.

    Returns: (ipc_md, bns_md, analysis_md, diff_text)
    """
    if not result.get("found") and result.get("error"):
        err = result["error"]
        analysis = result.get("analysis", "")
        return (
            f"*{err}*",
            "",
            analysis or "*No analysis available.*",
            "",
        )

    data = result.get("data") or {}

    ipc_num = data.get("ipc_num", "?")
    ipc_title = data.get("ipc_title", f"IPC Section {ipc_num}")
    ipc_text = data.get("ipc_text") or "*Not available*"
    bns_num = data.get("bns_num", "—")
    bns_title = data.get("bns_title", f"BNS Section {bns_num}")
    bns_text = data.get("bns_text") or "*No direct BNS equivalent / section repealed*"
    status = data.get("mapping_status", "")

    ipc_md = f"**{ipc_title}**\n\n{ipc_text}"
    bns_md = f"**{bns_title}**\n\n{bns_text}"
    if status:
        bns_md += f"\n\n*Mapping status: {status}*"

    analysis_md = result.get("analysis", "*Analysis not available.*")
    analysis_md += f"\n\n---\n*{DISCLAIMER}*"

    diff_text = result.get("diff", "")

    return ipc_md, bns_md, analysis_md, diff_text
