"""BNS-specific RAG with structured bilingual explanations and IPC cross-reference.

This module provides a specialised RAG pipeline for the Bharatiya Nyaya Sanhita (BNS) 2023
that re-ranks retrieved chunks to prioritise criminal-law content and generates structured
answers including Hindi key terms and IPC predecessor information.

Usage::

    from nyaya_dhwani.bns_explainer import explain_bns_section, format_bns_response
    from nyaya_dhwani.retriever import get_retriever

    retriever = get_retriever()
    result = explain_bns_section("What is the punishment for theft under BNS?", retriever)
    markdown = format_bns_response(result)
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import pandas as pd

from nyaya_dhwani.llm_client import chat_completions, extract_assistant_text
from nyaya_dhwani.keyword_boost import detect_section_references

if TYPE_CHECKING:
    from nyaya_dhwani.retriever import Retriever

logger = logging.getLogger(__name__)

# doc_types that contain BNS / IPC criminal law content
_BNS_DOC_TYPES = frozenset({"criminal_law", "law_mapping", "criminal_law_ipc"})

BNS_SYSTEM_PROMPT = """You are Nyaya Sahayak, an expert legal assistant specialising in the
Bharatiya Nyaya Sanhita (BNS) 2023 and its predecessor, the Indian Penal Code (IPC) 1860.

For each question, structure your answer with these numbered sections:

1. **Plain English Explanation**: A clear, jargon-free explanation accessible to any citizen.
2. **Hindi Key Terms** (हिंदी में मुख्य शब्द): Key legal terms in Hindi (Devanagari script) with brief meanings.
3. **IPC Predecessor**: Which IPC section(s) this replaces, and what changed in the transition to BNS.
4. **Punishment / Penalty**: The exact punishment or penalty if stated in the provided context.
5. **Practical Impact**: What this means for an ordinary citizen — rights, obligations, and remedies.

Rules:
- Use ONLY the provided Context. Do NOT rely on prior training knowledge beyond what is in context.
- Cite BNS section numbers explicitly (e.g. "BNS Section 303(2)").
- If the context is insufficient to answer fully, say so clearly in the relevant section.
- Respond in English; Hindi terms go in parentheses where relevant.
- Do NOT claim to be a lawyer or provide legal advice."""

DISCLAIMER = (
    "This information is for general awareness only and does not constitute legal advice. "
    "Consult a qualified lawyer for your specific situation."
)


def explain_bns_section(
    query: str,
    retriever: "Retriever",
    *,
    k: int = 10,
) -> dict:
    """Retrieve BNS-relevant chunks and generate a structured explanation.

    Returns a dict with keys:
    - ``english``: structured LLM response string
    - ``citations``: formatted citation block
    - ``sections_detected``: list of (act, num) tuples found in query
    - ``chunk_ids``: list of retrieved chunk IDs (for MLflow logging)
    - ``retrieval_ms``: retrieval latency in milliseconds
    - ``llm_ms``: LLM latency in milliseconds
    """
    t0 = time.monotonic()
    chunks_df = retriever.search(query.strip(), k=k)
    retrieval_ms = int((time.monotonic() - t0) * 1000)

    # Re-rank: criminal_law + law_mapping chunks first, then the rest.
    if not chunks_df.empty and "doc_type" in chunks_df.columns:
        bns_mask = chunks_df["doc_type"].isin(_BNS_DOC_TYPES)
        chunks_df = pd.concat(
            [chunks_df[bns_mask], chunks_df[~bns_mask]],
            ignore_index=True,
        ).head(k)

    sections = detect_section_references(query)
    sections_str = ""
    if sections:
        sections_str = "\nDetected section references: " + ", ".join(
            f"{a} {n}" for a, n in sections
        )

    texts = chunks_df["text"].tolist() if "text" in chunks_df.columns else []
    context = "\n\n---\n\n".join(str(t) for t in texts[:8])

    user_content = (
        f"Context:\n{context}"
        f"{sections_str}"
        f"\n\nQuestion: {query}"
        "\n\nPlease provide a structured explanation following the format in your instructions."
    )

    messages = [
        {"role": "system", "content": BNS_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    t1 = time.monotonic()
    raw = chat_completions(messages, max_tokens=2048, temperature=0.1)
    llm_ms = int((time.monotonic() - t1) * 1000)

    english_answer = extract_assistant_text(raw)

    # Build citations block
    cite_lines: list[str] = []
    for _, row in chunks_df.iterrows():
        title = str(row.get("title") or "").strip()
        source = str(row.get("source") or "").strip()
        doc_type = str(row.get("doc_type") or "").strip()
        bits = [x for x in (title, source, doc_type) if x]
        if bits:
            cite_lines.append("- " + " · ".join(bits[:3]))

    chunk_ids: list[str] = []
    if "chunk_id" in chunks_df.columns:
        chunk_ids = chunks_df["chunk_id"].dropna().tolist()

    return {
        "english": english_answer,
        "citations": "\n".join(cite_lines) if cite_lines else "(no sources retrieved)",
        "sections_detected": sections,
        "chunk_ids": chunk_ids,
        "retrieval_ms": retrieval_ms,
        "llm_ms": llm_ms,
    }


def format_bns_response(result: dict) -> str:
    """Format BNS explanation result as Gradio-compatible markdown."""
    answer = result.get("english", "")
    cites = result.get("citations", "")
    sections = result.get("sections_detected", [])

    parts: list[str] = []

    if sections:
        sec_tags = ", ".join(f"`{a} {n}`" for a, n in sections)
        parts.append(f"**Sections detected:** {sec_tags}\n")

    parts.append(answer)
    parts.append(f"\n---\n**Sources (retrieval)**\n{cites}")
    parts.append(f"\n---\n*{DISCLAIMER}*")

    return "\n".join(parts)
