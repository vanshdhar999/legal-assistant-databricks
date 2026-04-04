"""Government scheme eligibility checker.

Pipeline:
1. Hard filter: Databricks SQL query on ``gov_welfare_schemes`` Delta table
   (state, age, income, gender, caste).  Falls back to empty DataFrame if
   warehouse is not configured.
2. Soft match: RAG retrieval on ``government_scheme`` doc_type chunks.
3. LLM explanation: Llama Maverick summarises matched schemes for the user.

Usage::

    from nyaya_dhwani.scheme_checker import UserProfile, check_eligibility
    from nyaya_dhwani.retriever import get_retriever

    profile = UserProfile(state="Maharashtra", age=28, gender="Female",
                          income_annual_inr=180000, caste_category="OBC",
                          occupation="Farmer")
    retriever = get_retriever()
    candidates_df, explanation = check_eligibility(profile, retriever)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from nyaya_dhwani.retriever import Retriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INDIAN_STATES: list[str] = [
    "All India",
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya",
    "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim",
    "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand",
    "West Bengal",
    # UTs
    "Andaman and Nicobar Islands", "Chandigarh", "Delhi",
    "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry",
]

CASTE_CATEGORIES: list[str] = ["General", "OBC", "SC", "ST"]
GENDER_OPTIONS: list[str] = ["Male", "Female", "Other", "All"]
OCCUPATION_TAGS: list[str] = [
    "Farmer", "Student", "Self-employed", "Labour / Worker",
    "Government Employee", "BPL (Below Poverty Line)",
    "Senior Citizen", "Differently-abled", "Other",
]

_SCHEME_SYSTEM_PROMPT = """You are a government scheme advisor for India. Based on the user's
profile and the scheme information provided in the context, identify which schemes they are
likely eligible for.

For each relevant scheme, provide:
1. **Scheme Name**: The official scheme name
2. **Eligibility Match**: Why this user qualifies
3. **Key Benefits**: What they will receive (cash, subsidy, service, etc.)
4. **How to Apply**: Portal, office, or process if mentioned in context

Focus on schemes most directly relevant to the user's profile.
Be practical, specific, and use simple language suitable for rural users.

End with: "Visit **myscheme.gov.in** or your local government office to verify exact
eligibility and apply." """

DISCLAIMER = (
    "This information is for general awareness only. Scheme eligibility may vary. "
    "Verify at myscheme.gov.in or your nearest government office."
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class UserProfile:
    """Structured user profile for scheme eligibility matching."""
    state: str = "All India"
    age: int = 30
    gender: str = "All"
    income_annual_inr: int = 250000
    caste_category: str = "General"
    occupation: str = ""

    def as_query_string(self) -> str:
        """Convert profile to a natural-language query for RAG retrieval."""
        parts = [
            f"government welfare schemes for {self.gender.lower() if self.gender != 'All' else 'citizens'}",
            f"age {self.age}",
            f"state {self.state}",
            f"annual income {self.income_annual_inr}",
            f"category {self.caste_category}",
        ]
        if self.occupation:
            parts.append(f"occupation {self.occupation}")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Delta SQL hard filter
# ---------------------------------------------------------------------------

def query_eligible_schemes(
    profile: UserProfile,
    *,
    catalog: str = "workspace",
    schema: str = "india_legal",
    max_schemes: int = 50,
) -> pd.DataFrame:
    """Query the ``gov_welfare_schemes`` Delta table with hard eligibility filters.

    Requires ``NYAYA_SQL_WAREHOUSE_ID`` env var pointing to a Databricks SQL Warehouse.
    Returns an empty DataFrame (and logs a warning) if the warehouse is unavailable
    — the caller then relies on RAG-based soft matching.
    """
    warehouse_id = os.environ.get("NYAYA_SQL_WAREHOUSE_ID", "").strip()
    if not warehouse_id:
        logger.info("NYAYA_SQL_WAREHOUSE_ID not set — skipping SQL eligibility filter")
        return pd.DataFrame()

    table = f"`{catalog}`.`{schema}`.`gov_welfare_schemes`"

    # Normalise gender: M / F / ALL
    _gender_map = {"male": "M", "female": "F", "other": "ALL", "all": "ALL"}
    gender_val = _gender_map.get(profile.gender.lower(), "ALL")

    caste_val = profile.caste_category.upper()
    state_val = profile.state.replace("'", "''")  # basic SQL escape

    sql = f"""
        SELECT
            scheme_id, scheme_name, ministry, state, description,
            eligibility_raw, benefits,
            min_age, max_age, income_limit_inr, gender, caste_category,
            occupation_tags
        FROM {table}
        WHERE (UPPER(state) = 'ALL' OR UPPER(state) = 'ALL INDIA'
               OR UPPER(state) = UPPER('{state_val}'))
          AND (min_age IS NULL OR CAST(min_age AS INT) <= {profile.age})
          AND (max_age IS NULL OR CAST(max_age AS INT) >= {profile.age})
          AND (income_limit_inr IS NULL
               OR CAST(income_limit_inr AS BIGINT) >= {profile.income_annual_inr})
          AND (UPPER(gender) = 'ALL' OR gender IS NULL
               OR UPPER(gender) = '{gender_val}')
          AND (UPPER(caste_category) = 'ALL' OR caste_category IS NULL
               OR UPPER(caste_category) = '{caste_val}')
        ORDER BY scheme_name
        LIMIT {max_schemes}
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
        ):
            cols = [c.name for c in resp.manifest.schema.columns]
            df = pd.DataFrame(resp.result.data_array, columns=cols)
            logger.info("SQL eligibility filter: %d schemes matched", len(df))
            return df
    except Exception as exc:
        logger.warning("Databricks SQL eligibility query failed: %s", exc)

    return pd.DataFrame()


# ---------------------------------------------------------------------------
# RAG + LLM explanation
# ---------------------------------------------------------------------------

def explain_schemes_with_rag(
    profile: UserProfile,
    candidates: pd.DataFrame,
    retriever: "Retriever",
) -> str:
    """Build an LLM explanation of eligible schemes using RAG context.

    Retrieves scheme corpus chunks (``doc_type == "government_scheme"``) and
    combines them with SQL candidates (if any) to generate a user-friendly summary.
    """
    from nyaya_dhwani.llm_client import chat_completions, extract_assistant_text

    # Retrieve scheme-specific chunks
    scheme_query = profile.as_query_string()
    chunks_df = retriever.search(scheme_query, k=8)

    # Prioritise scheme chunks
    if not chunks_df.empty and "doc_type" in chunks_df.columns:
        scheme_mask = chunks_df["doc_type"] == "government_scheme"
        chunks_df = pd.concat(
            [chunks_df[scheme_mask], chunks_df[~scheme_mask]],
            ignore_index=True,
        ).head(8)

    texts = [str(t) for t in chunks_df["text"].tolist()] if "text" in chunks_df.columns else []
    rag_context = "\n\n---\n\n".join(texts[:6])

    # Add SQL-filtered scheme names as additional context
    sql_context = ""
    if not candidates.empty and "scheme_name" in candidates.columns:
        names = candidates["scheme_name"].dropna().head(15).tolist()
        sql_context = (
            "\n\nPre-filtered eligible schemes from database:\n"
            + "\n".join(f"- {n}" for n in names)
        )

    profile_block = (
        f"User Profile:\n"
        f"- State: {profile.state}\n"
        f"- Age: {profile.age} years\n"
        f"- Gender: {profile.gender}\n"
        f"- Annual Income: ₹{profile.income_annual_inr:,}\n"
        f"- Category: {profile.caste_category}\n"
        f"- Occupation: {profile.occupation or 'Not specified'}\n"
    )

    user_content = (
        f"{profile_block}\n"
        f"Context (scheme information):\n{rag_context}"
        f"{sql_context}\n\n"
        "Which government schemes is this person likely eligible for? "
        "Explain in simple language suitable for a rural user."
    )

    messages = [
        {"role": "system", "content": _SCHEME_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    raw = chat_completions(messages, max_tokens=2048, temperature=0.2)
    return extract_assistant_text(raw)


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def check_eligibility(
    profile: UserProfile,
    retriever: "Retriever",
) -> tuple[pd.DataFrame, str]:
    """Full eligibility pipeline: SQL hard filter → RAG soft match → LLM explanation.

    Returns:
        candidates_df: DataFrame of matched schemes from Delta (may be empty).
        explanation: Markdown-formatted LLM explanation of eligible schemes.
    """
    candidates = query_eligible_schemes(profile)
    explanation = explain_schemes_with_rag(profile, candidates, retriever)
    return candidates, explanation


def format_scheme_response(explanation: str, candidates_df: pd.DataFrame) -> str:
    """Format the scheme checker result as Gradio markdown."""
    parts: list[str] = [explanation]

    if not candidates_df.empty and "scheme_name" in candidates_df.columns:
        count = len(candidates_df)
        parts.append(f"\n---\n**{count} scheme(s) matched in database**")

    parts.append(f"\n---\n*{DISCLAIMER}*")
    return "\n".join(parts)
