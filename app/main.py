"""MULIA: Gradio app — Legal Q&A + BNS Explainer + Scheme Eligibility + IPC↔BNS Compare.

Built on top of the original Nyaya Dhwani RAG assistant.
See docs/PLAN.md and CLAUDE.md for architecture details.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Repo root on Databricks Repos / local clone
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import gradio as gr
import numpy as np

# ---------- Monkey-patch gradio_client bug (1.3.0 + Gradio 4.44.x) ----------
# get_api_info() crashes on Chatbot schemas where additionalProperties is True
# (a bool).  The internal recursive calls use the module-level name, so we must
# replace the actual function objects in the module namespace.
import gradio_client.utils as _gc_utils  # noqa: E402

_orig_inner = _gc_utils._json_schema_to_python_type
_orig_get_type = _gc_utils.get_type

def _safe_inner(schema, defs=None):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_inner(schema, defs)

def _safe_get_type(schema):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_get_type(schema)

_gc_utils._json_schema_to_python_type = _safe_inner
_gc_utils.get_type = _safe_get_type
# ---------- End monkey-patch ------------------------------------------------

from nyaya_dhwani.llm_client import chat_completions, extract_assistant_text, rag_user_message
from nyaya_dhwani.retriever import Retriever, get_retriever
from nyaya_dhwani.sarvam_client import (
    is_configured as sarvam_configured,
    numpy_audio_to_wav_bytes,
    speech_to_text_file,
    strip_markdown_for_tts,
    text_to_speech_wav_bytes,
    transcript_from_stt_response,
    translate_text,
    wav_bytes_to_numpy_float32,
)

# New feature modules (graceful import — app still works if any are missing)
try:
    from nyaya_dhwani.bns_explainer import explain_bns_section, format_bns_response
    _BNS_EXPLAINER_OK = True
except Exception:
    _BNS_EXPLAINER_OK = False

try:
    from nyaya_dhwani.scheme_checker import (
        UserProfile, check_eligibility, format_scheme_response,
        INDIAN_STATES, CASTE_CATEGORIES, GENDER_OPTIONS, OCCUPATION_TAGS,
    )
    _SCHEME_CHECKER_OK = True
except Exception:
    _SCHEME_CHECKER_OK = False

try:
    from nyaya_dhwani.ipc_bns_compare import compare_ipc_to_bns, format_comparison_response
    _IPC_BNS_OK = True
except Exception:
    _IPC_BNS_OK = False

try:
    from nyaya_dhwani.mlflow_logger import RAGQueryLogger
    _MLFLOW_LOGGER_OK = True
except Exception:
    _MLFLOW_LOGGER_OK = False
    class RAGQueryLogger:  # type: ignore[no-redef]
        """No-op fallback when mlflow_logger is unavailable."""
        def __init__(self, feature=""):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def log_retrieval(self, *a, **kw):
            pass
        def log_llm(self, *a, **kw):
            pass
        def log_language(self, *a, **kw):
            pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static data
# ---------------------------------------------------------------------------

TOPIC_SEEDS: dict[str, str] = {
    "Tenant rights": "What are my basic rights as a tenant in India regarding eviction and rent increases?",
    "Divorce law": "What are the grounds for divorce under Indian law for mutual consent?",
    "Consumer cases": "How do I file a consumer complaint in India for defective goods?",
    "Property law": "What documents should I check before buying residential property in India?",
    "Labour rights": "What are an employee's rights regarding notice period and gratuity?",
    "FIR / Police": "What is the procedure to file an FIR and what are my rights when arrested?",
    "Domestic violence": "What legal protections exist for victims of domestic violence in India?",
    "RTI": "How do I file a Right to Information application and what fees apply?",
}

BNS_TOPIC_SEEDS: dict[str, str] = {
    "Theft (BNS 303)": "What is the punishment for theft under BNS Section 303?",
    "Murder (BNS 101)": "Explain BNS Section 101 on murder and how it compares to IPC 302.",
    "Rape (BNS 63)": "What does BNS Section 63 say about rape and its punishment?",
    "Cheating (BNS 316)": "Explain BNS Section 316 on cheating and dishonest inducement.",
    "Domestic cruelty": "What does BNS say about cruelty by husband or his relatives?",
    "Right to bail": "When can a person accused under BNS be granted bail?",
}

SARVAM_LANGUAGES: list[tuple[str, str]] = [
    ("en", "English"),
    ("hi", "Hindi · हिन्दी"),
    ("bn", "Bengali"),
    ("te", "Telugu"),
    ("mr", "Marathi"),
    ("ta", "Tamil"),
    ("gu", "Gujarati"),
    ("kn", "Kannada"),
    ("ml", "Malayalam"),
    ("pa", "Punjabi"),
    ("or", "Odia"),
    ("ur", "Urdu"),
    ("as", "Assamese"),
]

UI_TO_BCP47: dict[str, str] = {
    "en": "en-IN", "hi": "hi-IN", "bn": "bn-IN", "te": "te-IN",
    "mr": "mr-IN", "ta": "ta-IN", "gu": "gu-IN", "kn": "kn-IN",
    "ml": "ml-IN", "pa": "pa-IN", "or": "od-IN", "ur": "hi-IN", "as": "bn-IN",
}

DISCLAIMER_EN = (
    "This information is for general awareness only and does not constitute legal advice. "
    "Consult a qualified lawyer for your specific situation."
)

SYSTEM_PROMPT = (
    "You are MULIA (Multilingual Legal Information Assistant), an assistant for Indian legal information. "
    "Answer using the Context below when it is relevant. Cite Acts or sections when the context supports it. "
    "If the context is insufficient, say so briefly. "
    "Do not claim to be a lawyer. Keep answers clear and structured. "
    "Respond in English."
)

# ---------------------------------------------------------------------------
# Runtime / retriever singleton
# ---------------------------------------------------------------------------

def bcp47_target(lang: str) -> str:
    return UI_TO_BCP47.get(lang, "en-IN")


class RAGRuntime:
    """Lazy-load retriever (FAISS, Vector Search, or fallback combo)."""

    def __init__(self) -> None:
        self._retriever: Retriever | None = None

    def load(self) -> None:
        if self._retriever is not None:
            return
        self._retriever = get_retriever()
        logger.info("Retriever loaded: %s", type(self._retriever).__name__)

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            raise RuntimeError("RAGRuntime not loaded")
        return self._retriever


_runtime: RAGRuntime | None = None


def get_runtime() -> RAGRuntime:
    global _runtime
    if _runtime is None:
        _runtime = RAGRuntime()
    return _runtime


# ---------------------------------------------------------------------------
# Shared helpers (Legal Chat tab — unchanged from original)
# ---------------------------------------------------------------------------

def _format_citations(chunks_df) -> str:
    lines: list[str] = []
    for _, row in chunks_df.iterrows():
        title = row.get("title") or ""
        source = row.get("source") or ""
        doc_type = row.get("doc_type") or ""
        bits = [str(x).strip() for x in (title, source, doc_type) if x and str(x).strip()]
        if bits:
            lines.append("- " + " · ".join(bits[:3]))
    return "\n".join(lines) if lines else "(no metadata)"


def _rag_answer_english(query_en: str) -> tuple[str, str]:
    """LLM answer in English + citations block."""
    rt = get_runtime()
    rt.load()
    q = query_en.strip()
    chunks_df = rt.retriever.search(q, k=7)
    texts = chunks_df["text"].tolist() if "text" in chunks_df.columns else []
    user_content = rag_user_message([str(t) for t in texts], q)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    raw = chat_completions(messages, max_tokens=2048, temperature=0.2)
    assistant_en = extract_assistant_text(raw)
    cites = _format_citations(chunks_df)
    return assistant_en, cites


_TRANSLATE_CHUNK_LIMIT = 500


def _chunked_translate(text: str, *, source: str, target: str) -> str:
    paragraphs = text.split("\n")
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 > _TRANSLATE_CHUNK_LIMIT and current:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n{para}" if current else para
    if current:
        chunks.append(current)

    translated_parts = []
    for chunk in chunks:
        if not chunk.strip():
            translated_parts.append(chunk)
            continue
        try:
            result = translate_text(chunk, source_language_code=source, target_language_code=target)
            translated_parts.append(result)
        except Exception as e:
            logger.warning("Mayura chunk translate failed, keeping original: %s", e)
            translated_parts.append(chunk)
    return "\n".join(translated_parts)


def _maybe_translate(text: str, *, source: str, target: str) -> str:
    if source == target:
        return text
    if not sarvam_configured():
        return text
    if len(text) > _TRANSLATE_CHUNK_LIMIT:
        return _chunked_translate(text, source=source, target=target)
    try:
        return translate_text(text, source_language_code=source, target_language_code=target)
    except Exception as e:
        logger.warning("Mayura translate failed, using original: %s", e)
        return text


def text_to_query_english(user_text: str, lang: str) -> str:
    t = user_text.strip()
    if not t:
        return t
    if lang == "en":
        return t
    if not sarvam_configured():
        logger.warning("SARVAM_API_KEY missing — using raw text for retrieval (degraded).")
        return t
    return _maybe_translate(t, source="auto", target="en-IN")


def resolve_user_message(
    text: str,
    audio: tuple[int, np.ndarray] | None,
    lang: str,
) -> tuple[str, str]:
    text = (text or "").strip()
    logger.debug("resolve_user_message: text=%r, audio type=%s",
                 text[:80] if text else "", type(audio).__name__)

    if text:
        q_en = text_to_query_english(text, lang)
        return (text, q_en)

    if audio is not None:
        sr, data = audio
        if data is not None and len(np.asarray(data)) > 0:
            if not sarvam_configured():
                raise RuntimeError("Set SARVAM_API_KEY for voice input (Sarvam STT).")
            wav = numpy_audio_to_wav_bytes(np.asarray(data), int(sr))
            mode = os.environ.get("SARVAM_STT_MODE", "translate").strip()
            lang_hint = bcp47_target(lang) if mode == "transcribe" else None
            st = speech_to_text_file(wav, mode=mode, language_code=lang_hint)
            tr = transcript_from_stt_response(st)
            if mode == "translate":
                return (f"🎤 {tr}", tr.strip())
            q_en = _maybe_translate(tr, source="auto", target="en-IN")
            return (f"🎤 {tr}", q_en.strip())

    raise ValueError(
        "Type a question or record audio. "
        "If you just recorded, wait for the audio to finish processing then try again."
    )


def build_reply_markdown(assistant_en: str, cites: str, lang: str) -> str:
    sources_block = f"**Sources (retrieval)**\n{cites}"

    if lang == "en" or not sarvam_configured():
        return (
            f"{assistant_en}\n\n---\n{sources_block}"
            f"\n\n---\n*{DISCLAIMER_EN}*"
        )

    tgt = bcp47_target(lang)
    body_translated = _maybe_translate(assistant_en, source="en-IN", target=tgt)
    disc_translated = _maybe_translate(DISCLAIMER_EN, source="en-IN", target=tgt)

    lang_label = dict(SARVAM_LANGUAGES).get(lang, lang)
    return (
        f"**{lang_label}:**\n\n{body_translated}\n\n"
        f"---\n**English:**\n\n{assistant_en}\n\n"
        f"---\n{sources_block}"
        f"\n\n---\n*{disc_translated}*"
    )


def maybe_tts(text_markdown: str, lang: str, enabled: bool) -> tuple[int, np.ndarray] | None:
    if not enabled or not sarvam_configured():
        return None
    narrative = text_markdown.split("\n---\n", 1)[0]
    import re
    narrative = re.sub(r"^\*\*[^*]+:\*\*\s*", "", narrative.strip())
    plain = strip_markdown_for_tts(narrative)
    if not plain.strip():
        return None
    tgt = bcp47_target(lang)
    try:
        wav = text_to_speech_wav_bytes(plain, target_language_code=tgt)
        sr, arr = wav_bytes_to_numpy_float32(wav)
        return (sr, arr)
    except Exception as e:
        logger.warning("TTS failed: %s", e)
        return None


def run_turn(
    message: str,
    audio: tuple[int, np.ndarray] | None,
    history: list | None,
    lang: str,
    tts_on: bool,
) -> tuple[str, list, tuple[int, np.ndarray] | None, None]:
    history = [list(pair) for pair in history] if history else []
    try:
        with RAGQueryLogger("legal_chat") as tracker:
            tracker.log_language(lang)
            user_show, q_en = resolve_user_message(message, audio, lang)
            assistant_en, cites = _rag_answer_english(q_en)
            reply_md = build_reply_markdown(assistant_en, cites, lang)
        history.append([user_show, reply_md])
        audio_out = maybe_tts(reply_md, lang, tts_on)
        return "", history, audio_out, None
    except Exception as e:
        logger.exception("run_turn")
        err = f"**Error:** {e}"
        history.append([message or "🎤 (audio)", err])
        return "", history, None, None


# ---------------------------------------------------------------------------
# BNS Explainer tab handlers
# ---------------------------------------------------------------------------

def run_bns_turn(
    query: str,
    history: list | None,
    lang: str,
) -> tuple[str, list]:
    """Handle a BNS Explainer query."""
    history = [list(pair) for pair in history] if history else []
    if not query.strip():
        return "", history

    if not _BNS_EXPLAINER_OK:
        err = "**BNS Explainer module not available.** Please check installation."
        history.append([query, err])
        return "", history

    try:
        rt = get_runtime()
        rt.load()
        with RAGQueryLogger("bns_explainer") as tracker:
            tracker.log_language(lang)
            result = explain_bns_section(query.strip(), rt.retriever, k=10)
            tracker.log_retrieval(query, result.get("chunk_ids", []), result.get("retrieval_ms", 0))
            tracker.log_llm(result.get("llm_ms", 0))
        reply = format_bns_response(result)
        history.append([query, reply])
        return "", history
    except Exception as e:
        logger.exception("run_bns_turn")
        history.append([query, f"**Error:** {e}"])
        return "", history


# ---------------------------------------------------------------------------
# Scheme Eligibility tab handlers
# ---------------------------------------------------------------------------

def run_scheme_check(
    state: str,
    age: int | float,
    gender: str,
    income: int | float,
    caste: str,
    occupation: str,
    lang: str,
) -> tuple[str, str]:
    """Run scheme eligibility check and return (explanation_md, status_msg)."""
    if not _SCHEME_CHECKER_OK:
        return "**Scheme Checker module not available.** Please check installation.", ""

    try:
        rt = get_runtime()
        rt.load()
        profile = UserProfile(
            state=str(state or "All India"),
            age=int(age or 30),
            gender=str(gender or "All"),
            income_annual_inr=int(income or 250000),
            caste_category=str(caste or "General"),
            occupation=str(occupation or ""),
        )
        with RAGQueryLogger("scheme_eligibility") as tracker:
            tracker.log_language(lang)
            candidates_df, explanation = check_eligibility(profile, rt.retriever)

        full_response = format_scheme_response(explanation, candidates_df)
        status = f"✅ Found {len(candidates_df)} scheme(s) in database" if not candidates_df.empty else "✅ RAG-based scheme search complete"
        return full_response, status
    except Exception as e:
        logger.exception("run_scheme_check")
        return f"**Error:** {e}", "❌ Check failed"


# ---------------------------------------------------------------------------
# IPC → BNS Comparison tab handlers
# ---------------------------------------------------------------------------

def run_ipc_bns_compare(
    ipc_input: str,
    lang: str,
) -> tuple[str, str, str, str]:
    """Run IPC→BNS comparison and return (ipc_md, bns_md, analysis_md, diff_text)."""
    if not _IPC_BNS_OK:
        msg = "**IPC↔BNS module not available.** Please check installation."
        return msg, "", "", ""

    if not ipc_input.strip():
        return "*Enter an IPC section number to begin.*", "", "", ""

    try:
        rt = get_runtime()
        rt.load()
        with RAGQueryLogger("ipc_bns_compare") as tracker:
            tracker.log_language(lang)
            result = compare_ipc_to_bns(ipc_input.strip(), rt.retriever)

        return format_comparison_response(result)
    except Exception as e:
        logger.exception("run_ipc_bns_compare")
        return f"**Error:** {e}", "", "", ""


# ---------------------------------------------------------------------------
# App builder
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    custom_css = """
    /* Cream-white justice theme */
    :root {
        --mulia-bg-top: #FFFDF7;
        --mulia-bg-bottom: #F8F3E7;
        --mulia-surface: #FFFCF4;
        --mulia-border: #E6DDC7;
        --mulia-text: #2B2A28;
        --mulia-muted: #4C5C73;
        --mulia-accent: #8C6A2F;
        --mulia-accent-strong: #7B5E29;
        --mulia-title: #1F2F45;
        --mulia-tab-idle: #EFE6D2;
        --mulia-tab-hover: #E6D8BC;
        --mulia-tab-active: #D8C7A3;
    }

    .gradio-container {
        background: linear-gradient(180deg, var(--mulia-bg-top) 0%, var(--mulia-bg-bottom) 100%) !important;
        color: var(--mulia-text) !important;
    }

    .gradio-container,
    .gradio-container p,
    .gradio-container span,
    .gradio-container label,
    .gradio-container .prose,
    .gradio-container .prose * {
        color: var(--mulia-text);
    }

    h1, h2, h3 {
        color: var(--mulia-title) !important;
        font-family: Georgia, "Times New Roman", serif;
        letter-spacing: 0.01em;
    }

    .gr-button-primary {
        background: var(--mulia-accent) !important;
        border-color: var(--mulia-accent-strong) !important;
        color: #FFFDF8 !important;
    }

    .gr-button-primary:hover {
        background: var(--mulia-accent-strong) !important;
        border-color: #6B5225 !important;
    }

    .gr-box, .gr-panel, .gr-form, .gradio-container .block {
        background-color: var(--mulia-surface) !important;
        border-color: var(--mulia-border) !important;
    }

    .gr-input, .gr-textbox, textarea, input, select {
        background-color: #FFFDF8 !important;
        border-color: #D8CCB2 !important;
        color: var(--mulia-text) !important;
    }

    /* Tabs: enforce strong contrast across idle/hover/active states */
    .tabs,
    .tabs .tab-nav {
        background: transparent !important;
    }

    .tabs .tab-nav button {
        background: var(--mulia-tab-idle) !important;
        color: #3A3326 !important;
        border: 1px solid #D3C2A0 !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 10px 10px 0 0 !important;
        margin-right: 6px !important;
        opacity: 1 !important;
    }

    .tabs .tab-nav button:hover {
        background: var(--mulia-tab-hover) !important;
        color: #2E2A20 !important;
    }

    .tabs .tab-nav button.selected,
    .tabs .tab-nav button[aria-selected="true"] {
        background: var(--mulia-tab-active) !important;
        color: #1F2F45 !important;
        border-color: #BCA77F !important;
        border-bottom: 3px solid var(--mulia-accent) !important;
    }

    .tabs .tabitem,
    .tabs .tabitem > div {
        background: var(--mulia-surface) !important;
        color: var(--mulia-text) !important;
    }

    /* Chat readability: white text with high-contrast bubbles */
    .chatbot .message.user,
    .chatbot .message.bot,
    [data-testid="chatbot"] .message.user,
    [data-testid="chatbot"] .message.bot {
        background: #2F3E58 !important;
        border: 1px solid #23324A !important;
    }

    .chatbot .message.user *,
    .chatbot .message.bot *,
    [data-testid="chatbot"] .message.user *,
    [data-testid="chatbot"] .message.bot * {
        color: #FFFFFF !important;
    }

    .chatbot .message a,
    [data-testid="chatbot"] .message a {
        color: #F3E6B8 !important;
    }

    footer {
        font-size: 0.85rem;
        color: var(--mulia-muted);
    }

    .diff-box textarea { font-family: monospace !important; font-size: 0.82rem; }
    """

    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="amber", secondary_hue="slate"),
        css=custom_css,
        title="MULIA - Multilingual Legal Information Assistant",
    ) as demo:
        gr.Markdown(
            "# MULIA · Multilingual Legal Information Assistant\n"
            "*Governance & Access to Justice — Legal information assistant for India · "
            "Not a substitute for legal counsel*"
        )

        lang_state = gr.State("en")

        # ── Welcome screen ──────────────────────────────────────────────────
        with gr.Column(visible=True) as welcome_col:
            gr.Markdown("### Welcome / स्वागत है")
            lang_radio = gr.Radio(
                choices=[(c[1], c[0]) for c in SARVAM_LANGUAGES],
                value="en",
                label="Select your language / अपनी भाषा चुनें",
                info=(
                    "Non-English questions are translated to English for retrieval, "
                    "then answers are translated back to your language."
                ),
            )
            begin_btn = gr.Button("Begin / शुरू करें", variant="primary")
            gr.Markdown(
                "<small>Not a substitute for legal counsel · General information only · "
                "Powered by Sarvam (STT / translate / TTS) when configured</small>"
            )

        # ── Main area (hidden until Begin is clicked) ───────────────────────
        with gr.Column(visible=False) as main_col:
            current_lang = gr.Markdown("*Session language: English*")

            with gr.Tabs():

                # ── Tab 1: Legal Q&A (original chat) ──────────────────────
                with gr.TabItem("⚖️ Legal Q&A"):
                    topic = gr.Radio(
                        choices=list(TOPIC_SEEDS.keys()),
                        label="Common topics",
                        value=None,
                    )
                    chatbot = gr.Chatbot(
                        label="MULIA",
                        height=420,
                        bubble_full_width=False,
                    )
                    msg = gr.Textbox(
                        placeholder="Type your legal question in any supported language…",
                        show_label=False,
                        lines=2,
                    )
                    audio_in = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        label="Or speak your question",
                    )
                    tts_cb = gr.Checkbox(label="Read answer aloud", value=True)
                    tts_out = gr.Audio(
                        label="Listen to answer",
                        type="numpy",
                        interactive=False,
                    )
                    submit = gr.Button("Send", variant="primary")

                # ── Tab 2: BNS Explainer ────────────────────────────────────
                with gr.TabItem("📖 BNS Explainer"):
                    gr.Markdown(
                        "### Bharatiya Nyaya Sanhita (BNS) Explainer\n"
                        "Ask any question about the BNS 2023. Get a structured answer including "
                        "Hindi key terms, IPC predecessor, and practical impact."
                    )
                    if not _BNS_EXPLAINER_OK:
                        gr.Markdown("⚠️ **BNS Explainer module unavailable** — check installation logs.")

                    bns_topic = gr.Radio(
                        choices=list(BNS_TOPIC_SEEDS.keys()),
                        label="Quick topics",
                        value=None,
                    )
                    bns_chatbot = gr.Chatbot(
                        label="BNS Explainer",
                        height=400,
                        bubble_full_width=False,
                    )
                    bns_query = gr.Textbox(
                        placeholder="e.g. What is BNS Section 303? How does BNS handle rape cases?",
                        show_label=False,
                        lines=2,
                    )
                    bns_submit = gr.Button("Explain", variant="primary")

                # ── Tab 3: Scheme Eligibility Checker ──────────────────────
                with gr.TabItem("🏛️ Scheme Finder"):
                    gr.Markdown(
                        "### Government Scheme Eligibility Checker\n"
                        "Fill in your profile to find central and state government welfare schemes "
                        "you may be eligible for."
                    )
                    if not _SCHEME_CHECKER_OK:
                        gr.Markdown("⚠️ **Scheme Checker module unavailable** — check installation logs.")

                    with gr.Row():
                        with gr.Column(scale=1):
                            scheme_state = gr.Dropdown(
                                choices=INDIAN_STATES if _SCHEME_CHECKER_OK else ["All India"],
                                value="All India",
                                label="State / UT",
                            )
                            scheme_age = gr.Number(
                                value=30,
                                label="Age (years)",
                                minimum=0,
                                maximum=120,
                                precision=0,
                            )
                            scheme_gender = gr.Radio(
                                choices=GENDER_OPTIONS if _SCHEME_CHECKER_OK else ["All"],
                                value="All",
                                label="Gender",
                            )
                            scheme_income = gr.Number(
                                value=250000,
                                label="Annual Income (₹)",
                                minimum=0,
                                precision=0,
                            )
                            scheme_caste = gr.Dropdown(
                                choices=CASTE_CATEGORIES if _SCHEME_CHECKER_OK else ["General"],
                                value="General",
                                label="Category (Caste)",
                            )
                            scheme_occupation = gr.Dropdown(
                                choices=([""] + OCCUPATION_TAGS) if _SCHEME_CHECKER_OK else [""],
                                value="",
                                label="Occupation (optional)",
                                allow_custom_value=True,
                            )
                            scheme_btn = gr.Button("Find Eligible Schemes", variant="primary")
                            scheme_status = gr.Markdown("")

                        with gr.Column(scale=2):
                            scheme_result = gr.Markdown(
                                "*Fill in your profile and click 'Find Eligible Schemes'.*"
                            )

                # ── Tab 4: IPC → BNS Comparison ────────────────────────────
                with gr.TabItem("⚡ IPC ↔ BNS"):
                    gr.Markdown(
                        "### IPC → BNS Clause Comparison Tool\n"
                        "Enter an IPC section number to see the BNS equivalent, a textual diff, "
                        "and an AI-generated analysis of what changed."
                    )
                    if not _IPC_BNS_OK:
                        gr.Markdown("⚠️ **IPC↔BNS module unavailable** — check installation logs.")

                    with gr.Row():
                        ipc_input = gr.Textbox(
                            placeholder="e.g. 302, IPC 378, Section 420",
                            label="IPC Section Number",
                            lines=1,
                            scale=3,
                        )
                        ipc_btn = gr.Button("Compare", variant="primary", scale=1)

                    with gr.Row():
                        ipc_text_out = gr.Markdown(
                            "*IPC text will appear here.*",
                            label="IPC 1860",
                        )
                        bns_text_out = gr.Markdown(
                            "*BNS equivalent will appear here.*",
                            label="BNS 2023",
                        )

                    ipc_analysis_out = gr.Markdown("*Analysis will appear here.*")
                    ipc_diff_out = gr.Textbox(
                        label="Unified Diff (IPC → BNS)",
                        lines=12,
                        interactive=False,
                        elem_classes=["diff-box"],
                    )

                # ── Tab 5: About ────────────────────────────────────────────
                with gr.TabItem("ℹ️ About"):
                    gr.Markdown("""
## MULIA · Multilingual Legal Information Assistant

**Governance & Access to Justice** — making Indian law accessible to every citizen.

### What's Inside

| Feature | Description |
|---------|-------------|
| ⚖️ Legal Q&A | Multilingual RAG chatbot for general Indian law questions |
| 📖 BNS Explainer | Structured explainer for Bharatiya Nyaya Sanhita (BNS) 2023 with Hindi terms and IPC comparison |
| 🏛️ Scheme Finder | Government welfare scheme eligibility checker (MyScheme database) |
| ⚡ IPC ↔ BNS | Clause-level comparison between IPC 1860 and BNS 2023 with AI analysis |

### Data Sources
- Bharatiya Nyaya Sanhita (BNS) 2023
- Indian Penal Code (IPC) 1860
- Constitution of India
- Government Welfare Schemes (MyScheme.gov.in)
- IPC→BNS Transition Mapping

### Databricks Architecture
- **Delta Lake**: Versioned legal corpus (legal_rag_corpus, constitution_articles, gov_welfare_schemes, ipc_sections)
- **Apache Spark / PySpark**: Data ingestion pipelines
- **SparkML**: TF-IDF scheme matching pipeline
- **Databricks Vector Search**: Managed semantic retrieval
- **Databricks AI Gateway**: Llama 4 Maverick (LLM inference)
- **MLflow**: Experiment tracking, model registry
- **Unity Catalog**: Governed data storage
- **Databricks Apps**: Application hosting

### Languages Supported
English, Hindi, Bengali, Telugu, Marathi, Tamil, Gujarati, Kannada, Malayalam, Punjabi, Odia, Urdu, Assamese

---
*Powered by Databricks (Llama Maverick + Vector Search + Delta Lake + MLflow) · Sarvam AI (translation, speech)*

**⚠️ Disclaimer**: This tool provides legal information, not legal advice. Always consult a qualified lawyer for your specific situation.
""")

        # ── Event wiring ────────────────────────────────────────────────────

        def on_begin(lang_code: str):
            labels = dict(SARVAM_LANGUAGES)
            label = labels.get(lang_code, lang_code)
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                lang_code,
                f"*Session language: {label}*",
            )

        begin_btn.click(
            on_begin,
            inputs=[lang_radio],
            outputs=[welcome_col, main_col, lang_state, current_lang],
        )

        # Legal Q&A tab
        def fill_topic(choice: str | None):
            if not choice:
                return gr.update()
            seed = TOPIC_SEEDS.get(choice, "")
            return gr.update(value=seed)

        topic.change(fill_topic, inputs=[topic], outputs=[msg])

        _run_turn_io = dict(
            fn=run_turn,
            inputs=[msg, audio_in, chatbot, lang_state, tts_cb],
            outputs=[msg, chatbot, tts_out, audio_in],
        )
        submit.click(**_run_turn_io)
        msg.submit(**_run_turn_io)
        audio_in.stop_recording(**_run_turn_io)

        # BNS Explainer tab
        def fill_bns_topic(choice: str | None):
            if not choice:
                return gr.update()
            return gr.update(value=BNS_TOPIC_SEEDS.get(choice, ""))

        bns_topic.change(fill_bns_topic, inputs=[bns_topic], outputs=[bns_query])

        _bns_io = dict(
            fn=run_bns_turn,
            inputs=[bns_query, bns_chatbot, lang_state],
            outputs=[bns_query, bns_chatbot],
        )
        bns_submit.click(**_bns_io)
        bns_query.submit(**_bns_io)

        # Scheme Finder tab
        scheme_btn.click(
            fn=run_scheme_check,
            inputs=[scheme_state, scheme_age, scheme_gender, scheme_income,
                    scheme_caste, scheme_occupation, lang_state],
            outputs=[scheme_result, scheme_status],
        )

        # IPC ↔ BNS tab
        ipc_btn.click(
            fn=run_ipc_bns_compare,
            inputs=[ipc_input, lang_state],
            outputs=[ipc_text_out, bns_text_out, ipc_analysis_out, ipc_diff_out],
        )
        ipc_input.submit(
            fn=run_ipc_bns_compare,
            inputs=[ipc_input, lang_state],
            outputs=[ipc_text_out, bns_text_out, ipc_analysis_out, ipc_diff_out],
        )

        gr.Markdown(
            "<small>Powered by Databricks (Llama Maverick + Vector Search + Delta Lake + MLflow) · "
            "Sarvam AI (translation, speech-to-text, text-to-speech)</small>"
        )

    return demo


# ---------------------------------------------------------------------------
# Secrets loader (Databricks Apps)
# ---------------------------------------------------------------------------

def _load_secrets_from_scope() -> None:
    """Load secrets from Databricks secret scope into env vars (for Databricks Apps)."""
    # Local runs often set only DATABRICKS_TOKEN for LLM calls. Databricks SDK
    # requires a workspace host (or another complete auth method) for secrets APIs.
    if os.environ.get("DATABRICKS_TOKEN", "").strip() and not os.environ.get("DATABRICKS_HOST", "").strip():
        logger.info("Skipping secret-scope lookup: DATABRICKS_HOST is not set for SDK auth.")
        return

    mapping = {
        "SARVAM_API_KEY": ("nyaya-dhwani", "sarvam_api_key"),
    }
    for env_var, (scope, key) in mapping.items():
        if os.environ.get(env_var, "").strip():
            continue
        try:
            from databricks.sdk import WorkspaceClient
            w = WorkspaceClient()
            val = w.secrets.get_secret(scope=scope, key=key)
            if val and val.value:
                import base64
                try:
                    decoded = base64.b64decode(val.value).decode("utf-8")
                except Exception:
                    decoded = val.value
                os.environ[env_var] = decoded
                logger.info("Loaded %s from secret scope %s/%s", env_var, scope, key)
        except Exception as exc:
            logger.warning("Could not load %s from secret scope: %s", env_var, exc)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    _load_secrets_from_scope()
    demo = build_app()
    demo.queue()
    demo.launch(share=True)


if __name__ == "__main__":
    main()
