"""Streamlit entry point for the Collegesaurus AI chatbot.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import secrets
import time

import streamlit as st

import config
from agent import build_chat
from logging_store import log_turn


# ---------------------------------------------------------------------------
# UI strings by language. Codes match _LANG_INSTRUCTIONS in agent.py.
# ---------------------------------------------------------------------------

UI = {
    "en": {
        "display": "English",
        "page_title": "Collegesaurus AI",
        "title": "🦕 Collegesaurus AI",
        "caption": "Ask about universities and scholarships in Lebanon.",
        "language_label": "Language",
        "about_heading": "### About",
        "about_body": (
            "Collegesaurus AI answers questions about Lebanese universities "
            "and external scholarships. It retrieves from the same pages you "
            "see on [the site](%s)."
        ),
        "try_heading": "### Try asking",
        "try_body": (
            "- What are AUB's tuition rates?\n"
            "- Which scholarships fund study abroad for Lebanese students?\n"
            "- List all universities you know about.\n"
            "- What does USJ require for a medical degree?"
        ),
        "clear_btn": "Clear conversation",
        "input_placeholder": "Ask about Lebanese universities or scholarships…",
        "thinking": "Thinking…",
        "no_index": (
            "No vector index found. Run `python ingest.py` first (with "
            "`GEMINI_API_KEY` set) to build it."
        ),
        "input_too_long": "Question too long — keep it under {max} characters.",
        "rate_limited": "Too many questions too fast — try again in 30 seconds.",
    },
    "ar": {
        "display": "العربية",
        "page_title": "Collegesaurus AI",
        "title": "🦕 كوليجسورس الذكي",
        "caption": "اسأل عن الجامعات والمنح في لبنان.",
        "language_label": "اللغة",
        "about_heading": "### عن التطبيق",
        "about_body": (
            "يُجيب كوليجسورس الذكي عن الأسئلة المتعلقة بالجامعات اللبنانية "
            "والمنح الخارجية. يستند إلى نفس صفحات [الموقع](%s)."
        ),
        "try_heading": "### جرّب أن تسأل",
        "try_body": (
            "- ما هي أقساط الجامعة الأمريكية في بيروت (AUB)؟\n"
            "- ما هي المنح التي تموّل الدراسة في الخارج للطلاب اللبنانيين؟\n"
            "- اذكر لي كل الجامعات التي تعرفها.\n"
            "- ما متطلبات كلية الطب في USJ؟"
        ),
        "clear_btn": "مسح المحادثة",
        "input_placeholder": "اسأل عن الجامعات اللبنانية أو المنح…",
        "thinking": "جارٍ التفكير…",
        "no_index": (
            "لم يتم العثور على فهرس المتجهات. شغّل `python ingest.py` أولاً "
            "(مع ضبط `GEMINI_API_KEY`) لبناء الفهرس."
        ),
        "input_too_long": "السؤال طويل جدًا — أبقه ضمن {max} حرفًا.",
        "rate_limited": "أسئلة كثيرة خلال وقت قصير — حاول بعد 30 ثانية.",
    },
    "auto": {
        "display": "Auto / تلقائي",
        "page_title": "Collegesaurus AI",
        "title": "🦕 Collegesaurus AI",
        "caption": "Ask about universities and scholarships in Lebanon. · "
                   "اسأل عن الجامعات والمنح في لبنان.",
        "language_label": "Language / اللغة",
        "about_heading": "### About",
        "about_body": (
            "Answers questions about Lebanese universities and external "
            "scholarships from [the site](%s). Write in English or Arabic "
            "and the bot will match you."
        ),
        "try_heading": "### Try asking",
        "try_body": (
            "- What are AUB's tuition rates?\n"
            "- ما هي أقساط AUB؟\n"
            "- Which scholarships fund study abroad?\n"
            "- List all universities you know about."
        ),
        "clear_btn": "Clear conversation",
        "input_placeholder": "Ask about Lebanese universities or scholarships…",
        "thinking": "Thinking…",
        "no_index": (
            "No vector index found. Run `python ingest.py` first (with "
            "`GEMINI_API_KEY` set) to build it."
        ),
        "input_too_long": "Question too long — keep it under {max} characters.",
        "rate_limited": "Too many questions too fast — try again in 30 seconds.",
    },
}


LANG_OPTIONS = [("auto", UI["auto"]["display"]),
                ("en",   UI["en"]["display"]),
                ("ar",   UI["ar"]["display"])]


st.set_page_config(
    page_title="Collegesaurus AI",
    page_icon="🦕",
    layout="centered",
    initial_sidebar_state="expanded",
)


def _apply_direction(direction: str) -> None:
    """Flip the whole Streamlit app LTR or RTL via injected CSS.

    Streamlit has no built-in dir switch. Setting `direction: rtl` on
    `.stApp` cascades to chat bubbles, markdown, sidebar, and chat input —
    everything that isn't rendered inside an iframe.
    """
    css = (
        ".stApp { direction: rtl; text-align: right; }"
        if direction == "rtl"
        else ".stApp { direction: ltr; text-align: left; }"
    )
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _ensure_index_exists() -> bool:
    """Return True if the Chroma directory looks populated."""
    return config.CHROMA_DIR.exists() and any(config.CHROMA_DIR.iterdir())


def _reset_chat(lang: str) -> None:
    st.session_state.chat = build_chat(lang)
    st.session_state.messages = []


def sidebar(ui: dict) -> None:
    with st.sidebar:
        # Language picker: drives both UI strings AND layout direction.
        # Arabic → RTL + Arabic chrome. English / Auto → LTR + English chrome.
        current_lang = st.session_state.get("lang", "auto")
        labels = [label for _, label in LANG_OPTIONS]
        idx = next(
            (i for i, (code, _) in enumerate(LANG_OPTIONS) if code == current_lang),
            0,
        )
        picked_label = st.radio(
            ui["language_label"], labels, index=idx, horizontal=True
        )
        picked_code = next(
            code for code, label in LANG_OPTIONS if label == picked_label
        )
        if picked_code != current_lang:
            st.session_state.lang = picked_code
            _reset_chat(picked_code)
            st.rerun()

        st.markdown(ui["about_heading"])
        st.markdown(ui["about_body"] % config.SITE_BASE_URL)
        st.markdown(ui["try_heading"])
        st.markdown(ui["try_body"])
        if st.button(ui["clear_btn"], use_container_width=True):
            _reset_chat(st.session_state.get("lang", "auto"))
            st.rerun()


def main() -> None:
    if "lang" not in st.session_state:
        st.session_state.lang = "auto"
    if "session_id" not in st.session_state:
        st.session_state.session_id = secrets.token_hex(8)
    lang = st.session_state.lang
    ui = UI[lang]

    # Direction follows language: Arabic → RTL, everything else LTR.
    _apply_direction("rtl" if lang == "ar" else "ltr")

    st.title(ui["title"])
    st.caption(ui["caption"])

    if not _ensure_index_exists():
        st.error(ui["no_index"])
        st.stop()

    if "chat" not in st.session_state:
        st.session_state.chat = build_chat(lang)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    sidebar(ui)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input(ui["input_placeholder"])
    if not prompt:
        return

    prompt = prompt.strip()
    if not prompt:
        return

    if len(prompt) > config.MAX_INPUT_CHARS:
        st.error(ui["input_too_long"].format(max=config.MAX_INPUT_CHARS))
        return

    now = time.time()
    hits = [
        t
        for t in st.session_state.get("rate_hits", [])
        if now - t < config.RATE_LIMIT_WINDOW_SECONDS
    ]
    if len(hits) >= config.RATE_LIMIT_MAX_PER_WINDOW:
        st.warning(ui["rate_limited"])
        return
    hits.append(now)
    st.session_state["rate_hits"] = hits

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(ui["thinking"]):
            result = st.session_state.chat.send(prompt)
        st.markdown(result.answer)

    st.session_state.messages.append({"role": "assistant", "content": result.answer})

    log_turn(
        session_id=st.session_state.session_id,
        lang=lang,
        question=prompt,
        answer=result.answer,
        tool_calls=result.tool_calls,
        latency_ms=result.latency_ms,
        error=result.error,
    )


if __name__ == "__main__":
    main()
