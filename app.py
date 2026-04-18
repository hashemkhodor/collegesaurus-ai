"""Streamlit entry point for the Collegesaurus AI chatbot.

Run locally:
    streamlit run app.py

Deploys to Streamlit Community Cloud unchanged — point the deploy at this
file, add GEMINI_API_KEY under "Secrets", and commit the `chroma_db/`
directory (or re-ingest on first boot).
"""

from __future__ import annotations

import streamlit as st

import config
from agent import build_agent

st.set_page_config(
    page_title="Collegesaurus AI",
    page_icon="🦕",
    layout="centered",
    initial_sidebar_state="collapsed",
)


def _ensure_index_exists() -> bool:
    """Return True if the Chroma directory looks populated."""
    return config.CHROMA_DIR.exists() and any(config.CHROMA_DIR.iterdir())


@st.cache_resource(show_spinner="Waking Collegesaurus up…")
def get_agent():
    return build_agent()


def sidebar():
    with st.sidebar:
        st.markdown("### About")
        st.markdown(
            "Collegesaurus AI answers questions about Lebanese universities "
            "and external scholarships. It retrieves from the same pages you "
            "see on [the site](%s)." % config.SITE_BASE_URL
        )
        st.markdown("### Try asking")
        st.markdown(
            "- What are AUB's tuition rates?\n"
            "- Which scholarships fund study abroad for Lebanese students?\n"
            "- List all universities you know about.\n"
            "- What does USJ require for a medical degree?"
        )
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.pop("messages", None)
            st.rerun()


def main() -> None:
    st.title("🦕 Collegesaurus AI")
    st.caption("Ask about universities and scholarships in Lebanon.")

    if not _ensure_index_exists():
        st.error(
            "No vector index found. Run `python ingest.py` first (with "
            "`GEMINI_API_KEY` set) to build it."
        )
        st.stop()

    sidebar()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask about Lebanese universities or scholarships…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    agent = get_agent()
    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("Thinking…"):
            response = agent.chat(prompt)
        answer = str(response)
        placeholder.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
