"""Build the agentic-RAG agent used by the Streamlit app."""

from __future__ import annotations

from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI

import config
from tools import build_tools


SYSTEM_PROMPT = """You are Collegesaurus, an assistant that helps Lebanese high-school
students (and their parents) choose a university or find a scholarship.

Ground rules:
1. ALWAYS use the provided tools before answering. Do not invent university
   names, majors, tuition figures, deadlines, or contact details — look them up.
2. When you cite a fact, include the source URL from the tool result.
3. If the tools return no useful matches, say so plainly. Don't guess.
4. Stay in scope: Lebanese higher education and the scholarships our dataset
   covers. If a question is off-topic, say the site focuses on Lebanese
   universities and external scholarships, and suggest what you can help with.
5. Prefer concise, structured answers: short paragraphs, bullet lists for
   options, a deadline or cost in bold if relevant.
6. You may be asked in Arabic or English — reply in the same language the
   user used.
"""


def build_agent() -> ReActAgent:
    api_key = config.require_api_key()

    llm = GoogleGenAI(model=config.GEMINI_CHAT_MODEL, api_key=api_key)
    embed_model = GoogleGenAIEmbedding(
        model_name=config.GEMINI_EMBED_MODEL, api_key=api_key
    )

    # Register globally so tools that instantiate indices pick them up.
    Settings.llm = llm
    Settings.embed_model = embed_model

    return ReActAgent.from_tools(
        tools=build_tools(),
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
        verbose=False,
        max_iterations=6,
    )
