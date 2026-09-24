# The FastAPI chatbot service (chatbot/). The Streamlit app at the repo root
# is not part of this image.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY chatbot/requirements.txt chatbot/requirements.txt
RUN pip install -r chatbot/requirements.txt

COPY chatbot/ chatbot/
# Index snapshot built by the deploy workflow (python -m chatbot.ingest). If it
# is missing, the app builds its index from the live site on its first poll.
COPY data/ data/

RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

ENV INDEX_PATH=/app/data/index.db
EXPOSE 8000
CMD ["uvicorn", "chatbot.server:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
