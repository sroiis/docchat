# Hugging Face Spaces build: clone the app, install, run on port 7860.
#
# The Space repo contains ONLY this wrapper (see README.md frontmatter).
# The application itself lives in the GitHub repo cloned below, so the Space
# is always built from the latest committed code.

FROM python:3.11-slim

WORKDIR /app

# node is required to build the React frontend.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl ca-certificates nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# Clone the application at build time.
ARG DOCCHAT_GITHUB_REPO=https://github.com/sroiis/docchat.git
RUN git clone --depth 1 "$DOCCHAT_GITHUB_REPO" .

RUN pip install --no-cache-dir -r requirements.txt

# Build the frontend.
RUN cd frontend && npm install && npm run build

# Runtime configuration. Secrets (Groq key, secret key) are injected by HF
# Spaces as environment variables; keep defaults here safe and free.
ENV DOCCHAT_DB_PATH=/app/data/docchat.db \
    DOCCHAT_FRONTEND_DIST=/app/frontend/dist \
    DOCCHAT_EMBEDDING_PROVIDER=tfidf \
    DOCCHAT_LLM_PROVIDER=openai \
    DOCCHAT_OPENAI_BASE_URL=https://api.groq.com/openai/v1 \
    DOCCHAT_LLM_MODEL=llama-3.3-70b-versatile \
    DOCCHAT_AUTH_ENABLED=true

# HF Spaces routes traffic to the PORT env var (7860 on free tier).
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
