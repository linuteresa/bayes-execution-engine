# syntax=docker/dockerfile:1
# Container for the async Bayes Execution Engine API.
# NOTE: the LLM itself (llama.cpp's llama-server) runs as a SEPARATE container/process.
# This image talks to it over HTTP via LLAMA_CPP_BASE_URL, so the GGUF weights never
# bloat the app image. See docker-compose.yml for the two-service topology.

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application.
COPY . .

# Drop root.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000"]
