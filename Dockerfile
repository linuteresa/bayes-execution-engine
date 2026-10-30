FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    JOB_STORE=sqlite \
    JOB_DB_PATH=/app/data/jobs.sqlite3

WORKDIR /app
ARG INSTALL_REDIS=false
ARG INSTALL_POSTGRES=false
ARG INSTALL_TELEMETRY=false
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN if [ "$INSTALL_REDIS" = "true" ]; then pip install -r requirements-redis.txt; fi
RUN if [ "$INSTALL_POSTGRES" = "true" ]; then pip install -r requirements-postgres.txt; fi
RUN if [ "$INSTALL_TELEMETRY" = "true" ]; then pip install -r requirements-telemetry.txt; fi
COPY . .
RUN useradd --create-home appuser && mkdir -p /app/data && chown -R appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)"
CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000"]
