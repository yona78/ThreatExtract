# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System deps: libmagic for python-magic content sniffing; curl for healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libmagic1 curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Belt-and-suspenders: the app also loads with local_files_only=True, but
    # these guarantee transformers/hub never touch the network at runtime.
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

WORKDIR /app

# Install dependencies first for better layer caching.
# On x86_64 we pull the CPU-only torch wheel to keep the image small; on arm64
# the default wheel is already CPU, so we skip the extra index there.
ARG TARGETARCH
COPY requirements.txt .
RUN if [ "$TARGETARCH" = "amd64" ]; then \
        pip install --no-cache-dir torch==2.5.1 \
            --index-url https://download.pytorch.org/whl/cpu ; \
    fi \
    && pip install --no-cache-dir -r requirements.txt

# Pre-downloaded weights for the CHOSEN model (SecureBERT-NER). Run
# `python download_model.py` BEFORE building so the directory below exists; this
# COPY is the ONLY way the model enters the image — nothing is ever downloaded
# at build or runtime. Only the chosen model is baked in, keeping the image lean
# and fully self-contained. Swap models by re-downloading and editing the path
# here, or by mounting model_cache/ at runtime (see docker-compose.yml).
COPY model_cache/CyberPeace-Institute__SecureBERT-NER/ /app/model_cache/CyberPeace-Institute__SecureBERT-NER/

# Application code.
COPY src/ /app/src/
COPY assets/ /app/assets/
COPY .streamlit/ /app/.streamlit/
COPY app.py /app/app.py

# Default model to load; override via docker-compose or `-e MODEL_PATH=...`
# to dynamically swap models without touching the code.
ENV MODEL_PATH=/app/model_cache/CyberPeace-Institute__SecureBERT-NER

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
