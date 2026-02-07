FROM python:3.11-slim AS builder

WORKDIR /app

# Install system build dependencies required for some Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and build wheels / install packages into /usr/local (builder)
COPY services/topic-classifier-service/requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir --prefix /usr/local -r requirements.txt

# Download models during build (creates cached layer)
COPY services/topic-classifier-service/download_models.py .
RUN python download_models.py

# NOTE: Cannot uninstall torch - optimum.onnxruntime requires it for imports
# Even though ONNX Runtime doesn't use PyTorch at runtime, the library needs it to load
# Image size: ~3.2GB (unavoidable with current dependencies)

## Final runtime image
FROM python:3.11-slim

WORKDIR /app

# Install minimal runtime system deps
RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local /usr/local

# Copy application code
COPY services/topic-classifier-service/app/ ./app/

# Copy TOPIC_TAXONOMY from models directory
COPY models/TOPIC_TAXONOMY.py ./models/TOPIC_TAXONOMY.py

# Expose port
EXPOSE 8009

# Health check (for API service)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8009/api/v1/health')" || exit 1

# Default: Run the API service
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8009"]
