# ---------- Dockerfile (GPU inference) ----------
# Use NVIDIA CUDA runtime base that matches the PyTorch +cu118 wheel
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Metadata / env
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install python3.11, pip and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip python3.11-distutils \
    build-essential \
    ca-certificates \
    curl \
    git \
    libssl-dev \
    libffi-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Ensure `python` and `pip` point to python3.11
RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    ln -sf /usr/bin/pip3 /usr/local/bin/pip

# Upgrade pip, setuptools, wheel
RUN python -m pip install --upgrade pip setuptools wheel

# Copy only requirements first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Install Python dependencies EXCEPT torch packages
RUN pip install --no-cache-dir -r /app/requirements.txt


# (Optional) download only necessary nltk corpora
RUN python - <<'PY'
import nltk
nltk.download('punkt')
PY

# Copy application code
COPY . /app

# Remove any existing .egg-info directories to avoid pip metadata conflicts
RUN find /app -name "*.egg-info" -exec rm -rf {} +

# Create required directories
RUN mkdir -p data/raw data/processed \
    models/trained_models models/checkpoints \
    final_models/trained_models final_models/vectorizers \
    artifacts logs training_results

# Create non-root user for running the app (recommended)
RUN useradd --create-home --shell /bin/bash appuser && chown -R appuser:appuser /app
USER appuser

# Expose HTTP port
EXPOSE 8000

# Simple healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start command (uvicorn as example FastAPI app)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
