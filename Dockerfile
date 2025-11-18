# ---------- Dockerfile (GPU inference) ----------
# Use NVIDIA CUDA runtime base that matches the PyTorch +cu118 wheel
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Metadata / env
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install python3.10, pip and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip python3.10-distutils \
    build-essential \
    ca-certificates \
    curl \
    git \
    libssl-dev \
    libffi-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*


# Ensure `python` and `pip` point to python3.10
RUN ln -sf /usr/bin/python3.10 /usr/local/bin/python && \
    ln -sf /usr/bin/pip3 /usr/local/bin/pip

# Upgrade pip, setuptools, wheel
RUN python -m pip install --upgrade pip setuptools wheel

# Copy only requirements first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
# NOTE: For PyTorch GPU wheels we expect requirements.txt to contain the +cu118 tags
RUN pip install --no-cache-dir -r /app/requirements.txt

# (Optional) download only necessary nltk corpora if your app uses NLTK
# Replace 'punkt' with the corpora you actually need.
RUN python - <<'PY'\n\
import nltk\n\
nltk.download('punkt')\n\
PY

# Copy application code
COPY . /app

# Create required directories
RUN mkdir -p data/raw data/processed \
    models/trained_models models/checkpoints \
    final_models/trained_models final_models/vectorizers final_models/preprocessing \
    artifacts logs training_results

# Install your package (editable mode if you need dev conveniences)
RUN pip install --no-cache-dir -e .

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
