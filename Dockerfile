# ASTRO RUNTIME IMAGE
# Version 11.20.0 uses Python 3.11 (Matches your local 'saenv')
FROM quay.io/astronomer/astro-runtime:11.20.0

# 1. Switch to root to install system dependencies
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    libffi-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Switch back to astro user
USER astro

# 3. Set PYTHONPATH so Airflow can natively find your 'src' folder
ENV PYTHONPATH="${PYTHONPATH}:/usr/local/airflow"

# 4. Install Python dependencies
COPY requirements-airflow.txt .
RUN pip install --no-cache-dir -r requirements-airflow.txt

# 5. Download NLTK Data (UPDATED)
# We added 'stopwords' and 'wordnet' which are required for preprocessing
RUN python - <<'PY'
import nltk
try:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
except:
    pass
PY

# 6. Security: Allow custom Artifact classes to be passed between tasks (XCom)
ENV AIRFLOW__CORE__ALLOWED_DESERIALIZATION_CLASSES="airflow.* src.sentiment_analysis.entity.artifact_entity.*"

# 7. Fix for "Failed to connect to bus" / DBus Error
# This tells Python libraries (like kagglehub) NOT to try using the system keyring
ENV PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring