**DagsHub MLflow Setup**

**Purpose:** Configure MLflow to log experiments to DagsHub without committing secrets to the repository.

- **Environment variables used by this project:**
  - `MLFLOW_TRACKING_URI` — set to your DagsHub MLflow endpoint, e.g. `https://dagshub.com/<owner>/<repo>.mlflow`
  - `MLFLOW_EXPERIMENT_NAME` — optional (defaults to `sentiment_analysis_lstm_attention`)
  - `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` — optional HTTP basic auth if required
  - `DAGSHUB_TOKEN` — optional personal access token (use only if your DagsHub setup requires it)

**How to set (example, do NOT paste tokens into files that get committed):**

1. Locally (for a shell session):

```bash
export MLFLOW_TRACKING_URI="https://dagshub.com/<owner>/<repo>.mlflow"
export MLFLOW_EXPERIMENT_NAME="sentiment_analysis_lstm_attention"
# If your repo is private, set either username/password or a token
export MLFLOW_TRACKING_USERNAME="<your-username>"
export MLFLOW_TRACKING_PASSWORD="<your-password-or-token>"
# or
export DAGSHUB_TOKEN="<your-token>"
```

2. With a `.env` file (DO NOT commit this file):

```dotenv
MLFLOW_TRACKING_URI=https://dagshub.com/<owner>/<repo>.mlflow
MLFLOW_EXPERIMENT_NAME=sentiment_analysis_lstm_attention
MLFLOW_TRACKING_USERNAME=<your-username>
MLFLOW_TRACKING_PASSWORD=<your-password-or-token>
```

**Notes:**
- The project contains a helper at `src/sentiment_analysis/utils/ml_utils/mlflow_config.py` which reads these env vars and calls `mlflow.set_tracking_uri()` and `mlflow.set_experiment()` before logging.
- Avoid committing credentials. Use CI secrets or a local `.env` kept out of git.
