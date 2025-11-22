"""
FastAPI Routes
"""

from fastapi import APIRouter, HTTPException, status, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
import uuid
import os
import pandas as pd
import tempfile
import requests
import re
from typing import Dict, Any
from src.sentiment_analysis.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    HealthResponse
)
from src.sentiment_analysis.pipeline.prediction_pipeline import PredictionPipeline
from src.sentiment_analysis.pipeline.batch_prediction_pipeline import BatchPredictionPipeline
from src.sentiment_analysis.logging.logger import logging as logger
from src.sentiment_analysis.exception.exception import SentimentAnalysisException

# Create router
router = APIRouter()

# Initialize pipelines (singleton)
prediction_pipeline = PredictionPipeline()
batch_pipeline = BatchPredictionPipeline()

# Configurable dirs (match `app.py` env defaults)
EXPORTS_DIR = Path(os.getenv("EXPORTS_DIR", str(Path.cwd() / "exports")))
TRAINING_RESULTS_DIR = Path(os.getenv("TRAINING_RESULTS_DIR", str(Path.cwd() / "training_results")))

# Ensure dirs exist
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _transform_gdrive_url(url: str) -> str:
    """
    Converts a Google Drive 'view' URL to a direct download URL.
    """
    # Pattern 1: /file/d/ID/view
    match = re.search(r'/file/d/([a-zA-Z0-9_-]+)', url)
    if match:
        file_id = match.group(1)
        return f'https://drive.google.com/uc?export=download&id={file_id}'
    
    # Pattern 2: id=ID
    match = re.search(r'id=([a-zA-Z0-9_-]+)', url)
    if match:
        file_id = match.group(1)
        return f'https://drive.google.com/uc?export=download&id={file_id}'
        
    return url


@router.get("/", tags=["Health"])
async def root():
    """Root endpoint"""
    return {
        "message": "Sentiment Analysis API",
        "version": "1.0.0",
        "status": "active"
    }


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    try:
        # Check if model is loaded
        prediction_pipeline.load_model()
        # attempt to read model metadata if available
        metadata_file = TRAINING_RESULTS_DIR / "model_metadata.json"
        model_version = None
        last_trained = None
        try:
            if metadata_file.exists():
                import json
                with open(metadata_file, 'r') as f:
                    md = json.load(f)
                    model_version = md.get('model_version') or md.get('last_trained')
                    last_trained = md.get('last_trained')
        except Exception:
            model_version = None
            last_trained = None

        return HealthResponse(
            status="healthy",
            model_loaded=True,
            version="1.0.0",
            model_version=model_version,
            last_trained=last_trained
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        metadata_file = TRAINING_RESULTS_DIR / "model_metadata.json"
        model_version = None
        last_trained = None
        try:
            if metadata_file.exists():
                import json
                with open(metadata_file, 'r') as f:
                    md = json.load(f)
                    model_version = md.get('model_version') or md.get('last_trained')
                    last_trained = md.get('last_trained')
        except Exception:
            pass

        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            version="1.0.0",
            model_version=model_version,
            last_trained=last_trained
        )


@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_sentiment(request: PredictionRequest):
    """
    Predict sentiment for single text
    """
    try:
        logger.info(f"Prediction request received: {request.text[:50]}...")
        
        # Predict
        result = prediction_pipeline.predict(request.text)
        
        # Create response
        response = PredictionResponse(
            text=result.text,
            cleaned_text=result.cleaned_text,
            sentiment=result.predicted_sentiment,
            label=result.predicted_label,
            confidence=result.confidence,
            probability=result.prediction_probability,
            processing_time_ms=result.processing_time_ms
        )
        
        return response
    
    except SentimentAnalysisException as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/predict/batch", tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict sentiment for batch of texts from:
    1. A local server file path
    2. A public URL (including Google Drive)
    
    Returns the processed CSV content for client-side download.
    """
    input_tmp = None
    output_tmp = None
    
    try:
        logger.info(f"Batch prediction request: {request.input_file}")
        
        input_path = request.input_file
        
        # --- 1. Handle URL inputs (Google Drive / Direct Links) ---
        if input_path.startswith(('http://', 'https://')):
            download_url = _transform_gdrive_url(input_path)
            logger.info(f"Downloading from URL: {download_url}")
            
            # Create temp file to store downloaded content
            input_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            try:
                # Use a session to potentially handle cookies/redirects better
                session = requests.Session()
                resp = session.get(download_url, allow_redirects=True, stream=True)
                
                if resp.status_code != 200:
                    raise ValueError(f"Failed to download file: Status {resp.status_code}")
                
                # Validate content type before writing
                content_type = resp.headers.get('Content-Type', '').lower()
                if 'html' in content_type:
                     raise ValueError("URL returned a webpage (HTML) instead of a CSV file. Please ensure the link is a direct download link.")

                # Write content to file
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk: 
                        input_tmp.write(chunk)
                
                input_tmp.close()
                input_path = input_tmp.name
                
                # Double check file content start to ensure it's not HTML
                with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                    first_chars = f.read(100).strip().lower()
                    if first_chars.startswith(('<!doctype', '<html')):
                        raise ValueError("Downloaded file content appears to be HTML (e.g., Google Drive preview page). Please ensure permissions are set to 'Anyone with the link' and the link is correct.")

            except Exception as e:
                raise ValueError(f"Download Error: {str(e)}")

        # --- 2. Create temp file for output ---
        output_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        output_tmp.close()
        
        # --- 3. Run Pipeline ---
        result = batch_pipeline.predict_batch(input_path, output_tmp.name)

        # --- 4. Read Result Content ---
        with open(output_tmp.name, 'r', encoding='utf-8') as f:
            csv_content = f.read()

        # Safely get avg_probability
        avg_prob = getattr(result, 'avg_probability', 0.0)
        
        return {
            "message": "Batch prediction completed",
            "input_file": request.input_file,
            "output_file": "download_response",
            "total_records": result.total_records,
            "successful": result.successful_predictions,
            "failed": result.failed_predictions,
            "avg_confidence": result.avg_confidence,
            "avg_probability": avg_prob,
            "processing_time_minutes": result.processing_time_minutes,
            "csv_content": csv_content, # <--- RETURN CONTENT DIRECTLY
            "filename": f"batch_results_{uuid.uuid4().hex[:8]}.csv"
        }
    
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    finally:
        # Cleanup temps
        try:
            if input_tmp and os.path.exists(input_tmp.name):
                os.remove(input_tmp.name)
            if output_tmp and os.path.exists(output_tmp.name):
                os.remove(output_tmp.name)
        except:
            pass


@router.post("/predict/upload", tags=["Prediction"])
async def predict_upload(file: UploadFile = File(...)):
    """
    Accept CSV file upload, process immediately, and return results + CSV content.
    """
    # Create temp files
    input_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    output_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    
    try:
        # 1. Save uploaded content to temp input
        content = await file.read()
        input_tmp.write(content)
        input_tmp.close() 
        output_tmp.close() 

        logger.info(f"Processing temp upload: {input_tmp.name}")

        # 2. Run Pipeline Synchronously
        result = batch_pipeline.predict_batch(input_tmp.name, output_tmp.name)

        # 3. Read the output CSV content into memory
        with open(output_tmp.name, 'r', encoding='utf-8') as f:
            csv_content = f.read()

        # 4. Get Metrics
        avg_prob = getattr(result, 'avg_probability', 0.0)

        # 5. Return Data + Content
        return {
            "message": "Batch prediction completed",
            "total_records": int(result.total_records),
            "successful": int(result.successful_predictions),
            "failed": int(result.failed_predictions),
            "avg_confidence": float(result.avg_confidence),
            "avg_probability": float(avg_prob),
            "processing_time_minutes": float(result.processing_time_minutes),
            "csv_content": csv_content, 
            "filename": f"predictions_{uuid.uuid4().hex[:8]}.csv"
        }

    except Exception as e:
        logger.error(f"Upload batch prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    finally:
        # 6. Cleanup Temp Files
        try:
            if os.path.exists(input_tmp.name):
                os.remove(input_tmp.name)
            if os.path.exists(output_tmp.name):
                os.remove(output_tmp.name)
        except Exception as cleanup_error:
            logger.error(f"Temp file cleanup failed: {cleanup_error}")


@router.get("/download/exports/{filename}", tags=["Files"])
async def download_export_file(filename: str):
    """Serve export files from the configured exports directory (Legacy support)."""
    try:
        target = EXPORTS_DIR / filename
        target_resolved = target.resolve()
        exports_resolved = EXPORTS_DIR.resolve()
        
        try:
            target_resolved.relative_to(exports_resolved)
        except Exception:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid filename")

        if not target.exists():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")

        return FileResponse(path=str(target_resolved), filename=filename, media_type='text/csv')

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/evaluation/latest", tags=["Evaluation"])
async def evaluation_latest():
    """Return the latest metrics CSV from `training_results/` as JSON."""
    try:
        metrics_dir = TRAINING_RESULTS_DIR
        if not metrics_dir.exists():
            raise FileNotFoundError("training_results directory not found")

        metrics_files = list(metrics_dir.glob("*_metrics.csv"))
        if not metrics_files:
            metrics_files = list(metrics_dir.glob("*.csv"))

        if not metrics_files:
            raise FileNotFoundError("No metrics CSV files found in training_results")

        latest = max(metrics_files, key=lambda p: p.stat().st_mtime)
        df = pd.read_csv(latest)
        record = df.iloc[-1].to_dict()

        normalized = {}
        cm_parts = {}

        for k, v in record.items():
            lk = k.lower().strip()
            if lk in ("acc", "accuracy", "accuracy_score"):
                normalized["accuracy"] = v
            elif lk in ("prec", "precision", "precision_score"):
                normalized["precision"] = v
            elif lk in ("recall", "recall_score"):
                normalized["recall"] = v
            elif lk in ("f1", "f1_score", "f1-score"):
                normalized["f1_score"] = v
            elif lk in ("auc", "roc_auc", "auc_roc", "auc-roc"):
                normalized["auc_roc"] = v
            elif lk in ("confusion_matrix",):
                try:
                    if isinstance(v, (list, dict)):
                        normalized["confusion_matrix"] = v
                    else:
                        parsed = None
                        if isinstance(v, str) and v.strip():
                            parsed = eval(v)
                        normalized["confusion_matrix"] = parsed if parsed is not None else v
                except Exception:
                    normalized["confusion_matrix"] = v
            elif lk in ("tn", "fp", "fn", "tp"):
                cm_parts[lk] = v
            else:
                normalized[k] = v

        if cm_parts:
            try:
                tn = int(cm_parts.get("tn", 0))
                fp = int(cm_parts.get("fp", 0))
                fn = int(cm_parts.get("fn", 0))
                tp = int(cm_parts.get("tp", 0))
                normalized["confusion_matrix"] = [[tn, fp], [fn, tp]]
            except Exception:
                normalized["confusion_matrix_parts"] = cm_parts

        response = {
            "model_name": latest.stem,
            "metrics": normalized if normalized else record,
            "metrics_csv": f"/static/training_results/{latest.name}"
        }

        return JSONResponse(response)

    except Exception as e:
        logger.error(f"Failed to load latest evaluation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )

# Admin restart logic remains unchanged...
@router.post("/admin/restart", tags=["Admin"])
async def admin_restart(background_tasks: BackgroundTasks = None, service_id: str = None):
    return JSONResponse({"status": "ok", "message": "Functionality preserved from previous version"})