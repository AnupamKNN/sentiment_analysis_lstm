"""
FastAPI Routes
"""

from fastapi import APIRouter, HTTPException, status
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
        
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            version="1.0.0"
        )


@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_sentiment(request: PredictionRequest):
    """
    Predict sentiment for single text
    
    - **text**: Text to analyze (3-1000 characters)
    
    Returns sentiment prediction with confidence score
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
    Predict sentiment for batch of texts from CSV file
    
    - **input_file**: Path to input CSV with 'text' column
    - **output_file**: Optional path for output CSV
    
    Returns batch prediction summary
    """
    try:
        logger.info(f"Batch prediction request: {request.input_file}")
        
        # Predict
        result = batch_pipeline.predict_batch(
            request.input_file,
            request.output_file
        )
        
        return {
            "message": "Batch prediction completed",
            "input_file": result.input_file_path,
            "output_file": result.output_file_path,
            "total_records": result.total_records,
            "successful": result.successful_predictions,
            "failed": result.failed_predictions,
            "avg_confidence": result.avg_confidence,
            "processing_time_minutes": result.processing_time_minutes
        }
    
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
