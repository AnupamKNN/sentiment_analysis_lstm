"""
API Request/Response Schemas
"""

from pydantic import BaseModel, Field, validator
from typing import Optional


class PredictionRequest(BaseModel):
    """Request schema for single prediction"""
    text: str = Field(..., min_length=3, max_length=1000, description="Text to analyze")
    
    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "text": "I love this product! It's amazing and works perfectly."
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for prediction"""
    text: str
    cleaned_text: str
    sentiment: str
    label: int
    confidence: float
    probability: float
    processing_time_ms: float
    
    class Config:
        schema_extra = {
            "example": {
                "text": "I love this product!",
                "cleaned_text": "i love this product",
                "sentiment": "Positive",
                "label": 1,
                "confidence": 0.95,
                "probability": 0.95,
                "processing_time_ms": 45.2
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request schema for batch prediction"""
    input_file: str = Field(..., description="Path to input CSV file")
    output_file: Optional[str] = Field(None, description="Path to output CSV file")
    
    class Config:
        schema_extra = {
            "example": {
                "input_file": "data/batch_input.csv",
                "output_file": "data/batch_output.csv"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    version: str
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "version": "1.0.0"
            }
        }
