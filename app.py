"""
FastAPI Application
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.sentiment_analysis.api.routes import router
from src.sentiment_analysis.logging.logger import logging as logger

# Create FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="Deep Learning-based Social Media Sentiment Analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)

logger.info("FastAPI application initialized")


@app.on_event("startup")
async def startup_event():
    """Startup event"""
    logger.info("="*50)
    logger.info("SENTIMENT ANALYSIS API STARTED")
    logger.info("="*50)


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event"""
    logger.info("="*50)
    logger.info("SENTIMENT ANALYSIS API STOPPED")
    logger.info("="*50)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)