"""
FastAPI Application
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from pathlib import Path
from contextlib import asynccontextmanager

from src.sentiment_analysis.api.routes import router
from src.sentiment_analysis.logging.logger import logging as logger

# Define lifespan context manager (Replaces on_event startup/shutdown)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager handles startup and shutdown logic.
    Code before yield runs on startup.
    Code after yield runs on shutdown.
    """
    # Startup logic
    logger.info("="*50)
    logger.info("SENTIMENT ANALYSIS API STARTED")
    logger.info("="*50)
    
    yield
    
    # Shutdown logic
    logger.info("="*50)
    logger.info("SENTIMENT ANALYSIS API STOPPED")
    logger.info("="*50)

# Create FastAPI app with lifespan
app = FastAPI(
    title="Sentiment Analysis API",
    description="Deep Learning-based Social Media Sentiment Analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
# Configure CORS origins from env (comma-separated) or fall back to wildcard for dev
allowed_origins = os.getenv("ALLOWED_ORIGINS")
if allowed_origins:
    origins = [o.strip() for o in allowed_origins.split(",") if o.strip()]
else:
    origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)

# Mount useful static folders for frontend consumption
# `training_results` is used by the dashboard (plots/metrics). `exports` holds batch outputs.
# Use env vars to configure directories (defaults to repo root paths)
repo_root = Path.cwd()
TRAINING_RESULTS_DIR = Path(os.getenv("TRAINING_RESULTS_DIR", str(repo_root / "training_results")))
EXPORTS_DIR = Path(os.getenv("EXPORTS_DIR", str(repo_root / "exports")))

# Ensure directories exist so StaticFiles can mount if needed
TRAINING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static/training_results", StaticFiles(directory=str(TRAINING_RESULTS_DIR)), name="training_results")
app.mount("/static/exports", StaticFiles(directory=str(EXPORTS_DIR)), name="exports")

logger.info("FastAPI application initialized")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)