"""Privacy Shield Backend API."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.config import get_settings
from app.utils.reproducibility import set_seed
from app.utils.logging import get_logger

# Setup logging
logger = get_logger(__name__)

# Initialize settings
settings = get_settings()
settings.ensure_paths()

# Set seed for reproducibility
set_seed(settings.ml_random_seed)

# Create FastAPI app
app = FastAPI(
    title="Privacy Shield",
    description="Secure & Private Fraud Detection Against Model Inversion Attacks",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("Privacy Shield Backend Started")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Frontend URL: {settings.frontend_url}")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("Privacy Shield Backend Shutting Down")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "application": "Privacy Shield",
        "status": "running",
        "version": "1.0.0",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


# Import API routes
from app.api import dataset, baseline, privacy, mia, experiments, evaluation

app.include_router(dataset.router, prefix="/api/dataset", tags=["dataset"])
app.include_router(baseline.router, prefix="/api/baseline", tags=["baseline"])
app.include_router(privacy.router, prefix="/api/privacy", tags=["privacy"])
app.include_router(mia.router, prefix="/api/mia", tags=["mia"])
app.include_router(experiments.router, prefix="/api/experiments", tags=["experiments"])
app.include_router(evaluation.router, prefix="/api/evaluation", tags=["evaluation"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=settings.reload,
    )
