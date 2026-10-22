"""Dataset API endpoints."""

from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
import json
from pathlib import Path

from app.config import get_settings
from app.data.loader import DatasetLoader
from app.data.profiler import DatasetProfiler
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

settings = get_settings()


# Global dataset storage
current_dataset = None
dataset_profiler = None


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload CSV dataset."""
    global current_dataset, dataset_profiler

    try:
        # Save uploaded file
        file_path = Path(settings.storage_path) / file.filename
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Load and validate
        loader = DatasetLoader(str(file_path), sample_size=settings.dataset_sample_size)
        current_dataset = loader.load()

        validation = loader.validate(current_dataset)

        logger.info(f"Dataset uploaded: {current_dataset.shape}")

        return {
            "message": "Dataset uploaded successfully",
            "shape": current_dataset.shape,
            "validation": validation,
        }

    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/profile")
async def profile_dataset():
    """Get dataset profile."""
    global current_dataset, dataset_profiler

    if current_dataset is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")

    try:
        dataset_profiler = DatasetProfiler(current_dataset)
        profile = dataset_profiler.profile_dataset()

        return profile

    except Exception as e:
        logger.error(f"Profiling error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/info")
async def dataset_info():
    """Get basic dataset information."""
    global current_dataset

    if current_dataset is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")

    return {
        "shape": current_dataset.shape,
        "columns": current_dataset.columns.tolist(),
        "dtypes": current_dataset.dtypes.astype(str).to_dict(),
        "missing_values": current_dataset.isnull().sum().to_dict(),
    }


@router.get("/status")
async def dataset_status():
    """Get dataset status."""
    global current_dataset

    if current_dataset is None:
        return {"loaded": False, "shape": None}

    return {
        "loaded": True,
        "shape": current_dataset.shape,
        "columns": len(current_dataset.columns),
        "rows": len(current_dataset),
    }
