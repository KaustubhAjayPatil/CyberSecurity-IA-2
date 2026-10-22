"""Baseline model API endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

router = APIRouter()


class TrainingRequest(BaseModel):
    """Training request."""

    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    hidden_dim: int = 256
    num_hidden_layers: int = 3
    dropout: float = 0.3


@router.post("/train")
async def train_baseline(request: TrainingRequest):
    """Train baseline model - placeholder."""
    return {"message": "Training endpoint - see example scripts for full implementation"}


@router.get("/metrics")
async def get_baseline_metrics():
    """Get baseline model metrics - placeholder."""
    return {"message": "Metrics endpoint - see example scripts for full implementation"}
