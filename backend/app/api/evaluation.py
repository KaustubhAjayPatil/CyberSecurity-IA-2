"""Evaluation API endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/comparison")
async def get_comparison():
    """Get privacy-utility comparison - placeholder."""
    return {"message": "Evaluation API - see example scripts for full implementation"}
