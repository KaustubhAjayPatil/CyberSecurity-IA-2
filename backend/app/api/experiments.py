"""Experiments API endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/list")
async def list_experiments():
    """List experiments - placeholder."""
    return {"message": "Experiments API - see example scripts for full implementation"}
