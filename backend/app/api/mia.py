"""Model Inversion Attack API endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.post("/run")
async def run_mia():
    """Run MIA - placeholder."""
    return {"message": "MIA API - see example scripts for full implementation"}
