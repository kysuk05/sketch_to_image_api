from fastapi import APIRouter
import torch

router = APIRouter()
pipe = None

@router.get("/check")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": pipe is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }