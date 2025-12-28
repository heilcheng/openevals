"""API v1 endpoints."""

from fastapi import APIRouter

from .benchmarks import router as benchmarks_router
from .models import router as models_router
from .tasks import router as tasks_router
from .websocket import router as websocket_router

api_router = APIRouter()
api_router.include_router(benchmarks_router, prefix="/benchmarks", tags=["benchmarks"])
api_router.include_router(models_router, prefix="/models", tags=["models"])
api_router.include_router(tasks_router, prefix="/tasks", tags=["tasks"])
api_router.include_router(websocket_router, tags=["websocket"])
