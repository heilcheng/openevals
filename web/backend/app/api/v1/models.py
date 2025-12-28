"""Model configuration API endpoints."""

import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ...models import get_db
from ...models.benchmark import ModelConfig
from ...schemas.benchmark import ModelConfigCreate, ModelConfigResponse
from ...services.benchmark_service import BenchmarkService
from ...utils.benchmark_adapter import DEFAULT_MODEL_CONFIGS

router = APIRouter()


def get_service(db: Session = Depends(get_db)) -> BenchmarkService:
    return BenchmarkService(db)


def serialize_config(config: ModelConfig) -> ModelConfigResponse:
    """Convert database model to response schema."""
    return ModelConfigResponse(
        id=config.id,
        name=config.name,
        model_type=config.model_type,
        config=json.loads(config.config_json),
        created_at=config.created_at,
        updated_at=config.updated_at,
    )


@router.get("/types")
async def get_model_types():
    """Get available model types and their default configurations."""
    return {
        "types": list(DEFAULT_MODEL_CONFIGS.keys()),
        "defaults": DEFAULT_MODEL_CONFIGS,
    }


@router.post("", response_model=ModelConfigResponse)
async def create_model_config(
    config: ModelConfigCreate, service: BenchmarkService = Depends(get_service)
):
    """Create a new saved model configuration."""
    try:
        db_config = service.create_model_config(config)
        return serialize_config(db_config)
    except Exception as e:
        if "UNIQUE constraint" in str(e):
            raise HTTPException(status_code=400, detail="Model name already exists")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=list[ModelConfigResponse])
async def list_model_configs(service: BenchmarkService = Depends(get_service)):
    """List all saved model configurations."""
    configs = service.get_model_configs()
    return [serialize_config(c) for c in configs]


@router.get("/{config_id}", response_model=ModelConfigResponse)
async def get_model_config(
    config_id: str, service: BenchmarkService = Depends(get_service)
):
    """Get a specific model configuration."""
    config = service.get_model_config(config_id)
    if not config:
        raise HTTPException(status_code=404, detail="Model configuration not found")
    return serialize_config(config)


@router.delete("/{config_id}")
async def delete_model_config(
    config_id: str, service: BenchmarkService = Depends(get_service)
):
    """Delete a model configuration."""
    if service.delete_model_config(config_id):
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Model configuration not found")
