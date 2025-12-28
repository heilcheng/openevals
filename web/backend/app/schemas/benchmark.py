"""Pydantic schemas for benchmark-related API operations."""

from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, Field


# --- Model Configuration Schemas ---


class ModelConfigBase(BaseModel):
    """Base schema for model configuration."""

    name: str = Field(..., min_length=1, max_length=255)
    model_type: str = Field(
        ..., description="Model type: gemma, mistral, llama, huggingface"
    )
    config: dict[str, Any] = Field(default_factory=dict)


class ModelConfigCreate(ModelConfigBase):
    """Schema for creating a new model configuration."""

    pass


class ModelConfigResponse(ModelConfigBase):
    """Schema for model configuration response."""

    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# --- Task Schemas ---


class TaskInfo(BaseModel):
    """Information about an available task type."""

    name: str
    type: str
    description: str
    default_config: dict[str, Any] = Field(default_factory=dict)


# --- Benchmark Run Schemas ---


class BenchmarkRunCreate(BaseModel):
    """Schema for creating a new benchmark run."""

    name: str = Field(..., min_length=1, max_length=255)
    models: list[str] = Field(
        ..., min_length=1, description="List of model names to benchmark"
    )
    tasks: list[str] = Field(..., min_length=1, description="List of task names to run")
    model_configs: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Optional custom configs per model"
    )
    task_configs: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Optional custom configs per task"
    )


class TaskResultResponse(BaseModel):
    """Schema for a single task result."""

    id: str
    model_name: str
    task_name: str
    status: str
    overall: Optional[dict[str, Any]] = None
    details: Optional[dict[str, Any]] = None
    error_message: Optional[str] = None
    duration_seconds: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class BenchmarkRunResponse(BaseModel):
    """Schema for benchmark run response."""

    id: str
    name: str
    status: str
    config: dict[str, Any]
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    task_results: list[TaskResultResponse] = Field(default_factory=list)

    class Config:
        from_attributes = True


class BenchmarkRunList(BaseModel):
    """Schema for listing benchmark runs."""

    id: str
    name: str
    status: str
    models_count: int
    tasks_count: int
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime


# --- Leaderboard Schemas ---


class LeaderboardEntry(BaseModel):
    """A single entry in the leaderboard."""

    rank: int
    model_name: str
    average_score: float
    task_scores: dict[str, float]
    total_runs: int


# --- WebSocket Progress Schemas ---


class ProgressUpdate(BaseModel):
    """Real-time progress update for a benchmark run."""

    run_id: str
    status: str
    current_model: Optional[str] = None
    current_task: Optional[str] = None
    progress_percent: float = 0.0
    completed_tasks: int = 0
    total_tasks: int = 0
    message: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
