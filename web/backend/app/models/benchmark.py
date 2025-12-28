"""Database models for benchmark runs and results."""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship
import enum

from .database import Base


class RunStatus(str, enum.Enum):
    """Status of a benchmark run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BenchmarkRun(Base):
    """A single benchmark run configuration and status."""

    __tablename__ = "benchmark_runs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    status = Column(SQLEnum(RunStatus), default=RunStatus.PENDING)
    config_json = Column(Text, nullable=False)  # JSON string of full config
    error_message = Column(Text, nullable=True)

    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    task_results = relationship(
        "TaskResult", back_populates="benchmark_run", cascade="all, delete-orphan"
    )


class TaskResult(Base):
    """Results for a single model-task evaluation."""

    __tablename__ = "task_results"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id = Column(String(36), ForeignKey("benchmark_runs.id"), nullable=False)
    model_name = Column(String(255), nullable=False)
    task_name = Column(String(255), nullable=False)
    status = Column(
        String(50), default="pending"
    )  # pending, running, completed, failed

    overall_json = Column(Text, nullable=True)  # JSON string of overall metrics
    details_json = Column(Text, nullable=True)  # JSON string of detailed results
    error_message = Column(Text, nullable=True)

    duration_seconds = Column(String(20), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    benchmark_run = relationship("BenchmarkRun", back_populates="task_results")


class ModelConfig(Base):
    """Saved model configurations for reuse."""

    __tablename__ = "model_configs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), unique=True, nullable=False)
    model_type = Column(
        String(50), nullable=False
    )  # gemma, mistral, llama, huggingface
    config_json = Column(Text, nullable=False)  # JSON string of model config

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
