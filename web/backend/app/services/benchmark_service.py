"""Business logic for benchmark operations."""

import json
import logging
from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Session

from ..models.benchmark import BenchmarkRun, TaskResult, ModelConfig, RunStatus
from ..schemas.benchmark import (
    BenchmarkRunCreate,
    ModelConfigCreate,
    LeaderboardEntry,
)
from ..utils.benchmark_adapter import get_benchmark_adapter

logger = logging.getLogger("openevals.web.service")


class BenchmarkService:
    """Service layer for benchmark operations."""

    def __init__(self, db: Session):
        self.db = db
        self.adapter = get_benchmark_adapter()

    # --- Model Config Operations ---

    def create_model_config(self, config: ModelConfigCreate) -> ModelConfig:
        """Create a new saved model configuration."""
        db_config = ModelConfig(
            name=config.name,
            model_type=config.model_type,
            config_json=json.dumps(config.config),
        )
        self.db.add(db_config)
        self.db.commit()
        self.db.refresh(db_config)
        return db_config

    def get_model_configs(self) -> list[ModelConfig]:
        """Get all saved model configurations."""
        return self.db.query(ModelConfig).all()

    def get_model_config(self, config_id: str) -> Optional[ModelConfig]:
        """Get a specific model configuration."""
        return self.db.query(ModelConfig).filter(ModelConfig.id == config_id).first()

    def delete_model_config(self, config_id: str) -> bool:
        """Delete a model configuration."""
        config = self.get_model_config(config_id)
        if config:
            self.db.delete(config)
            self.db.commit()
            return True
        return False

    # --- Benchmark Run Operations ---

    def create_benchmark_run(self, run_data: BenchmarkRunCreate) -> BenchmarkRun:
        """Create a new benchmark run."""
        config = {
            "name": run_data.name,
            "models": run_data.models,
            "tasks": run_data.tasks,
            "model_configs": run_data.model_configs,
            "task_configs": run_data.task_configs,
        }

        db_run = BenchmarkRun(
            name=run_data.name, status=RunStatus.PENDING, config_json=json.dumps(config)
        )
        self.db.add(db_run)
        self.db.commit()
        self.db.refresh(db_run)

        # Create placeholder task results
        for model in run_data.models:
            for task in run_data.tasks:
                task_result = TaskResult(
                    run_id=db_run.id, model_name=model, task_name=task, status="pending"
                )
                self.db.add(task_result)

        self.db.commit()
        return db_run

    def get_benchmark_run(self, run_id: str) -> Optional[BenchmarkRun]:
        """Get a specific benchmark run with results."""
        return self.db.query(BenchmarkRun).filter(BenchmarkRun.id == run_id).first()

    def get_benchmark_runs(
        self, status: Optional[str] = None, limit: int = 50, offset: int = 0
    ) -> list[BenchmarkRun]:
        """Get benchmark runs with optional filtering."""
        query = self.db.query(BenchmarkRun)

        if status:
            query = query.filter(BenchmarkRun.status == status)

        return (
            query.order_by(BenchmarkRun.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

    def update_run_status(
        self, run_id: str, status: RunStatus, error_message: Optional[str] = None
    ) -> Optional[BenchmarkRun]:
        """Update the status of a benchmark run."""
        run = self.get_benchmark_run(run_id)
        if run:
            run.status = status
            if status == RunStatus.RUNNING and not run.started_at:
                run.started_at = datetime.utcnow()
            elif status in [RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED]:
                run.completed_at = datetime.utcnow()
            if error_message:
                run.error_message = error_message
            self.db.commit()
            self.db.refresh(run)
        return run

    def update_task_result(
        self,
        run_id: str,
        model_name: str,
        task_name: str,
        status: str,
        overall: Optional[dict] = None,
        details: Optional[dict] = None,
        error_message: Optional[str] = None,
        duration: Optional[float] = None,
    ) -> Optional[TaskResult]:
        """Update a task result."""
        result = (
            self.db.query(TaskResult)
            .filter(
                TaskResult.run_id == run_id,
                TaskResult.model_name == model_name,
                TaskResult.task_name == task_name,
            )
            .first()
        )

        if result:
            result.status = status
            if overall:
                result.overall_json = json.dumps(overall)
            if details:
                result.details_json = json.dumps(details)
            if error_message:
                result.error_message = error_message
            if duration:
                result.duration_seconds = str(duration)
            self.db.commit()
            self.db.refresh(result)

        return result

    def delete_benchmark_run(self, run_id: str) -> bool:
        """Delete a benchmark run and its results."""
        run = self.get_benchmark_run(run_id)
        if run:
            self.db.delete(run)
            self.db.commit()
            return True
        return False

    # --- Leaderboard Operations ---

    def get_leaderboard(
        self, task_filter: Optional[str] = None
    ) -> list[LeaderboardEntry]:
        """Generate leaderboard from completed benchmark results."""
        # Get all completed runs
        runs = (
            self.db.query(BenchmarkRun)
            .filter(BenchmarkRun.status == RunStatus.COMPLETED)
            .all()
        )

        # Aggregate scores by model
        model_scores: dict[str, dict] = {}

        for run in runs:
            for result in run.task_results:
                if result.status != "completed" or not result.overall_json:
                    continue

                if task_filter and result.task_name != task_filter:
                    continue

                model = result.model_name
                if model not in model_scores:
                    model_scores[model] = {
                        "task_scores": {},
                        "task_counts": {},
                        "total_runs": 0,
                    }

                try:
                    overall = json.loads(result.overall_json)
                    # Try to extract accuracy or score
                    score = overall.get(
                        "accuracy", overall.get("score", overall.get("exact_match", 0))
                    )
                    if isinstance(score, (int, float)):
                        task = result.task_name
                        if task not in model_scores[model]["task_scores"]:
                            model_scores[model]["task_scores"][task] = 0
                            model_scores[model]["task_counts"][task] = 0
                        model_scores[model]["task_scores"][task] += score
                        model_scores[model]["task_counts"][task] += 1
                        model_scores[model]["total_runs"] += 1
                except Exception:
                    continue

        # Calculate averages and create entries
        entries = []
        for model, data in model_scores.items():
            task_avgs = {}
            for task, total_score in data["task_scores"].items():
                count = data["task_counts"][task]
                task_avgs[task] = total_score / count if count > 0 else 0

            avg_score = sum(task_avgs.values()) / len(task_avgs) if task_avgs else 0

            entries.append(
                LeaderboardEntry(
                    rank=0,  # Will be set after sorting
                    model_name=model,
                    average_score=avg_score,
                    task_scores=task_avgs,
                    total_runs=data["total_runs"],
                )
            )

        # Sort by average score and assign ranks
        entries.sort(key=lambda x: x.average_score, reverse=True)
        for i, entry in enumerate(entries):
            entry.rank = i + 1

        return entries

    # --- Statistics ---

    def get_stats(self) -> dict:
        """Get overall statistics."""
        total_runs = self.db.query(BenchmarkRun).count()
        completed_runs = (
            self.db.query(BenchmarkRun)
            .filter(BenchmarkRun.status == RunStatus.COMPLETED)
            .count()
        )
        running_runs = (
            self.db.query(BenchmarkRun)
            .filter(BenchmarkRun.status == RunStatus.RUNNING)
            .count()
        )
        total_models = self.db.query(ModelConfig).count()

        # Get unique models from results
        unique_models = self.db.query(TaskResult.model_name).distinct().count()

        return {
            "total_runs": total_runs,
            "completed_runs": completed_runs,
            "running_runs": running_runs,
            "saved_models": total_models,
            "unique_models_tested": unique_models,
            "available_tasks": len(self.adapter.get_available_tasks()),
        }
