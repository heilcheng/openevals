"""Benchmark run API endpoints."""

import json
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from ...models import get_db
from ...models.benchmark import BenchmarkRun, RunStatus
from ...schemas.benchmark import (
    BenchmarkRunCreate,
    BenchmarkRunResponse,
    BenchmarkRunList,
    TaskResultResponse,
    LeaderboardEntry,
)
from ...services.benchmark_service import BenchmarkService
from ...utils.benchmark_adapter import get_benchmark_adapter

router = APIRouter()


def get_service(db: Session = Depends(get_db)) -> BenchmarkService:
    return BenchmarkService(db)


def serialize_run(run: BenchmarkRun) -> BenchmarkRunResponse:
    """Convert database model to response schema."""
    task_results = []
    for tr in run.task_results:
        task_results.append(
            TaskResultResponse(
                id=tr.id,
                model_name=tr.model_name,
                task_name=tr.task_name,
                status=tr.status,
                overall=json.loads(tr.overall_json) if tr.overall_json else None,
                details=json.loads(tr.details_json) if tr.details_json else None,
                error_message=tr.error_message,
                duration_seconds=tr.duration_seconds,
                created_at=tr.created_at,
            )
        )

    return BenchmarkRunResponse(
        id=run.id,
        name=run.name,
        status=run.status.value if isinstance(run.status, RunStatus) else run.status,
        config=json.loads(run.config_json),
        error_message=run.error_message,
        started_at=run.started_at,
        completed_at=run.completed_at,
        created_at=run.created_at,
        updated_at=run.updated_at,
        task_results=task_results,
    )


async def execute_benchmark(run_id: str, db_session_maker):
    """Background task to execute a benchmark run."""
    from ...models.database import SessionLocal

    db = SessionLocal()
    try:
        service = BenchmarkService(db)
        adapter = get_benchmark_adapter()

        # Get the run
        run = service.get_benchmark_run(run_id)
        if not run:
            return

        config = json.loads(run.config_json)

        # Update status to running
        service.update_run_status(run_id, RunStatus.RUNNING)

        # Create config file
        config_path = adapter.create_config_file(
            models=config["models"],
            tasks=config["tasks"],
            model_configs=config.get("model_configs", {}),
            task_configs=config.get("task_configs", {}),
        )

        def progress_callback(update: dict):
            """Handle progress updates."""
            # Update task results if we have a current model/task
            if update.get("current_model") and update.get("current_task"):
                service.update_task_result(
                    run_id=run_id,
                    model_name=update["current_model"],
                    task_name=update["current_task"],
                    status="running",
                )

        try:
            results = await adapter.run_benchmark(
                run_id=run_id,
                config_path=config_path,
                progress_callback=progress_callback,
            )

            # Save results to database
            for model_name, model_results in results.items():
                for task_name, task_result in model_results.items():
                    if "error" in task_result:
                        service.update_task_result(
                            run_id=run_id,
                            model_name=model_name,
                            task_name=task_name,
                            status="failed",
                            error_message=task_result["error"],
                        )
                    else:
                        # Extract overall metrics
                        overall = {}
                        if "overall" in task_result:
                            overall = task_result["overall"]
                        elif "accuracy" in task_result:
                            overall = {"accuracy": task_result["accuracy"]}

                        service.update_task_result(
                            run_id=run_id,
                            model_name=model_name,
                            task_name=task_name,
                            status="completed",
                            overall=overall,
                            details=task_result,
                        )

            service.update_run_status(run_id, RunStatus.COMPLETED)

        except Exception as e:
            service.update_run_status(run_id, RunStatus.FAILED, str(e))

    finally:
        db.close()


@router.post("", response_model=BenchmarkRunResponse)
async def create_benchmark(
    run_data: BenchmarkRunCreate,
    background_tasks: BackgroundTasks,
    service: BenchmarkService = Depends(get_service),
):
    """Create and start a new benchmark run."""
    run = service.create_benchmark_run(run_data)

    # Start benchmark in background
    from ...models.database import SessionLocal

    background_tasks.add_task(execute_benchmark, run.id, SessionLocal)

    return serialize_run(run)


@router.get("", response_model=list[BenchmarkRunList])
async def list_benchmarks(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    service: BenchmarkService = Depends(get_service),
):
    """List all benchmark runs."""
    runs = service.get_benchmark_runs(status=status, limit=limit, offset=offset)

    result = []
    for run in runs:
        config = json.loads(run.config_json)
        result.append(
            BenchmarkRunList(
                id=run.id,
                name=run.name,
                status=(
                    run.status.value
                    if isinstance(run.status, RunStatus)
                    else run.status
                ),
                models_count=len(config.get("models", [])),
                tasks_count=len(config.get("tasks", [])),
                started_at=run.started_at,
                completed_at=run.completed_at,
                created_at=run.created_at,
            )
        )

    return result


@router.get("/stats")
async def get_stats(service: BenchmarkService = Depends(get_service)):
    """Get overall benchmark statistics."""
    return service.get_stats()


@router.get("/leaderboard", response_model=list[LeaderboardEntry])
async def get_leaderboard(
    task: Optional[str] = None, service: BenchmarkService = Depends(get_service)
):
    """Get the model leaderboard."""
    return service.get_leaderboard(task_filter=task)


@router.get("/{run_id}", response_model=BenchmarkRunResponse)
async def get_benchmark(run_id: str, service: BenchmarkService = Depends(get_service)):
    """Get a specific benchmark run with results."""
    run = service.get_benchmark_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Benchmark run not found")
    return serialize_run(run)


@router.post("/{run_id}/cancel")
async def cancel_benchmark(
    run_id: str, service: BenchmarkService = Depends(get_service)
):
    """Cancel a running benchmark."""
    run = service.get_benchmark_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Benchmark run not found")

    if run.status != RunStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Benchmark is not running")

    adapter = get_benchmark_adapter()
    if adapter.cancel_run(run_id):
        service.update_run_status(run_id, RunStatus.CANCELLED)
        return {"status": "cancelled"}

    raise HTTPException(status_code=400, detail="Could not cancel benchmark")


@router.delete("/{run_id}")
async def delete_benchmark(
    run_id: str, service: BenchmarkService = Depends(get_service)
):
    """Delete a benchmark run."""
    if service.delete_benchmark_run(run_id):
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Benchmark run not found")
