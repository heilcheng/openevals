"""Task type API endpoints."""

from fastapi import APIRouter

from ...schemas.benchmark import TaskInfo
from ...utils.benchmark_adapter import get_benchmark_adapter

router = APIRouter()


@router.get("", response_model=list[TaskInfo])
async def list_tasks():
    """List all available benchmark tasks."""
    adapter = get_benchmark_adapter()
    tasks = adapter.get_available_tasks()

    return [
        TaskInfo(
            name=name,
            type=info["type"],
            description=info["description"],
            default_config=info["default_config"],
        )
        for name, info in tasks.items()
    ]


@router.get("/{task_name}", response_model=TaskInfo)
async def get_task(task_name: str):
    """Get details for a specific task."""
    adapter = get_benchmark_adapter()
    tasks = adapter.get_available_tasks()

    if task_name not in tasks:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Task not found")

    info = tasks[task_name]
    return TaskInfo(
        name=task_name,
        type=info["type"],
        description=info["description"],
        default_config=info["default_config"],
    )
