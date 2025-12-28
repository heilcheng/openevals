"""
Bridge between the web API and the openevals core library.

This adapter wraps the GemmaBenchmark class to provide:
- Async execution with progress callbacks
- WebSocket-compatible progress updates
- Integration with the database models
"""

import asyncio
import logging
import tempfile
import os
import sys
from typing import Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path to import openevals
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
    ),
)

from openevals.core.benchmark import GemmaBenchmark  # noqa: E402

logger = logging.getLogger("openevals.web.adapter")


# Available task types with descriptions
AVAILABLE_TASKS = {
    "gsm8k": {
        "type": "gsm8k",
        "description": "Grade School Math 8K - Mathematical reasoning benchmark",
        "default_config": {
            "subset": "main",
            "split": "test",
            "num_samples": 100,
            "num_shots": 5,
        },
    },
    "truthfulqa": {
        "type": "truthfulqa",
        "description": "TruthfulQA - Truthfulness and factuality benchmark",
        "default_config": {
            "subset": "multiple_choice",
            "split": "validation",
            "num_samples": 100,
        },
    },
    "mmlu": {
        "type": "mmlu",
        "description": "MMLU - Massive Multitask Language Understanding",
        "default_config": {
            "subset": "all",
            "split": "test",
            "num_samples": 100,
            "num_shots": 5,
        },
    },
    "hellaswag": {
        "type": "hellaswag",
        "description": "HellaSwag - Commonsense reasoning benchmark",
        "default_config": {"split": "validation", "num_samples": 100},
    },
}

# Default model configurations
DEFAULT_MODEL_CONFIGS = {
    "gemma": {
        "type": "gemma",
        "size": "2b",
        "variant": "it",
        "torch_dtype": "bfloat16",
        "device_map": "auto",
    },
    "mistral": {
        "type": "mistral",
        "size": "7B",
        "torch_dtype": "bfloat16",
        "device_map": "auto",
    },
    "llama": {
        "type": "llama",
        "size": "8B",
        "version": "3.1",
        "torch_dtype": "bfloat16",
        "device_map": "auto",
    },
}


class BenchmarkAdapter:
    """
    Async adapter for running benchmarks with progress tracking.

    This class bridges the synchronous openevals library with
    the async FastAPI backend, providing real-time progress updates.
    """

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._active_runs: dict[str, bool] = {}  # run_id -> is_cancelled

    def get_available_tasks(self) -> dict[str, dict]:
        """Return available task types and their configurations."""
        return AVAILABLE_TASKS

    def get_default_model_config(self, model_type: str) -> dict[str, Any]:
        """Get default configuration for a model type."""
        return DEFAULT_MODEL_CONFIGS.get(model_type, {})

    def create_config_file(
        self,
        models: list[str],
        tasks: list[str],
        model_configs: dict[str, dict],
        task_configs: dict[str, dict],
    ) -> str:
        """
        Create a temporary YAML config file for the benchmark.

        Returns the path to the created config file.
        """
        import yaml

        # Build models section
        models_section = {}
        for model_name in models:
            if model_name in model_configs:
                models_section[model_name] = model_configs[model_name]
            else:
                # Try to infer from name
                for default_type, default_config in DEFAULT_MODEL_CONFIGS.items():
                    if default_type in model_name.lower():
                        models_section[model_name] = default_config.copy()
                        break
                else:
                    # Generic huggingface model
                    models_section[model_name] = {
                        "type": "huggingface",
                        "model_id": model_name,
                        "torch_dtype": "bfloat16",
                        "device_map": "auto",
                    }

        # Build tasks section
        tasks_section = {}
        for task_name in tasks:
            task_info = AVAILABLE_TASKS.get(task_name, {})
            base_config = {
                "type": task_info.get("type", task_name),
                **task_info.get("default_config", {}),
            }
            if task_name in task_configs:
                base_config.update(task_configs[task_name])
            tasks_section[task_name] = base_config

        config = {"models": models_section, "tasks": tasks_section}

        # Write to temp file
        fd, path = tempfile.mkstemp(suffix=".yaml", prefix="benchmark_config_")
        with os.fdopen(fd, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        return path

    async def run_benchmark(
        self,
        run_id: str,
        config_path: str,
        progress_callback: Optional[Callable[[dict], None]] = None,
    ) -> dict[str, Any]:
        """
        Run a benchmark asynchronously with progress updates.

        Args:
            run_id: Unique identifier for this run
            config_path: Path to the YAML config file
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary containing benchmark results
        """
        self._active_runs[run_id] = False

        def run_sync():
            results = {}
            try:
                benchmark = GemmaBenchmark(config_path)

                # Get total work items
                model_names = list(benchmark.config.get("models", {}).keys())
                task_names = list(benchmark.config.get("tasks", {}).keys())
                total_tasks = len(model_names) * len(task_names)
                completed = 0

                # Send initial progress
                if progress_callback:
                    progress_callback(
                        {
                            "run_id": run_id,
                            "status": "running",
                            "progress_percent": 0,
                            "completed_tasks": 0,
                            "total_tasks": total_tasks,
                            "message": "Loading models...",
                        }
                    )

                # Load models with progress
                for i, model_name in enumerate(model_names):
                    if self._active_runs.get(run_id):
                        raise Exception("Benchmark cancelled")

                    if progress_callback:
                        progress_callback(
                            {
                                "run_id": run_id,
                                "status": "running",
                                "current_model": model_name,
                                "progress_percent": (i / len(model_names)) * 10,
                                "message": f"Loading model: {model_name}",
                            }
                        )

                    benchmark.load_models([model_name])

                # Load tasks
                if progress_callback:
                    progress_callback(
                        {
                            "run_id": run_id,
                            "status": "running",
                            "progress_percent": 10,
                            "message": "Loading tasks...",
                        }
                    )

                benchmark.load_tasks()

                # Run evaluations with progress
                for model_name in model_names:
                    if model_name not in benchmark.models:
                        continue

                    model = benchmark.models[model_name]
                    results[model_name] = {}

                    for task_name in task_names:
                        if self._active_runs.get(run_id):
                            raise Exception("Benchmark cancelled")

                        if task_name not in benchmark.tasks:
                            continue

                        task = benchmark.tasks[task_name]

                        if progress_callback:
                            progress_callback(
                                {
                                    "run_id": run_id,
                                    "status": "running",
                                    "current_model": model_name,
                                    "current_task": task_name,
                                    "progress_percent": 10
                                    + (completed / total_tasks) * 90,
                                    "completed_tasks": completed,
                                    "total_tasks": total_tasks,
                                    "message": f"Running {model_name} / {task_name}",
                                }
                            )

                        try:
                            result = task.evaluate(model)
                            results[model_name][task_name] = result
                        except Exception as e:
                            logger.error(
                                f"Error evaluating {model_name} on {task_name}: {e}"
                            )
                            results[model_name][task_name] = {"error": str(e)}

                        completed += 1

                # Final progress
                if progress_callback:
                    progress_callback(
                        {
                            "run_id": run_id,
                            "status": "completed",
                            "progress_percent": 100,
                            "completed_tasks": total_tasks,
                            "total_tasks": total_tasks,
                            "message": "Benchmark completed!",
                        }
                    )

                return results

            except Exception as e:
                if progress_callback:
                    progress_callback(
                        {"run_id": run_id, "status": "failed", "message": str(e)}
                    )
                raise
            finally:
                # Cleanup temp config
                try:
                    os.unlink(config_path)
                except Exception:
                    pass
                self._active_runs.pop(run_id, None)

        # Run in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, run_sync)

    def cancel_run(self, run_id: str) -> bool:
        """Cancel an active benchmark run."""
        if run_id in self._active_runs:
            self._active_runs[run_id] = True
            return True
        return False


# Singleton instance
_adapter_instance: Optional[BenchmarkAdapter] = None


def get_benchmark_adapter() -> BenchmarkAdapter:
    """Get the singleton BenchmarkAdapter instance."""
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = BenchmarkAdapter()
    return _adapter_instance
