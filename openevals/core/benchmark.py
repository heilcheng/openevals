"""
Core orchestration logic for the OpenEvalsing Suite.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import yaml

# Import the factory and manager to be used directly
from openevals.core.interfaces import BenchmarkFactory
from openevals.core.model_loader import get_model_manager

# Import the configuration validation tools
from openevals.utils.config_validation import ConfigurationError, validate_config_file


# Define a custom exception for evaluation errors
class EvaluationError(Exception):
    """Custom exception for errors during benchmark evaluation."""

    pass


class Benchmark:
    """
    Main class for orchestrating benchmark runs.

    This class handles loading configuration, models, and tasks,
    and then runs the evaluations and stores the results.
    """

    def __init__(self, config_path: str):
        """
        Initialize the benchmark with a configuration file.

        Args:
            config_path: Path to the YAML configuration file.
        """
        self.logger = logging.getLogger("openevals.core")
        self.config_path = config_path
        self.config = self._load_config()

        self.models: Dict[str, Any] = {}
        self.tasks: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}

    def _load_config(self) -> Dict[str, Any]:
        """
        Load and validate the benchmark configuration from a YAML file.

        Returns:
            A dictionary containing the validated configuration.
        """
        self.logger.info(
            f"Loading and validating configuration from: {self.config_path}"
        )
        try:
            # Use the validation utility to load and check the config
            validated_config = validate_config_file(self.config_path)
            # Convert the Pydantic model to a dict for compatibility with the class
            config_data = validated_config.model_dump()
            self.logger.info("Configuration loaded and validated successfully.")
            return config_data
        except ConfigurationError as e:
            self.logger.error(
                f"Configuration validation failed for {self.config_path}: {e}"
            )
            raise
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred while loading configuration: {e}"
            )
            raise

    def load_models(self, model_names: Optional[List[str]] = None):
        """
        Load models as specified in the configuration using the ModelManager.

        Args:
            model_names: An optional list of specific models to load.
                         If None, all models from the config will be loaded.
        """
        self.logger.info("Loading models...")
        model_manager = get_model_manager()
        model_configs = self.config.get("models", {})

        # If model_names is provided, use it to filter. Otherwise, load all.
        models_to_load = model_names if model_names else list(model_configs.keys())

        for model_name in models_to_load:
            if model_name not in model_configs:
                self.logger.warning(
                    f"Model '{model_name}' requested but not found in configuration. Skipping."
                )
                continue

            config = model_configs[model_name]
            try:
                # Use the ModelManager to load the model
                self.models[model_name] = model_manager.load_model(model_name, config)
                self.logger.info(f"Successfully loaded model '{model_name}'")
            except Exception as e:
                self.logger.error(
                    f"Failed to load model '{model_name}': {e}", exc_info=True
                )

    def load_tasks(self, task_names: Optional[List[str]] = None):
        """
        Load tasks as specified in the configuration using the BenchmarkFactory.

        Args:
            task_names: An optional list of specific tasks to load.
                        If None, all tasks from the config will be loaded.
        """
        self.logger.info("Loading tasks...")
        task_configs = self.config.get("tasks", {})

        # If task_names is provided, use it to filter. Otherwise, load all.
        tasks_to_load = task_names if task_names else list(task_configs.keys())

        for task_name in tasks_to_load:
            if task_name not in task_configs:
                self.logger.warning(
                    f"Task '{task_name}' requested but not found in configuration. Skipping."
                )
                continue

            config = task_configs[task_name]
            task_type = config.get("type")
            if not task_type:
                self.logger.error(
                    f"Task '{task_name}' in config is missing a 'type'. Skipping."
                )
                continue

            try:
                # Use the factory to create a benchmark instance
                self.tasks[task_name] = BenchmarkFactory.create_benchmark(
                    task_type, config
                )
                self.logger.info(
                    f"Successfully loaded task '{task_name}' of type '{task_type}'"
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to load task '{task_name}': {e}", exc_info=True
                )

    def run_benchmarks(self) -> Dict[str, Any]:
        """
        Run all loaded benchmarks for all loaded models.

        Returns:
            A dictionary containing the results.

        Raises:
            EvaluationError: If no models or tasks are available for evaluation.
        """
        self.logger.info("Starting all benchmark runs...")

        if not self.models or not self.tasks:
            raise EvaluationError("No models or tasks loaded. Nothing to run.")

        for model_name, model in self.models.items():
            self.logger.info(f"--- Running benchmarks for model: {model_name} ---")
            if model_name not in self.results:
                self.results[model_name] = {}

            for task_name, task in self.tasks.items():
                self.logger.info(f"  - Task: {task_name}")
                try:
                    task_result = task.evaluate(model)
                    self.results[model_name][task_name] = task_result
                except Exception as e:
                    self.logger.error(
                        f"    ERROR running task {task_name} for model {model_name}: {e}"
                    )
                    self.results[model_name][task_name] = {"error": str(e)}

        self.logger.info("All benchmark runs complete.")
        return self.results

    def save_results(self, output_path: str) -> str:
        """
        Save the benchmark results to a YAML file.

        Args:
            output_path: The path to the output YAML file.

        Returns:
            The absolute path to the saved file.
        """
        self.logger.info(f"Saving results to {output_path}...")
        try:
            # Ensure the output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # Write the results dictionary to the specified YAML file
            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(self.results, f, default_flow_style=False, sort_keys=False)

            abs_path = os.path.abspath(output_path)
            self.logger.info(f"Successfully saved results to {abs_path}")
            return abs_path
        except Exception as e:
            self.logger.error(
                f"Failed to save results to {output_path}: {e}", exc_info=True
            )
            raise
