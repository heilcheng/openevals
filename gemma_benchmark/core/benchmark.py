"""
Core orchestration logic for the Gemma Benchmarking Suite.
"""

import logging
import yaml
from typing import Dict, Any

# Import the configuration validation tools
from gemma_benchmark.utils.config_validation import validate_config_file, ConfigurationError

# Define a custom exception for evaluation errors
class EvaluationError(Exception):
    """Custom exception for errors during benchmark evaluation."""
    pass

class GemmaBenchmark:
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
        self.logger = logging.getLogger("gemma_benchmark.core")
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
        self.logger.info(f"Loading and validating configuration from: {self.config_path}")
        try:
            # Use the validation utility to load and check the config
            validated_config = validate_config_file(self.config_path)
            # Convert the Pydantic model to a dict for compatibility with the class
            config_data = validated_config.model_dump()
            self.logger.info("Configuration loaded and validated successfully.")
            return config_data
        except ConfigurationError as e:
            self.logger.error(f"Configuration validation failed for {self.config_path}: {e}")
            raise
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while loading configuration: {e}")
            raise

    def load_models(self, model_loader_map: Dict[str, Any]):
        """
        Load models as specified in the configuration.
        
        Args:
            model_loader_map: A mapping from model type to a model loader class.
        """
        self.logger.info("Loading models...")
        model_configs = self.config.get("models", {})
        
        for model_name, config in model_configs.items():
            model_type = config.get("type")
            if model_type in model_loader_map:
                try:
                    loader = model_loader_map[model_type](config)
                    self.models[model_name] = loader.load_model()
                    self.logger.info(f"Loaded model '{model_name}' of type '{model_type}'")
                except Exception as e:
                    self.logger.error(f"Failed to load model '{model_name}': {e}")
            else:
                self.logger.warning(f"No loader available for model type '{model_type}'")

    def load_tasks(self, task_map: Dict[str, Any]):
        """
        Load tasks as specified in the configuration.
        
        Args:
            task_map: A mapping from task type to a benchmark task class.
        """
        self.logger.info("Loading tasks...")
        task_configs = self.config.get("tasks", {})
        
        for task_name, config in task_configs.items():
            task_type = config.get("type")
            if task_type in task_map:
                try:
                    self.tasks[task_name] = task_map[task_type](config)
                    self.logger.info(f"Loaded task '{task_name}' of type '{task_type}'")
                except Exception as e:
                    self.logger.error(f"Failed to load task '{task_name}': {e}")
            else:
                self.logger.warning(f"No implementation available for task type '{task_type}'")

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
                    self.logger.error(f"    ERROR running task {task_name} for model {model_name}: {e}")
                    self.results[model_name][task_name] = {"error": str(e)}
        
        self.logger.info("All benchmark runs complete.")
        return self.results