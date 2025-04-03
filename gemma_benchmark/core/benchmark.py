"""
Core benchmarking framework for Gemma models.
"""

import os
import yaml
import importlib
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

class GemmaBenchmark:
    """Main benchmarking class that orchestrates the evaluation process."""
    
    def __init__(self, config_path: str):
        """
        Initialize the benchmark with a configuration file.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.logger = logging.getLogger("gemma_benchmark")
        self.config = self._load_config(config_path)
        self.models = {}
        self.tasks = {}
        self.results = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and parse YAML configuration file."""
        self.logger.info(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def load_models(self, model_names: Optional[List[str]] = None) -> None:
        """
        Load specified models or all models in config.
        
        Args:
            model_names: Optional list of model names to load (loads all if None)
        """
        self.logger.info("Loading models...")
        model_configs = self.config.get("models", {})
        
        if model_names:
            model_configs = {k: v for k, v in model_configs.items() if k in model_names}
        
        for model_name, model_config in model_configs.items():
            self.logger.info(f"Loading model: {model_name}")
            model_type = model_config.get("type", "gemma")
            
            # Import the appropriate model loader
            try:
                # Dynamic import based on model type
                module_path = f"gemma_benchmark.core.model_loader"
                module = importlib.import_module(module_path)
                model_loader_class = getattr(module, f"{model_type.capitalize()}Loader")
                
                # Instantiate the loader and load the model
                model_loader = model_loader_class()
                model = model_loader.load_model(
                    size=model_config.get("size", "2b"),
                    variant=model_config.get("variant", "it"),
                    cache_dir=model_config.get("cache_dir")
                )
                
                self.models[model_name] = {"model": model, "config": model_config}
                self.logger.info(f"Successfully loaded model: {model_name}")
            except (ImportError, AttributeError) as e:
                self.logger.error(f"Failed to load model {model_name}: {e}")
    
    def load_tasks(self, task_names: Optional[List[str]] = None) -> None:
        """
        Load specified tasks or all tasks in config.
        
        Args:
            task_names: Optional list of task names to load (loads all if None)
        """
        self.logger.info("Loading tasks...")
        task_configs = self.config.get("tasks", {})
        
        if task_names:
            task_configs = {k: v for k, v in task_configs.items() if k in task_names}
        
        for task_name, task_config in task_configs.items():
            self.logger.info(f"Loading task: {task_name}")
            task_type = task_config.get("type", task_name)
            
            try:
                # Import the task module
                module_path = f"gemma_benchmark.tasks.{task_type}"
                module = importlib.import_module(module_path)
                
                # Get the task class
                task_class_name = f"{task_type.capitalize()}Benchmark"
                task_class = getattr(module, task_class_name)
                
                # Initialize the task
                task = task_class(task_config)
                self.tasks[task_name] = {"task": task, "config": task_config}
                self.logger.info(f"Successfully loaded task: {task_name}")
            except (ImportError, AttributeError) as e:
                self.logger.error(f"Failed to load task {task_name}: {e}")
    
    def run_benchmarks(self) -> Dict[str, Dict[str, Any]]:
        """
        Run all loaded benchmarks for all loaded models.
        
        Returns:
            Dictionary containing benchmark results
        """
        self.logger.info("Running benchmarks...")
        
        # Create results structure
        for model_name, model_info in self.models.items():
            self.results[model_name] = {}
            model = model_info["model"]
            
            for task_name, task_info in self.tasks.items():
                self.logger.info(f"Evaluating {model_name} on {task_name}...")
                task = task_info["task"]
                
                try:
                    # Run the evaluation
                    result = task.evaluate(model)
                    self.results[model_name][task_name] = result
                    self.logger.info(f"Completed evaluation of {model_name} on {task_name}")
                except Exception as e:
                    self.logger.error(f"Error evaluating {model_name} on {task_name}: {e}")
                    self.results[model_name][task_name] = {"error": str(e)}
        
        return self.results
    
    def save_results(self, output_path: Optional[str] = None) -> str:
        """
        Save results to disk.
        
        Args:
            output_path: Path to save results (defaults to timestamp-based path)
            
        Returns:
            Path where results were saved
        """
        if output_path is None:
            # Generate timestamp-based directory
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.config.get("output", {}).get("path", "results")
            output_path = os.path.join(output_dir, timestamp, "results.yaml")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as YAML
        self.logger.info(f"Saving results to {output_path}")
        with open(output_path, 'w') as f:
            yaml.dump(self.results, f, default_flow_style=False)
        
        return output_path