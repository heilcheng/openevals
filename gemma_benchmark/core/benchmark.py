"""
Core benchmarking framework for Gemma models with enhanced error handling and authentication.
"""

import os
import yaml
import json
import importlib
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from ..auth import ensure_authenticated, get_authentication_status
from ..utils.config_validation import validate_config


class BenchmarkError(Exception):
    """Custom exception for benchmark-related errors."""
    pass


class ModelLoadingError(BenchmarkError):
    """Exception raised when model loading fails."""
    pass


class TaskLoadingError(BenchmarkError):
    """Exception raised when task loading fails."""
    pass


class EvaluationError(BenchmarkError):
    """Exception raised during evaluation."""
    pass


class GemmaBenchmark:
    """Enhanced benchmarking class with comprehensive error handling and authentication."""
    
    def __init__(self, config_path: str, validate_config_schema: bool = True):
        """
        Initialize the benchmark with a configuration file.
        
        Args:
            config_path: Path to the YAML configuration file
            validate_config_schema: Whether to validate the configuration schema
        """
        self.logger = logging.getLogger("gemma_benchmark")
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # Validate configuration if requested
        if validate_config_schema:
            try:
                from gemma_benchmark.utils.config_validation import validate_config
                validate_config(self.config)
                self.logger.info("Configuration validation passed")    
            except ImportError:
                self.logger.warning("Config validation module not available, skipping validation")
            except Exception as e:    
                self.logger.warning(f"Configuration validation failed: {e}")
        
        self.models = {}
        self.tasks = {}
        self.results = {}
        self.evaluation_metadata = {
            "start_time": None,
            "end_time": None,
            "config_path": config_path,
            "total_models": 0,
            "total_tasks": 0,
            "successful_evaluations": 0,
            "failed_evaluations": 0
        }
        
        # Check authentication status
        auth_status = get_authentication_status()
        if auth_status["authenticated"]:
            self.logger.info(f"Authenticated as: {auth_status['username']}")
        else:
            self.logger.warning("Not authenticated with HuggingFace - model access may be limited")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and parse YAML configuration file with enhanced error handling."""
        self.logger.info(f"Loading configuration from {config_path}")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate basic structure
            required_sections = ["models", "tasks"]
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required configuration section: {section}")
                if not config[section]:
                    raise ValueError(f"Configuration section '{section}' cannot be empty")
            
            self.logger.info(f"Loaded configuration with {len(config['models'])} models and {len(config['tasks'])} tasks")
            return config
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration: {e}")
    
    def ensure_authentication(self) -> bool:
        """
        Ensure user is authenticated for model access.
        
        Returns:
            True if authentication successful, False otherwise
        """
        self.logger.info("Checking authentication...")
        
        if not ensure_authenticated():
            self.logger.error("Authentication failed. Some models may not be accessible.")
            return False
        
        return True
    
    def load_models(self, model_names: Optional[List[str]] = None) -> None:
        """
        Load specified models with enhanced error handling.
        
        Args:
            model_names: Optional list of model names to load (loads all if None)
        """
        self.logger.info("Loading models...")
        model_configs = self.config.get("models", {})
        
        if model_names:
            # Filter to only requested models
            missing_models = [name for name in model_names if name not in model_configs]
            if missing_models:
                raise ValueError(f"Requested models not found in config: {missing_models}")
            model_configs = {k: v for k, v in model_configs.items() if k in model_names}
        
        failed_models = []
        
        for model_name, model_config in model_configs.items():
            self.logger.info(f"Loading model: {model_name}")
            
            try:
                model_type = model_config.get("type", "gemma").lower()
                
                # Import the appropriate model loader
                module_path = "gemma_benchmark.core.model_loader"
                module = importlib.import_module(module_path)
                
                # Get the correct loader class
                if model_type == "gemma":
                    loader_class = getattr(module, "GemmaLoader")
                elif model_type == "mistral":
                    loader_class = getattr(module, "MistralLoader")
                elif model_type == "huggingface":
                    loader_class = getattr(module, "HuggingFaceLoader")
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")
                
                # Instantiate the loader and load the model
                model_loader = loader_class()
                
                # Prepare loading arguments
                load_kwargs = {
                    "cache_dir": model_config.get("cache_dir"),
                    "quantization": model_config.get("quantization", True),
                    "device_map": model_config.get("device_map", "auto"),
                    "max_memory": model_config.get("max_memory")
                }
                
                # Add model-specific arguments
                if model_type in ["gemma", "mistral"]:
                    load_kwargs.update({
                        "size": model_config.get("size", "2b"),
                        "variant": model_config.get("variant", "it")
                    })
                elif model_type == "huggingface":
                    load_kwargs.update({
                        "model_id": model_config.get("model_id"),
                        "tokenizer_id": model_config.get("tokenizer_id")
                    })
                
                # Remove None values
                load_kwargs = {k: v for k, v in load_kwargs.items() if v is not None}
                
                # Load the model
                model = model_loader.load_model(**load_kwargs)
                
                self.models[model_name] = {
                    "model": model, 
                    "config": model_config,
                    "loader_type": model_type
                }
                self.logger.info(f"Successfully loaded model: {model_name}")
                
            except Exception as e:
                error_msg = f"Failed to load model {model_name}: {e}"
                self.logger.error(error_msg)
                failed_models.append((model_name, str(e)))
                
                # Store failed model info for reporting
                self.models[model_name] = {
                    "model": None,
                    "config": model_config,
                    "error": str(e),
                    "status": "failed"
                }
        
        # Report results
        successful_models = len([m for m in self.models.values() if m.get("model") is not None])
        self.evaluation_metadata["total_models"] = len(model_configs)
        
        self.logger.info(f"Model loading complete: {successful_models}/{len(model_configs)} successful")
        
        if failed_models:
            self.logger.warning("Failed models:")
            for name, error in failed_models:
                self.logger.warning(f"  - {name}: {error}")
        
        if successful_models == 0:
            raise ModelLoadingError("No models were successfully loaded")
    
    def load_tasks(self, task_names: Optional[List[str]] = None) -> None:
        """
        Load specified tasks with enhanced error handling.
        
        Args:
            task_names: Optional list of task names to load (loads all if None)
        """
        self.logger.info("Loading tasks...")
        task_configs = self.config.get("tasks", {})
        
        if task_names:
            # Filter to only requested tasks
            missing_tasks = [name for name in task_names if name not in task_configs]
            if missing_tasks:
                raise ValueError(f"Requested tasks not found in config: {missing_tasks}")
            task_configs = {k: v for k, v in task_configs.items() if k in task_names}
        
        failed_tasks = []
        
        for task_name, task_config in task_configs.items():
            self.logger.info(f"Loading task: {task_name}")
            
            try:
                task_type = task_config.get("type", task_name).lower()
                
                # Import the task module
                module_path = f"gemma_benchmark.tasks.{task_type}"
                module = importlib.import_module(module_path)
                
                # Get the task class (convert to PascalCase + Benchmark)
                task_class_name = f"{task_type.capitalize()}Benchmark"
                if not hasattr(module, task_class_name):
                    # Try alternative naming conventions
                    alternative_names = [
                        f"{task_type.upper()}Benchmark",
                        f"{task_type.title()}Benchmark"
                    ]
                    task_class = None
                    for alt_name in alternative_names:
                        if hasattr(module, alt_name):
                            task_class = getattr(module, alt_name)
                            break
                    
                    if task_class is None:
                        raise AttributeError(f"Task class not found. Tried: {[task_class_name] + alternative_names}")
                else:
                    task_class = getattr(module, task_class_name)
                
                # Initialize the task
                task = task_class(task_config)
                self.tasks[task_name] = {
                    "task": task, 
                    "config": task_config,
                    "type": task_type
                }
                self.logger.info(f"Successfully loaded task: {task_name}")
                
            except Exception as e:
                error_msg = f"Failed to load task {task_name}: {e}"
                self.logger.error(error_msg)
                failed_tasks.append((task_name, str(e)))
                
                # Store failed task info for reporting
                self.tasks[task_name] = {
                    "task": None,
                    "config": task_config,
                    "error": str(e),
                    "status": "failed"
                }
        
        # Report results
        successful_tasks = len([t for t in self.tasks.values() if t.get("task") is not None])
        self.evaluation_metadata["total_tasks"] = len(task_configs)
        
        self.logger.info(f"Task loading complete: {successful_tasks}/{len(task_configs)} successful")
        
        if failed_tasks:
            self.logger.warning("Failed tasks:")
            for name, error in failed_tasks:
                self.logger.warning(f"  - {name}: {error}")
        
        if successful_tasks == 0:
            raise TaskLoadingError("No tasks were successfully loaded")
    
    def run_benchmarks(self) -> Dict[str, Dict[str, Any]]:
        """
        Run all loaded benchmarks with comprehensive error handling.
        
        Returns:
            Dictionary containing benchmark results
        """
        self.logger.info("Starting benchmark evaluation...")
        self.evaluation_metadata["start_time"] = datetime.datetime.now().isoformat()
        
        # Get only successfully loaded models and tasks
        successful_models = {k: v for k, v in self.models.items() if v.get("model") is not None}
        successful_tasks = {k: v for k, v in self.tasks.items() if v.get("task") is not None}
        
        if not successful_models:
            raise EvaluationError("No models available for evaluation")
        if not successful_tasks:
            raise EvaluationError("No tasks available for evaluation")
        
        total_evaluations = len(successful_models) * len(successful_tasks)
        current_evaluation = 0
        
        self.logger.info(f"Running {total_evaluations} evaluations ({len(successful_models)} models × {len(successful_tasks)} tasks)")
        
        # Create results structure
        for model_name in successful_models.keys():
            self.results[model_name] = {}
        
        # Run evaluations
        for model_name, model_info in successful_models.items():
            model = model_info["model"]
            
            for task_name, task_info in successful_tasks.items():
                current_evaluation += 1
                task = task_info["task"]
                
                self.logger.info(f"[{current_evaluation}/{total_evaluations}] Evaluating {model_name} on {task_name}")
                
                try:
                    # Run the evaluation with timeout handling if needed
                    result = task.evaluate(model)
                    
                    # Add metadata
                    result["evaluation_metadata"] = {
                        "model_name": model_name,
                        "task_name": task_name,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "model_config": model_info["config"],
                        "task_config": task_info["config"]
                    }
                    
                    self.results[model_name][task_name] = result
                    self.evaluation_metadata["successful_evaluations"] += 1
                    
                    # Log key metrics if available
                    if "overall" in result and "accuracy" in result["overall"]:
                        accuracy = result["overall"]["accuracy"]
                        self.logger.info(f"  Result: {accuracy:.4f} accuracy")
                    
                except Exception as e:
                    error_msg = f"Error evaluating {model_name} on {task_name}: {e}"
                    self.logger.error(error_msg)
                    
                    self.results[model_name][task_name] = {
                        "error": str(e),
                        "status": "failed",
                        "evaluation_metadata": {
                            "model_name": model_name,
                            "task_name": task_name,
                            "timestamp": datetime.datetime.now().isoformat(),
                            "error_type": type(e).__name__
                        }
                    }
                    self.evaluation_metadata["failed_evaluations"] += 1
        
        self.evaluation_metadata["end_time"] = datetime.datetime.now().isoformat()
        
        # Log summary
        success_rate = (self.evaluation_metadata["successful_evaluations"] / total_evaluations) * 100
        self.logger.info(f"Benchmark evaluation complete!")
        self.logger.info(f"Success rate: {success_rate:.1f}% ({self.evaluation_metadata['successful_evaluations']}/{total_evaluations})")
        
        return self.results
    
    def save_results(self, output_path: Optional[str] = None) -> str:
        """
        Save results to disk with enhanced metadata.
        
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
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare complete results package
        complete_results = {
            "results": self.results,
            "metadata": self.evaluation_metadata,
            "config": self.config,
            "system_info": self._get_system_info()
        }
        
        # Save as YAML (primary format)
        self.logger.info(f"Saving results to {output_path}")
        with open(output_path, 'w') as f:
            yaml.dump(complete_results, f, default_flow_style=False)
        
        # Also save as JSON for easier programmatic access
        json_path = output_path.replace('.yaml', '.json').replace('.yml', '.json')
        with open(json_path, 'w') as f:
            json.dump(complete_results, f, indent=2, default=str)
        
        # Save summary report
        summary_path = os.path.join(output_dir, "summary.txt")
        self._save_summary_report(summary_path)
        
        self.logger.info(f"Results saved to: {output_path}")
        self.logger.info(f"JSON format: {json_path}")
        self.logger.info(f"Summary: {summary_path}")
        
        return output_path
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for metadata."""
        import platform
        
        info = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        try:
            import torch
            info["pytorch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["cuda_version"] = torch.version.cuda
                info["gpu_count"] = torch.cuda.device_count()
        except ImportError:
            pass
        
        return info
    
    def _save_summary_report(self, output_path: str) -> None:
        """Save a human-readable summary report."""
        with open(output_path, 'w') as f:
            f.write("=== GEMMA BENCHMARK SUMMARY ===\n\n")
            
            # Metadata
            f.write(f"Config: {self.config_path}\n")
            f.write(f"Start: {self.evaluation_metadata['start_time']}\n")
            f.write(f"End: {self.evaluation_metadata['end_time']}\n")
            f.write(f"Success Rate: {(self.evaluation_metadata['successful_evaluations'] / (self.evaluation_metadata['successful_evaluations'] + self.evaluation_metadata['failed_evaluations']) * 100):.1f}%\n\n")
            
            # Results
            f.write("=== RESULTS ===\n\n")
            for model_name, model_results in self.results.items():
                f.write(f"Model: {model_name}\n")
                for task_name, task_result in model_results.items():
                    if "error" in task_result:
                        f.write(f"  {task_name}: FAILED - {task_result['error']}\n")
                    elif "overall" in task_result and "accuracy" in task_result["overall"]:
                        accuracy = task_result["overall"]["accuracy"]
                        f.write(f"  {task_name}: {accuracy:.4f}\n")
                    else:
                        f.write(f"  {task_name}: COMPLETED\n")
                f.write("\n")