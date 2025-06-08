"""
Configuration validation utilities for the Gemma Benchmarking Suite.

This module provides validation for YAML configuration files to ensure
they contain all required fields and valid values.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum


class ModelType(str, Enum):
    """Supported model types."""
    GEMMA = "gemma"
    MISTRAL = "mistral"
    LLAMA = "llama"
    HUGGINGFACE = "huggingface"


class TaskType(str, Enum):
    """Supported task types."""
    MMLU = "mmlu"
    GSM8K = "gsm8k"
    HUMANEVAL = "humaneval"
    ARC = "arc"
    TRUTHFULQA = "truthfulqa"
    EFFICIENCY = "efficiency"


class DeviceType(str, Enum):
    """Supported device types."""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class PrecisionType(str, Enum):
    """Supported precision types."""
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"


class ModelConfig(BaseModel):
    """Configuration for a single model."""
    type: ModelType
    size: str = Field(..., description="Model size (e.g., '2b', '7b', '9b')")
    variant: str = Field(default="it", description="Model variant")
    cache_dir: Optional[str] = Field(default=None, description="Model cache directory")
    quantization: bool = Field(default=True, description="Enable 4-bit quantization")
    device_map: Optional[Union[str, Dict[str, str]]] = Field(default="auto", description="Device mapping")
    max_memory: Optional[Dict[str, str]] = Field(default=None, description="Maximum memory per device")
    model_id: Optional[str] = Field(default=None, description="Custom model ID for HuggingFace models")
    
    @validator('size')
    def validate_size(cls, v):
        """Validate model size format."""
        if not v.endswith('b') and not v.isdigit():
            raise ValueError("Model size must end with 'b' (e.g., '2b', '7b') or be a number")
        return v.lower()
    
    @validator('variant')
    def validate_variant(cls, v):
        """Validate model variant."""
        valid_variants = ['it', 'instruct', 'chat', 'base']
        if v.lower() not in valid_variants:
            raise ValueError(f"Model variant must be one of: {valid_variants}")
        return v.lower()


class TaskConfig(BaseModel):
    """Base configuration for benchmark tasks."""
    type: TaskType
    
    class Config:
        extra = "allow"  # Allow task-specific fields


class MMLUTaskConfig(TaskConfig):
    """Configuration for MMLU benchmark."""
    type: TaskType = Field(TaskType.MMLU, const=True)
    subset: str = Field(default="all", description="Subject subset or 'all'")
    shot_count: int = Field(default=5, ge=0, le=10, description="Number of few-shot examples")
    
    @validator('subset')
    def validate_subset(cls, v):
        """Validate MMLU subset."""
        valid_subsets = [
            "all", "mathematics", "computer_science", "physics", "chemistry",
            "biology", "philosophy", "history", "law", "economics", "psychology"
        ]
        if v not in valid_subsets:
            # Allow any string but warn about unknown subsets
            logging.getLogger(__name__).warning(f"Unknown MMLU subset: {v}")
        return v


class GSM8KTaskConfig(TaskConfig):
    """Configuration for GSM8K benchmark."""
    type: TaskType = Field(TaskType.GSM8K, const=True)
    shot_count: int = Field(default=5, ge=0, le=10, description="Number of few-shot examples")
    use_chain_of_thought: bool = Field(default=True, description="Use chain-of-thought prompting")


class HumanEvalTaskConfig(TaskConfig):
    """Configuration for HumanEval benchmark."""
    type: TaskType = Field(TaskType.HUMANEVAL, const=True)
    timeout: int = Field(default=10, ge=5, le=60, description="Code execution timeout in seconds")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="Sampling temperature")
    max_new_tokens: int = Field(default=256, ge=50, le=1024, description="Maximum tokens to generate")


class EfficiencyTaskConfig(TaskConfig):
    """Configuration for efficiency benchmark."""
    type: TaskType = Field(TaskType.EFFICIENCY, const=True)
    sample_prompts: List[str] = Field(
        default_factory=lambda: [
            "Explain the theory of relativity",
            "Write a short story about a robot",
            "Summarize the key events of World War II"
        ],
        description="List of sample prompts for testing"
    )
    output_lengths: List[int] = Field(
        default_factory=lambda: [128, 256, 512, 1024],
        description="List of output lengths to test"
    )
    
    @validator('sample_prompts')
    def validate_sample_prompts(cls, v):
        """Validate sample prompts."""
        if len(v) < 1:
            raise ValueError("At least one sample prompt is required")
        if any(len(prompt.strip()) == 0 for prompt in v):
            raise ValueError("All sample prompts must be non-empty")
        return v
    
    @validator('output_lengths')
    def validate_output_lengths(cls, v):
        """Validate output lengths."""
        if len(v) < 1:
            raise ValueError("At least one output length is required")
        if any(length <= 0 for length in v):
            raise ValueError("All output lengths must be positive")
        return sorted(v)  # Sort for consistency


class EvaluationConfig(BaseModel):
    """Configuration for evaluation settings."""
    runs: int = Field(default=1, ge=1, le=10, description="Number of evaluation runs")
    batch_size: Union[int, str] = Field(default="auto", description="Batch size or 'auto'")
    max_batch_size: int = Field(default=32, ge=1, description="Maximum batch size when using auto")
    statistical_tests: bool = Field(default=False, description="Enable statistical significance testing")
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99, description="Confidence level for statistics")
    
    @validator('batch_size')
    def validate_batch_size(cls, v):
        """Validate batch size."""
        if isinstance(v, str) and v != "auto":
            raise ValueError("Batch size must be a positive integer or 'auto'")
        if isinstance(v, int) and v <= 0:
            raise ValueError("Batch size must be positive")
        return v


class OutputConfig(BaseModel):
    """Configuration for output settings."""
    path: str = Field(default="results", description="Output directory path")
    save_predictions: bool = Field(default=False, description="Save individual predictions")
    visualize: bool = Field(default=True, description="Generate visualizations")
    export_formats: List[str] = Field(
        default_factory=lambda: ["yaml", "json"],
        description="Export formats for results"
    )
    dashboard: bool = Field(default=False, description="Launch interactive dashboard")
    
    @validator('export_formats')
    def validate_export_formats(cls, v):
        """Validate export formats."""
        valid_formats = ["yaml", "json", "csv", "xlsx"]
        for fmt in v:
            if fmt not in valid_formats:
                raise ValueError(f"Invalid export format: {fmt}. Valid formats: {valid_formats}")
        return v


class HardwareConfig(BaseModel):
    """Configuration for hardware settings."""
    device: DeviceType = Field(default=DeviceType.AUTO, description="Device to use")
    precision: PrecisionType = Field(default=PrecisionType.BFLOAT16, description="Floating point precision")
    quantization: bool = Field(default=True, description="Enable quantization")
    mixed_precision: bool = Field(default=True, description="Enable mixed precision training")
    gradient_checkpointing: bool = Field(default=False, description="Enable gradient checkpointing")
    torch_compile: bool = Field(default=False, description="Enable torch.compile optimization")


class BenchmarkConfig(BaseModel):
    """Complete benchmark configuration."""
    models: Dict[str, ModelConfig]
    tasks: Dict[str, Union[
        MMLUTaskConfig, 
        GSM8KTaskConfig, 
        HumanEvalTaskConfig, 
        EfficiencyTaskConfig,
        TaskConfig  # Fallback for other task types
    ]]
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    
    @validator('models')
    def validate_models_not_empty(cls, v):
        """Ensure at least one model is configured."""
        if len(v) == 0:
            raise ValueError("At least one model must be configured")
        return v
    
    @validator('tasks')
    def validate_tasks_not_empty(cls, v):
        """Ensure at least one task is configured."""
        if len(v) == 0:
            raise ValueError("At least one task must be configured")
        return v
    
    @root_validator
    def validate_hardware_compatibility(cls, values):
        """Validate hardware configuration compatibility."""
        hardware = values.get('hardware', {})
        models = values.get('models', {})
        
        # Check quantization compatibility
        if hardware.get('quantization') and hardware.get('precision') == PrecisionType.FLOAT32:
            logging.getLogger(__name__).warning(
                "Quantization is less effective with float32 precision. Consider using bfloat16."
            )
        
        # Check model size vs quantization
        for model_name, model_config in models.items():
            if isinstance(model_config, dict):
                model_config = ModelConfig(**model_config)
            
            size = model_config.size.rstrip('b')
            if size.isdigit() and int(size) >= 9 and not model_config.quantization:
                logging.getLogger(__name__).warning(
                    f"Model {model_name} ({model_config.size}) is large but quantization is disabled. "
                    "This may cause memory issues."
                )
        
        return values


def validate_config(config_dict: Dict[str, Any]) -> BenchmarkConfig:
    """
    Validate a configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary loaded from YAML
        
    Returns:
        Validated BenchmarkConfig object
        
    Raises:
        ValidationError: If configuration is invalid
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Validate required sections first
        required_sections = ['models', 'tasks']
        for section in required_sections:
            if section not in config_dict:
                raise ValueError(f"Missing required section: {section}")
        
        # Parse task configurations with better error handling
        parsed_tasks = {}
        for task_name, task_config in config_dict.get('tasks', {}).items():
            try:
                task_type = task_config.get('type')
                
                if task_type == TaskType.MMLU:
                    parsed_tasks[task_name] = MMLUTaskConfig(**task_config)
                elif task_type == TaskType.GSM8K:
                    parsed_tasks[task_name] = GSM8KTaskConfig(**task_config)
                elif task_type == TaskType.HUMANEVAL:
                    parsed_tasks[task_name] = HumanEvalTaskConfig(**task_config)
                elif task_type == TaskType.EFFICIENCY:
                    parsed_tasks[task_name] = EfficiencyTaskConfig(**task_config)
                else:
                    logger.warning(f"Unknown task type {task_type} for task {task_name}, using base config")
                    parsed_tasks[task_name] = TaskConfig(**task_config)
            except Exception as e:
                logger.error(f"Failed to parse task {task_name}: {e}")
                raise ValueError(f"Invalid configuration for task {task_name}: {e}")
        
        config_dict['tasks'] = parsed_tasks
        
        
        # Validate the complete configuration
        validated_config = BenchmarkConfig(**config_dict)
        
        logger.info("Configuration validation successful")
        return validated_config
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise


def create_example_config() -> Dict[str, Any]:
    """
    Create an example configuration dictionary.
    
    Returns:
        Example configuration dictionary
    """
    return {
        "models": {
            "gemma-2b": {
                "type": "gemma",
                "size": "2b",
                "variant": "it",
                "quantization": True
            }
        },
        "tasks": {
            "mmlu": {
                "type": "mmlu",
                "subset": "mathematics",
                "shot_count": 5
            },
            "efficiency": {
                "type": "efficiency",
                "sample_prompts": [
                    "Explain quantum computing",
                    "Write a Python function"
                ],
                "output_lengths": [128, 256]
            }
        },
        "evaluation": {
            "runs": 1,
            "batch_size": "auto"
        },
        "output": {
            "path": "results",
            "visualize": True
        },
        "hardware": {
            "device": "auto",
            "precision": "bfloat16",
            "quantization": True
        }
    }


def validate_config_file(config_path: str) -> BenchmarkConfig:
    """
    Validate a configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Validated BenchmarkConfig object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If configuration is invalid
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return validate_config(config_dict)