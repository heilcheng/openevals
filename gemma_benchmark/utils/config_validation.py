"""
Comprehensive configuration validation system for the Gemma Benchmarking Suite.

This module provides robust configuration validation using Pydantic with:
- Schema validation for all configuration sections
- Detailed error reporting with suggestions
- Type checking and value range validation
- Configuration file templates and examples
- Backward compatibility support
"""

import logging
import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Literal
from enum import Enum

try:
    from pydantic import BaseModel, Field, validator, root_validator, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    logging.getLogger(__name__).error("pydantic not available. Install with: pip install pydantic>=2.0.0")


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""
    pass


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
    """Configuration schema for a single model."""
    
    type: ModelType = Field(..., description="Model type (gemma, mistral, llama, huggingface)")
    size: str = Field(..., description="Model size (e.g., '2b', '7b', '9b')")
    variant: str = Field(default="it", description="Model variant (it, instruct, chat, base)")
    cache_dir: Optional[str] = Field(default=None, description="Model cache directory")
    quantization: bool = Field(default=True, description="Enable 4-bit quantization")
    device_map: Union[str, Dict[str, str]] = Field(default="auto", description="Device mapping strategy")
    max_memory: Optional[Dict[str, str]] = Field(default=None, description="Maximum memory per device")
    model_id: Optional[str] = Field(default=None, description="Custom model ID for HuggingFace models")
    torch_dtype: Optional[str] = Field(default="bfloat16", description="PyTorch data type")
    
    @validator('size')
    def validate_size(cls, v):
        """Validate model size format."""
        v = str(v).lower()
        # Allow patterns like "2b", "7b", "13b" or just numbers
        import re
        if not re.match(r'^\d+[bm]?$', v):
            raise ValueError(
                f"Invalid model size '{v}'. Expected format: '2b', '7b', '13b', etc. "
                f"or just numbers like '2', '7', '13'"
            )
        # Normalize to include 'b' suffix
        if not v.endswith(('b', 'm')):
            v = v + 'b'
        return v
    
    @validator('variant')
    def validate_variant(cls, v):
        """Validate model variant."""
        valid_variants = ['it', 'instruct', 'chat', 'base', 'code']
        if v.lower() not in valid_variants:
            raise ValueError(
                f"Invalid model variant '{v}'. Valid options: {valid_variants}. "
                f"Most common: 'it' (instruction-tuned) or 'base'"
            )
        return v.lower()
    
    @validator('torch_dtype')
    def validate_torch_dtype(cls, v):
        """Validate PyTorch data type."""
        if v is None:
            return v
        valid_dtypes = ['float32', 'float16', 'bfloat16', 'int8', 'int4']
        if v not in valid_dtypes:
            raise ValueError(
                f"Invalid torch_dtype '{v}'. Valid options: {valid_dtypes}. "
                f"Recommended: 'bfloat16' for most models"
            )
        return v
    
    @validator('device_map')
    def validate_device_map(cls, v):
        """Validate device mapping configuration."""
        if isinstance(v, str):
            valid_strategies = ['auto', 'balanced', 'balanced_low_0', 'sequential']
            if v not in valid_strategies:
                raise ValueError(
                    f"Invalid device_map strategy '{v}'. Valid options: {valid_strategies}. "
                    f"Recommended: 'auto' for automatic device placement"
                )
        return v
    
    @root_validator
    def validate_model_compatibility(cls, values):
        """Validate model type and size compatibility."""
        model_type = values.get('type')
        size = values.get('size', '').lower()
        model_id = values.get('model_id')
        
        # Gemma-specific validation
        if model_type == ModelType.GEMMA:
            valid_gemma_sizes = ['2b', '9b', '27b']
            if size not in valid_gemma_sizes:
                raise ValueError(
                    f"Invalid Gemma model size '{size}'. Valid Gemma sizes: {valid_gemma_sizes}"
                )
        
        # HuggingFace type requires model_id
        if model_type == ModelType.HUGGINGFACE and not model_id:
            raise ValueError(
                "model_id is required when type is 'huggingface'. "
                "Example: model_id: 'microsoft/DialoGPT-medium'"
            )
        
        # Quantization warnings for large models
        quantization = values.get('quantization', True)
        if not quantization and size in ['27b', '70b']:
            import warnings
            warnings.warn(
                f"Large model ({size}) without quantization may cause memory issues. "
                f"Consider enabling quantization=true"
            )
        
        return values

    class Config:
        use_enum_values = True


class BaseTaskConfig(BaseModel):
    """Base configuration for all benchmark tasks."""
    
    type: TaskType = Field(..., description="Task type")
    
    class Config:
        extra = "allow"  # Allow task-specific fields
        use_enum_values = True


class MMLUTaskConfig(BaseTaskConfig):
    """Configuration for MMLU benchmark."""
    
    type: Literal[TaskType.MMLU] = TaskType.MMLU
    subset: str = Field(default="all", description="Subject subset or 'all'")
    shot_count: int = Field(default=5, ge=0, le=10, description="Number of few-shot examples")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Sampling temperature")
    max_new_tokens: int = Field(default=10, ge=1, le=100, description="Maximum tokens to generate")
    
    @validator('subset')
    def validate_subset(cls, v):
        """Validate MMLU subset."""
        valid_subsets = [
            "all", "mathematics", "computer_science", "physics", "chemistry",
            "biology", "philosophy", "history", "law", "economics", "psychology",
            "social_sciences", "humanities", "stem", "other"
        ]
        if v not in valid_subsets:
            # Allow any string but warn about unknown subsets
            import warnings
            warnings.warn(
                f"Unknown MMLU subset '{v}'. Valid options: {valid_subsets[:10]}... "
                f"Using anyway, but verify this is correct."
            )
        return v


class GSM8KTaskConfig(BaseTaskConfig):
    """Configuration for GSM8K benchmark."""
    
    type: Literal[TaskType.GSM8K] = TaskType.GSM8K
    shot_count: int = Field(default=5, ge=0, le=10, description="Number of few-shot examples")
    use_chain_of_thought: bool = Field(default=True, description="Use chain-of-thought prompting")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="Sampling temperature")
    max_new_tokens: int = Field(default=300, ge=50, le=1000, description="Maximum tokens for solution")


class HumanEvalTaskConfig(BaseTaskConfig):
    """Configuration for HumanEval benchmark."""
    
    type: Literal[TaskType.HUMANEVAL] = TaskType.HUMANEVAL
    timeout: int = Field(default=10, ge=5, le=60, description="Code execution timeout in seconds")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="Sampling temperature")
    max_new_tokens: int = Field(default=256, ge=50, le=1024, description="Maximum tokens to generate")
    num_samples: int = Field(default=164, ge=1, le=500, description="Number of problems to evaluate")


class ARCTaskConfig(BaseTaskConfig):
    """Configuration for ARC benchmark."""
    
    type: Literal[TaskType.ARC] = TaskType.ARC
    subset: Literal["ARC-Easy", "ARC-Challenge"] = Field(default="ARC-Challenge", description="ARC subset")
    shot_count: int = Field(default=5, ge=0, le=10, description="Number of few-shot examples")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="Sampling temperature")


class TruthfulQATaskConfig(BaseTaskConfig):
    """Configuration for TruthfulQA benchmark."""
    
    type: Literal[TaskType.TRUTHFULQA] = TaskType.TRUTHFULQA
    task_type: Literal["mc1", "mc2", "generation"] = Field(default="mc1", description="TruthfulQA task type")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_new_tokens: int = Field(default=100, ge=20, le=500, description="Maximum tokens for generation")


class EfficiencyTaskConfig(BaseTaskConfig):
    """Configuration for efficiency benchmark."""
    
    type: Literal[TaskType.EFFICIENCY] = TaskType.EFFICIENCY
    sample_prompts: List[str] = Field(
        default_factory=lambda: [
            "Explain the theory of relativity in simple terms",
            "Write a short story about a robot who discovers emotions",
            "Summarize the key events of World War II"
        ],
        min_items=1,
        description="List of sample prompts for testing"
    )
    output_lengths: List[int] = Field(
        default_factory=lambda: [64, 128, 256, 512],
        min_items=1,
        description="List of output lengths to test"
    )
    warmup_runs: int = Field(default=2, ge=1, le=5, description="Number of warmup iterations")
    measurement_runs: int = Field(default=3, ge=1, le=10, description="Number of measurement runs")
    
    @validator('sample_prompts')
    def validate_sample_prompts(cls, v):
        """Validate sample prompts."""
        if not v or len(v) == 0:
            raise ValueError("At least one sample prompt is required")
        
        for i, prompt in enumerate(v):
            if not prompt or not prompt.strip():
                raise ValueError(f"Sample prompt {i+1} is empty")
            if len(prompt.strip()) < 10:
                raise ValueError(f"Sample prompt {i+1} is too short (minimum 10 characters)")
        
        return [prompt.strip() for prompt in v]
    
    @validator('output_lengths')
    def validate_output_lengths(cls, v):
        """Validate output lengths."""
        if not v or len(v) == 0:
            raise ValueError("At least one output length is required")
        
        for length in v:
            if length <= 0:
                raise ValueError(f"Output length must be positive, got {length}")
            if length > 4096:
                import warnings
                warnings.warn(f"Large output length ({length}) may cause memory issues")
        
        return sorted(v)  # Sort for consistency


class EvaluationConfig(BaseModel):
    """Configuration for evaluation settings."""
    
    runs: int = Field(default=1, ge=1, le=10, description="Number of evaluation runs")
    batch_size: Union[int, str] = Field(default="auto", description="Batch size or 'auto'")
    max_batch_size: int = Field(default=32, ge=1, le=128, description="Maximum batch size when using auto")
    statistical_tests: bool = Field(default=False, description="Enable statistical significance testing")
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99, description="Confidence level for statistics")
    timeout: int = Field(default=3600, ge=60, le=86400, description="Maximum evaluation time in seconds")
    
    @validator('batch_size')
    def validate_batch_size(cls, v):
        """Validate batch size."""
        if isinstance(v, str):
            if v != "auto":
                raise ValueError("Batch size must be a positive integer or 'auto'")
        elif isinstance(v, int):
            if v <= 0:
                raise ValueError("Batch size must be positive")
            if v > 128:
                import warnings
                warnings.warn(f"Large batch size ({v}) may cause memory issues")
        else:
            raise ValueError("Batch size must be an integer or 'auto'")
        return v


class OutputConfig(BaseModel):
    """Configuration for output settings."""
    
    path: str = Field(default="results", description="Output directory path")
    save_predictions: bool = Field(default=False, description="Save individual predictions")
    save_detailed_results: bool = Field(default=True, description="Save detailed benchmark results")
    visualize: bool = Field(default=True, description="Generate visualizations")
    export_formats: List[str] = Field(
        default_factory=lambda: ["yaml", "json"],
        description="Export formats for results"
    )
    dashboard: bool = Field(default=False, description="Launch interactive dashboard")
    compress_results: bool = Field(default=False, description="Compress result files")
    
    @validator('export_formats')
    def validate_export_formats(cls, v):
        """Validate export formats."""
        valid_formats = ["yaml", "json", "csv", "xlsx", "pickle"]
        invalid_formats = [fmt for fmt in v if fmt not in valid_formats]
        
        if invalid_formats:
            raise ValueError(
                f"Invalid export formats: {invalid_formats}. "
                f"Valid formats: {valid_formats}"
            )
        return v
    
    @validator('path')
    def validate_output_path(cls, v):
        """Validate output path."""
        try:
            # Check if path is writable
            path = Path(v)
            path.mkdir(parents=True, exist_ok=True)
            
            # Test write permissions
            test_file = path / ".write_test"
            test_file.touch()
            test_file.unlink()
            
        except Exception as e:
            raise ValueError(f"Output path '{v}' is not writable: {str(e)}")
        
        return str(path)


class HardwareConfig(BaseModel):
    """Configuration for hardware settings."""
    
    device: DeviceType = Field(default=DeviceType.AUTO, description="Device to use")
    precision: PrecisionType = Field(default=PrecisionType.BFLOAT16, description="Floating point precision")
    quantization: bool = Field(default=True, description="Enable quantization")
    mixed_precision: bool = Field(default=True, description="Enable mixed precision")
    gradient_checkpointing: bool = Field(default=False, description="Enable gradient checkpointing")
    torch_compile: bool = Field(default=False, description="Enable torch.compile optimization")
    memory_limit: Optional[str] = Field(default=None, description="Memory limit (e.g., '16GB')")
    
    @validator('memory_limit')
    def validate_memory_limit(cls, v):
        """Validate memory limit format."""
        if v is None:
            return v
        
        import re
        if not re.match(r'^\d+(\.\d+)?[GMK]B$', v.upper()):
            raise ValueError(
                f"Invalid memory limit format '{v}'. "
                f"Expected format: '16GB', '8.5GB', '1024MB', etc."
            )
        return v.upper()
    
    @root_validator
    def validate_hardware_compatibility(cls, values):
        """Validate hardware configuration compatibility."""
        quantization = values.get('quantization', True)
        precision = values.get('precision')
        device = values.get('device')
        
        # Warn about suboptimal combinations
        if quantization and precision == PrecisionType.FLOAT32:
            import warnings
            warnings.warn(
                "Quantization is less effective with float32 precision. "
                "Consider using bfloat16 or float16 for better memory savings."
            )
        
        # Check device availability
        if device == DeviceType.CUDA:
            try:
                import torch
                if not torch.cuda.is_available():
                    raise ValueError(
                        "CUDA device specified but not available. "
                        "Use device='auto' for automatic detection."
                    )
            except ImportError:
                pass  # PyTorch not available, will be caught elsewhere
        
        return values

    class Config:
        use_enum_values = True


class BenchmarkConfig(BaseModel):
    """Complete benchmark configuration with comprehensive validation."""
    
    models: Dict[str, ModelConfig] = Field(..., description="Model configurations")
    tasks: Dict[str, Union[
        MMLUTaskConfig,
        GSM8KTaskConfig, 
        HumanEvalTaskConfig,
        ARCTaskConfig,
        TruthfulQATaskConfig,
        EfficiencyTaskConfig,
        BaseTaskConfig
    ]] = Field(..., description="Task configurations")
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig, description="Evaluation settings")
    output: OutputConfig = Field(default_factory=OutputConfig, description="Output settings")
    hardware: HardwareConfig = Field(default_factory=HardwareConfig, description="Hardware settings")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('models')
    def validate_models_not_empty(cls, v):
        """Ensure at least one model is configured."""
        if not v or len(v) == 0:
            raise ValueError(
                "At least one model must be configured. "
                "Add a model configuration to the 'models' section."
            )
        
        # Validate model names
        for model_name in v.keys():
            if not model_name or not model_name.strip():
                raise ValueError("Model names cannot be empty")
            if len(model_name) > 50:
                raise ValueError(f"Model name '{model_name}' is too long (max 50 characters)")
        
        return v
    
    @validator('tasks')
    def validate_tasks_not_empty(cls, v):
        """Ensure at least one task is configured."""
        if not v or len(v) == 0:
            raise ValueError(
                "At least one task must be configured. "
                "Add a task configuration to the 'tasks' section."
            )
        
        # Validate task names
        for task_name in v.keys():
            if not task_name or not task_name.strip():
                raise ValueError("Task names cannot be empty")
        
        return v
    
    @root_validator
    def validate_resource_compatibility(cls, values):
        """Validate resource requirements and compatibility."""
        models = values.get('models', {})
        hardware = values.get('hardware', {})
        evaluation = values.get('evaluation', {})
        
        # Estimate total memory requirements
        total_memory_gb = 0
        large_model_count = 0
        
        for model_name, model_config in models.items():
            if isinstance(model_config, dict):
                size = model_config.get('size', '2b')
            else:
                size = model_config.size
            
            # Estimate memory usage
            size_num = int(size.rstrip('bm'))
            model_memory = size_num * 2  # Rough estimate: 2GB per billion parameters
            
            if model_config.quantization if hasattr(model_config, 'quantization') else True:
                model_memory *= 0.5  # Quantization reduces memory by ~50%
            
            total_memory_gb += model_memory
            
            if size_num >= 9:
                large_model_count += 1
        
        # Memory warnings
        if total_memory_gb > 32 and large_model_count > 1:
            import warnings
            warnings.warn(
                f"Configuration may require {total_memory_gb:.1f}GB memory for {large_model_count} large models. "
                f"Consider enabling quantization or reducing concurrent models."
            )
        
        # Batch size warnings for multiple large models
        batch_size = evaluation.get('batch_size', 'auto') if isinstance(evaluation, dict) else getattr(evaluation, 'batch_size', 'auto')
        if large_model_count > 1 and isinstance(batch_size, int) and batch_size > 4:
            import warnings
            warnings.warn(
                f"Large batch size ({batch_size}) with multiple large models may cause memory issues. "
                f"Consider using batch_size='auto' or reducing to 2-4."
            )
        
        return values

    class Config:
        extra = "forbid"  # Don't allow unknown fields at top level
        validate_assignment = True


class ConfigValidator:
    """Main configuration validator with enhanced error reporting."""
    
    def __init__(self):
        self.logger = logging.getLogger("gemma_benchmark.config_validator")
        
        if not PYDANTIC_AVAILABLE:
            raise ImportError(
                "Pydantic is required for configuration validation. "
                "Install with: pip install pydantic>=2.0.0"
            )
    
    def validate_config_dict(self, config_dict: Dict[str, Any]) -> BenchmarkConfig:
        """
        Validate a configuration dictionary with detailed error reporting.
        
        Args:
            config_dict: Configuration dictionary loaded from YAML
            
        Returns:
            Validated BenchmarkConfig object
            
        Raises:
            ConfigurationError: If validation fails with detailed error information
        """
        try:
            # Pre-process task configurations to use correct types
            if 'tasks' in config_dict:
                config_dict['tasks'] = self._process_task_configs(config_dict['tasks'])
            
            # Validate the complete configuration
            validated_config = BenchmarkConfig(**config_dict)
            
            self.logger.info("Configuration validation successful")
            return validated_config
            
        except ValidationError as e:
            # Convert Pydantic errors to user-friendly messages
            error_details = self._format_validation_errors(e)
            raise ConfigurationError(
                f"Configuration validation failed:\n{error_details}\n\n"
                f"See configuration documentation for valid options."
            )
        except Exception as e:
            raise ConfigurationError(f"Configuration validation error: {str(e)}")
    
    def _process_task_configs(self, tasks_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Process task configurations to use correct Pydantic models."""
        processed_tasks = {}
        
        for task_name, task_config in tasks_dict.items():
            task_type = task_config.get('type')
            
            try:
                if task_type == TaskType.MMLU:
                    processed_tasks[task_name] = MMLUTaskConfig(**task_config)
                elif task_type == TaskType.GSM8K:
                    processed_tasks[task_name] = GSM8KTaskConfig(**task_config)
                elif task_type == TaskType.HUMANEVAL:
                    processed_tasks[task_name] = HumanEvalTaskConfig(**task_config)
                elif task_type == TaskType.ARC:
                    processed_tasks[task_name] = ARCTaskConfig(**task_config)
                elif task_type == TaskType.TRUTHFULQA:
                    processed_tasks[task_name] = TruthfulQATaskConfig(**task_config)
                elif task_type == TaskType.EFFICIENCY:
                    processed_tasks[task_name] = EfficiencyTaskConfig(**task_config)
                else:
                    # Use base task config for unknown types
                    self.logger.warning(f"Unknown task type '{task_type}' for task '{task_name}', using base configuration")
                    processed_tasks[task_name] = BaseTaskConfig(**task_config)
                    
            except ValidationError as e:
                raise ConfigurationError(
                    f"Invalid configuration for task '{task_name}' (type: {task_type}):\n"
                    f"{self._format_validation_errors(e)}"
                )
        
        return processed_tasks
    
    def _format_validation_errors(self, validation_error: ValidationError) -> str:
        """Format Pydantic validation errors into user-friendly messages."""
        error_messages = []
        
        for error in validation_error.errors():
            location = " -> ".join(str(loc) for loc in error['loc'])
            message = error['msg']
            value = error.get('input', 'N/A')
            error_type = error['type']
            
            # Create user-friendly error message
            if error_type == 'missing':
                friendly_msg = f"Missing required field: {location}"
            elif error_type == 'value_error':
                friendly_msg = f"Invalid value for {location}: {message}"
            elif error_type in ['type_error', 'enum']:
                friendly_msg = f"Invalid type/value for {location}: {message} (got: {value})"
            else:
                friendly_msg = f"Error in {location}: {message}"
            
            error_messages.append(f"  • {friendly_msg}")
        
        return "\n".join(error_messages)
    
    def validate_config_file(self, config_path: str) -> BenchmarkConfig:
        """
        Validate a configuration file with enhanced error reporting.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Validated BenchmarkConfig object
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        if not config_path.suffix.lower() in ['.yaml', '.yml']:
            raise ConfigurationError(
                f"Configuration file must be YAML format (.yaml or .yml), got: {config_path.suffix}"
            )
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
                
            if config_dict is None:
                raise ConfigurationError("Configuration file is empty")
                
            if not isinstance(config_dict, dict):
                raise ConfigurationError("Configuration file must contain a YAML dictionary")
            
            return self.validate_config_dict(config_dict)
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML syntax in {config_path}: {str(e)}")
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        except PermissionError:
            raise ConfigurationError(f"Permission denied reading configuration file: {config_path}")


def create_example_config() -> Dict[str, Any]:
    """Create a comprehensive example configuration."""
    return {
        "models": {
            "gemma-2b": {
                "type": "gemma",
                "size": "2b",
                "variant": "it",
                "quantization": True,
                "cache_dir": "cache/models"
            },
            "gemma-9b": {
                "type": "gemma", 
                "size": "9b",
                "variant": "it",
                "quantization": True,
                "max_memory": {"0": "15GB", "1": "15GB"}
            }
        },
        "tasks": {
            "mmlu": {
                "type": "mmlu",
                "subset": "mathematics",
                "shot_count": 5,
                "temperature": 0.0
            },
            "gsm8k": {
                "type": "gsm8k",
                "shot_count": 5,
                "use_chain_of_thought": True
            },
            "efficiency": {
                "type": "efficiency",
                "sample_prompts": [
                    "Explain quantum computing",
                    "Write a Python function",
                    "Summarize machine learning"
                ],
                "output_lengths": [128, 256, 512]
            }
        },
        "evaluation": {
            "runs": 2,
            "batch_size": "auto",
            "statistical_tests": True
        },
        "output": {
            "path": "results",
            "visualize": True,
            "export_formats": ["yaml", "json"]
        },
        "hardware": {
            "device": "auto",
            "precision": "bfloat16",
            "quantization": True
        }
    }


def save_example_config(output_path: str) -> None:
    """Save an example configuration file."""
    config = create_example_config()
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# Convenience functions
def validate_config(config_dict: Dict[str, Any]) -> BenchmarkConfig:
    """Convenience function to validate a configuration dictionary."""
    validator = ConfigValidator()
    return validator.validate_config_dict(config_dict)


def validate_config_file(config_path: str) -> BenchmarkConfig:
    """Convenience function to validate a configuration file."""
    validator = ConfigValidator()
    return validator.validate_config_file(config_path)