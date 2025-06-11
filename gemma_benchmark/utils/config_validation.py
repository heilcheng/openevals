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


# Enhanced enums with extensibility
class ModelType(str, Enum):
    """Supported model types."""
    GEMMA = "gemma"
    MISTRAL = "mistral" 
    LLAMA = "llama"
    HUGGINGFACE = "huggingface"
    
    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if a value is a valid model type."""
        return value.lower() in [item.value for item in cls]


class TaskType(str, Enum):
    """Supported task types."""
    MMLU = "mmlu"
    GSM8K = "gsm8k"
    HUMANEVAL = "humaneval"
    ARC = "arc"
    TRUTHFULQA = "truthfulqa"
    EFFICIENCY = "efficiency"
    
    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if a value is a valid task type."""
        return value.lower() in [item.value for item in cls]


# Model size configuration with flexible validation
class FlexibleModelConfig(BaseModel):
    """Enhanced model configuration with flexible validation."""
    
    type: str = Field(..., description="Model type")
    size: str = Field(..., description="Model size")
    variant: str = Field(default="it", description="Model variant")
    cache_dir: Optional[str] = Field(default=None, description="Model cache directory")
    quantization: bool = Field(default=True, description="Enable 4-bit quantization")
    device_map: Union[str, Dict[str, str]] = Field(default="auto", description="Device mapping strategy")
    max_memory: Optional[Dict[str, str]] = Field(default=None, description="Maximum memory per device")
    model_id: Optional[str] = Field(default=None, description="Custom model ID for HuggingFace models")
    torch_dtype: Optional[str] = Field(default="bfloat16", description="PyTorch data type")
    
    # Extended configuration for custom models
    custom_loader: Optional[str] = Field(default=None, description="Custom loader class name")
    loader_kwargs: Optional[Dict[str, Any]] = Field(default=None, description="Additional loader arguments")
    supported_sizes: Optional[List[str]] = Field(default=None, description="Override supported sizes")
    
    @validator('type')
    def validate_type_flexible(cls, v):
        """Flexible model type validation."""
        if not ModelType.is_valid(v):
            # Allow custom types but warn
            import warnings
            warnings.warn(
                f"Unknown model type '{v}'. Supported types: {[t.value for t in ModelType]}. "
                f"Using custom type - ensure appropriate loader is available."
            )
        return v.lower()
    
    @validator('size')
    def validate_size_flexible(cls, v):
        """Flexible model size validation."""
        v = str(v).lower()
        # Allow any reasonable size format
        import re
        if not re.match(r'^\d+(\.\d+)?[bkmg]?$', v):
            # Still validate basic format but be more permissive
            import warnings
            warnings.warn(
                f"Unusual model size format '{v}'. Expected formats: '2b', '7b', '13b', '2.7b', etc."
            )
        # Normalize to include 'b' suffix if it's just a number
        if re.match(r'^\d+(\.\d+)?$', v):
            v = v + 'b'
        return v
    
    @validator('variant')
    def validate_variant_flexible(cls, v):
        """Flexible variant validation."""
        # Extended list of common variants
        common_variants = [
            'it', 'instruct', 'chat', 'base', 'code', 'math', 'reasoning',
            'fine-tuned', 'rlhf', 'dpo', 'sft', 'unfiltered'
        ]
        
        if v.lower() not in common_variants:
            import warnings
            warnings.warn(
                f"Unknown model variant '{v}'. Common variants: {common_variants[:5]}... "
                f"Using anyway - ensure this is correct for your model."
            )
        return v.lower()
    
    @root_validator
    def validate_model_compatibility_flexible(cls, values):
        """Enhanced model compatibility validation."""
        model_type = values.get('type')
        size = values.get('size', '').lower()
        model_id = values.get('model_id')
        supported_sizes = values.get('supported_sizes')
        
        # Use custom supported sizes if provided
        if supported_sizes and size not in [s.lower() for s in supported_sizes]:
            raise ValueError(
                f"Model size '{size}' not in supported sizes {supported_sizes} for this configuration"
            )
        
        # Flexible Gemma validation
        if model_type == ModelType.GEMMA and not supported_sizes:
            # Default Gemma sizes but allow extensions
            default_gemma_sizes = ['2b', '7b', '9b', '27b']
            if size not in default_gemma_sizes:
                import warnings
                warnings.warn(
                    f"Size '{size}' not in default Gemma sizes {default_gemma_sizes}. "
                    f"This may be a new model or custom configuration."
                )
        
        # HuggingFace type handling
        if model_type == ModelType.HUGGINGFACE:
            if not model_id and not values.get('custom_loader'):
                raise ValueError(
                    "Either 'model_id' or 'custom_loader' is required when type is 'huggingface'"
                )
        
        return values

    class Config:
        extra = "allow"  # Allow additional custom fields
        use_enum_values = True


# Flexible task configuration
class FlexibleTaskConfig(BaseModel):
    """Flexible base configuration for benchmark tasks."""
    
    type: str = Field(..., description="Task type")
    
    # Common parameters that most tasks might use
    shot_count: Optional[int] = Field(default=None, ge=0, le=20, description="Few-shot examples")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="Sampling temperature")
    max_new_tokens: Optional[int] = Field(default=None, ge=1, le=4096, description="Maximum tokens to generate")
    
    # Custom task parameters
    custom_parameters: Optional[Dict[str, Any]] = Field(default=None, description="Custom task parameters")
    
    @validator('type')
    def validate_type_flexible(cls, v):
        """Flexible task type validation."""
        if not TaskType.is_valid(v):
            import warnings
            warnings.warn(
                f"Unknown task type '{v}'. Supported types: {[t.value for t in TaskType]}. "
                f"Using custom type - ensure appropriate task class is available."
            )
        return v.lower()
    
    class Config:
        extra = "allow"  # Allow task-specific fields
        use_enum_values = True


# Configuration registry for extensibility
class ConfigurationRegistry:
    """Registry for custom model loaders and task types."""
    
    _model_loaders: Dict[str, type] = {}
    _task_types: Dict[str, type] = {}
    _model_size_validators: Dict[str, callable] = {}
    
    @classmethod
    def register_model_loader(cls, model_type: str, loader_class: type):
        """Register a custom model loader."""
        cls._model_loaders[model_type.lower()] = loader_class
        logging.getLogger(__name__).info(f"Registered custom model loader for type: {model_type}")
    
    @classmethod
    def register_task_type(cls, task_type: str, task_class: type):
        """Register a custom task type."""
        cls._task_types[task_type.lower()] = task_class
        logging.getLogger(__name__).info(f"Registered custom task type: {task_type}")
    
    @classmethod
    def register_size_validator(cls, model_type: str, validator_func: callable):
        """Register a custom size validator for a model type."""
        cls._model_size_validators[model_type.lower()] = validator_func
        logging.getLogger(__name__).info(f"Registered size validator for model type: {model_type}")
    
    @classmethod
    def get_model_loader(cls, model_type: str) -> Optional[type]:
        """Get custom model loader if registered."""
        return cls._model_loaders.get(model_type.lower())
    
    @classmethod
    def get_task_class(cls, task_type: str) -> Optional[type]:
        """Get custom task class if registered."""
        return cls._task_types.get(task_type.lower())
    
    @classmethod
    def validate_model_size(cls, model_type: str, size: str) -> tuple[bool, Optional[str]]:
        """Validate model size using custom validator if available."""
        validator = cls._model_size_validators.get(model_type.lower())
        if validator:
            try:
                return validator(size), None
            except Exception as e:
                return False, str(e)
        return True, None  # No custom validator, allow any size


# Enhanced benchmark configuration
class FlexibleBenchmarkConfig(BaseModel):
    """Enhanced benchmark configuration with extensibility."""
    
    models: Dict[str, FlexibleModelConfig] = Field(..., description="Model configurations")
    tasks: Dict[str, FlexibleTaskConfig] = Field(..., description="Task configurations")
    
    # Standard sections (unchanged)
    evaluation: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Evaluation settings")
    output: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Output settings")
    hardware: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Hardware settings")
    
    # New extensibility features
    extensions: Optional[Dict[str, Any]] = Field(default=None, description="Custom extensions and plugins")
    custom_loaders: Optional[Dict[str, str]] = Field(default=None, description="Custom loader module paths")
    
    @validator('models')
    def validate_models_not_empty(cls, v):
        """Ensure at least one model is configured."""
        if not v or len(v) == 0:
            raise ValueError(
                "At least one model must be configured. "
                "Add a model configuration to the 'models' section."
            )
        return v
    
    @validator('tasks')
    def validate_tasks_not_empty(cls, v):
        """Ensure at least one task is configured."""
        if not v or len(v) == 0:
            raise ValueError(
                "At least one task must be configured. "
                "Add a task configuration to the 'tasks' section."
            )
        return v
    
    @root_validator
    def setup_extensions(cls, values):
        """Setup custom extensions and loaders."""
        extensions = values.get('extensions', {})
        custom_loaders = values.get('custom_loaders', {})
        
        # Load custom modules if specified
        for loader_name, module_path in (custom_loaders or {}).items():
            try:
                import importlib
                module = importlib.import_module(module_path)
                # Register any loaders or tasks found in the module
                if hasattr(module, 'register_components'):
                    module.register_components(ConfigurationRegistry)
                logging.getLogger(__name__).info(f"Loaded custom module: {module_path}")
            except Exception as e:
                logging.getLogger(__name__).warning(f"Failed to load custom module {module_path}: {e}")
        
        return values

    class Config:
        extra = "allow"  # Allow additional top-level fields for future extensions
        validate_assignment = True


class FlexibleConfigValidator:
    """Enhanced configuration validator with extensibility support."""
    
    def __init__(self):
        self.logger = logging.getLogger("gemma_benchmark.config_validator")
        
        if not PYDANTIC_AVAILABLE:
            raise ImportError(
                "Pydantic is required for configuration validation. "
                "Install with: pip install pydantic>=2.0.0"
            )
    
    def validate_config_dict(self, config_dict: Dict[str, Any]) -> FlexibleBenchmarkConfig:
        """
        Validate configuration with enhanced flexibility.
        """
        try:
            # Pre-process models
            if 'models' in config_dict:
                processed_models = {}
                for model_name, model_config in config_dict['models'].items():
                    processed_models[model_name] = FlexibleModelConfig(**model_config)
                config_dict['models'] = processed_models
            
            # Pre-process tasks
            if 'tasks' in config_dict:
                processed_tasks = {}
                for task_name, task_config in config_dict['tasks'].items():
                    processed_tasks[task_name] = FlexibleTaskConfig(**task_config)
                config_dict['tasks'] = processed_tasks
            
            # Validate complete configuration
            validated_config = FlexibleBenchmarkConfig(**config_dict)
            
            self.logger.info("Configuration validation successful with flexible schema")
            return validated_config
            
        except ValidationError as e:
            error_details = self._format_validation_errors(e)
            raise ConfigurationError(
                f"Configuration validation failed:\n{error_details}\n\n"
                f"Note: This validator supports custom model types and tasks. "
                f"Ensure custom components are properly registered."
            )
        except Exception as e:
            raise ConfigurationError(f"Configuration validation error: {str(e)}")
    
    def _format_validation_errors(self, validation_error: ValidationError) -> str:
        """Format validation errors with helpful suggestions."""
        error_messages = []
        
        for error in validation_error.errors():
            location = " -> ".join(str(loc) for loc in error['loc'])
            message = error['msg']
            error_type = error['type']
            
            if error_type == 'missing':
                friendly_msg = f"Missing required field: {location}"
            elif 'unknown' in message.lower() or 'unsupported' in message.lower():
                friendly_msg = f"Unknown value for {location}: {message}\n" \
                             f"  💡 Consider registering a custom component or check for typos"
            else:
                friendly_msg = f"Error in {location}: {message}"
            
            error_messages.append(f"  • {friendly_msg}")
        
        return "\n".join(error_messages)
    
    def validate_config_file(self, config_path: str) -> FlexibleBenchmarkConfig:
        """Validate configuration file with enhanced error reporting."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
                
            if config_dict is None:
                raise ConfigurationError("Configuration file is empty")
                
            return self.validate_config_dict(config_dict)
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML syntax in {config_path}: {str(e)}")


# Example custom components registration
def register_custom_gemma_sizes():
    """Example of how to register custom Gemma model sizes."""
    def validate_gemma_size(size: str) -> bool:
        # Extended Gemma sizes including hypothetical future models
        valid_sizes = ['2b', '7b', '9b', '27b', '65b', '180b', '405b']
        return size.lower() in valid_sizes
    
    ConfigurationRegistry.register_size_validator('gemma', validate_gemma_size)


def create_flexible_example_config() -> Dict[str, Any]:
    """Create an example configuration showcasing flexibility."""
    return {
        "models": {
            "gemma-2b": {
                "type": "gemma",
                "size": "2b",
                "variant": "it",
                "quantization": True
            },
            "custom-model": {
                "type": "custom-transformer",  # Custom type
                "size": "3.5b",  # Custom size
                "variant": "reasoning",  # Custom variant
                "model_id": "myorg/custom-model-3.5b",
                "custom_loader": "MyCustomLoader",
                "loader_kwargs": {
                    "special_param": "value"
                }
            }
        },
        "tasks": {
            "mmlu": {
                "type": "mmlu",
                "subset": "mathematics",
                "shot_count": 5
            },
            "custom-reasoning": {  # Custom task
                "type": "custom-reasoning",
                "difficulty": "hard",
                "custom_parameters": {
                    "reasoning_steps": 5,
                    "verification": True
                }
            }
        },
        "extensions": {
            "enable_custom_metrics": True,
            "custom_visualizations": ["heatmap", "treemap"]
        },
        "custom_loaders": {
            "custom_models": "myorg.custom_loaders"
        }
    }


# Convenience functions
def validate_config_flexible(config_dict: Dict[str, Any]) -> FlexibleBenchmarkConfig:
    """Convenience function for flexible validation."""
    validator = FlexibleConfigValidator()
    return validator.validate_config_dict(config_dict)


def validate_config_file_flexible(config_path: str) -> FlexibleBenchmarkConfig:
    """Convenience function for flexible file validation."""
    validator = FlexibleConfigValidator()
    return validator.validate_config_file(config_path)