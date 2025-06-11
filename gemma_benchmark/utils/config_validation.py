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
    # This will be a hard error if not available, so logging is sufficient.
    logging.getLogger(__name__).error(
        "Pydantic is required for this module. Please install with: pip install pydantic"
    )


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""

    pass


# --- Enums for Supported Types ---


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


# --- Pydantic Models for Configuration Schema ---


class FlexibleModelConfig(BaseModel):
    """Configuration for a single model."""

    type: str
    size: str
    variant: str = "it"
    cache_dir: Optional[str] = None
    quantization: bool = True
    device_map: Union[str, Dict[str, str]] = "auto"
    model_id: Optional[str] = None  # For HuggingFace models
    torch_dtype: Optional[str] = "bfloat16"

    @validator("type")
    def validate_type(cls, v):
        # Allow custom types but warn if not a standard one
        if v.lower() not in [m.value for m in ModelType]:
            logging.getLogger(__name__).warning(
                f"Custom model type '{v}' used. Ensure a corresponding loader is available."
            )
        return v.lower()

    @root_validator
    def check_huggingface_model_id(cls, values):
        model_type, model_id = values.get("type"), values.get("model_id")
        if model_type == ModelType.HUGGINGFACE and not model_id:
            raise ValueError("'model_id' is required for models of type 'huggingface'")
        return values

    class Config:
        extra = "allow"  # Allow other fields for custom loaders


class FlexibleTaskConfig(BaseModel):
    """Base configuration for a benchmark task."""

    type: str

    @validator("type")
    def validate_type(cls, v):
        # Allow custom task types
        if v.lower() not in [t.value for t in TaskType]:
            logging.getLogger(__name__).warning(
                f"Custom task type '{v}' used. Ensure a corresponding task class is registered."
            )
        return v.lower()

    class Config:
        extra = "allow"  # Allow task-specific fields


class FlexibleBenchmarkConfig(BaseModel):
    """Top-level configuration schema for the entire benchmark suite."""

    models: Dict[str, FlexibleModelConfig]
    tasks: Dict[str, FlexibleTaskConfig]
    evaluation: Dict[str, Any] = Field(default_factory=dict)
    output: Dict[str, Any] = Field(default_factory=dict)
    hardware: Dict[str, Any] = Field(default_factory=dict)

    @validator("models", "tasks")
    def check_not_empty(cls, v, field):
        if not v:
            raise ValueError(f"'{field.name}' section cannot be empty.")
        return v


# --- Main Validator Class ---


class FlexibleConfigValidator:
    """Validator for benchmark configuration files."""

    def __init__(self):
        self.logger = logging.getLogger("gemma_benchmark.config_validator")
        if not PYDANTIC_AVAILABLE:
            raise ImportError("Pydantic is required for configuration validation.")

    def validate_config_dict(
        self, config_dict: Dict[str, Any]
    ) -> FlexibleBenchmarkConfig:
        """Validate a configuration dictionary against the schema."""
        try:
            validated_config = FlexibleBenchmarkConfig(**config_dict)
            self.logger.info("Configuration validation successful.")
            return validated_config
        except ValidationError as e:
            error_details = self._format_validation_errors(e)
            raise ConfigurationError(
                f"Configuration validation failed:\n{error_details}"
            )
        except Exception as e:
            raise ConfigurationError(
                f"An unexpected configuration error occurred: {str(e)}"
            )

    def _format_validation_errors(self, validation_error: ValidationError) -> str:
        """Format Pydantic validation errors into a user-friendly string."""
        error_messages = []
        for error in validation_error.errors():
            location = " -> ".join(map(str, error["loc"]))
            message = error["msg"]
            error_messages.append(f"  - Error at '{location}': {message}")
        return "\n".join(error_messages)

    def validate_config_file(self, config_path: str) -> FlexibleBenchmarkConfig:
        """Load and validate a configuration file."""
        config_path_obj = Path(config_path)
        if not config_path_obj.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path_obj, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
            if not isinstance(config_dict, dict):
                raise ConfigurationError(
                    "Configuration file is not a valid dictionary."
                )
            return self.validate_config_dict(config_dict)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML syntax in {config_path}: {e}")
        except Exception as e:
            # Re-raise configuration errors directly
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(
                f"Failed to process config file {config_path}: {e}"
            )


# --- REVISED CONVENIENCE FUNCTIONS ---


def validate_config(config_dict: Dict[str, Any]) -> FlexibleBenchmarkConfig:
    """
    Convenience function for validating a configuration dictionary.
    """
    validator = FlexibleConfigValidator()
    return validator.validate_config_dict(config_dict)


def validate_config_file(config_path: str) -> FlexibleBenchmarkConfig:
    """
    Convenience function for validating a configuration file from a path.
    """
    validator = FlexibleConfigValidator()
    return validator.validate_config_file(config_path)
