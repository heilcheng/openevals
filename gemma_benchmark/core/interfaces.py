"""
Core interfaces and protocols to break circular dependencies.

This module defines the contracts between different components without
creating import dependencies, enabling clean separation of concerns.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Dict, Any, Optional, List, Union
from dataclasses import dataclass
import torch


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_new_tokens: int = 100
    temperature: float = 0.0
    do_sample: bool = False
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.0
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None


@dataclass
class ModelInfo:
    """Information about a loaded model."""

    name: str
    type: str
    size: Optional[str] = None
    variant: Optional[str] = None
    device: Optional[str] = None
    memory_usage: Optional[float] = None  # In GB
    parameters: Optional[int] = None


class ModelInterface(Protocol):
    """Protocol defining the interface that all model wrappers must implement."""

    @property
    def model_name(self) -> str:
        """Get the model name."""
        ...

    @property
    def model_info(self) -> ModelInfo:
        """Get model information."""
        ...

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        ...

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts."""
        ...

    def cleanup(self) -> None:
        """Clean up model resources."""
        ...

class ModelWrapper:
    """Concrete wrapper class for language models."""
    
    def __init__(self, model, tokenizer, model_name: str):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self._device = None
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        if hasattr(model, 'device'):
            self._device = model.device
        else:
            self._device = torch.device('cpu')
    
    @property
    def model_info(self) -> ModelInfo:
        info = ModelInfo(
            name=self.model_name,
            type=self.model.config.model_type if hasattr(self.model, 'config') else 'unknown',
            device=str(self._device),
        )
        return info
    
    def generate(self, prompt: str, max_new_tokens: int = 100, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        
        if self._device:
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        return [self.generate(prompt, **kwargs) for prompt in prompts]
    
    def cleanup(self) -> None:
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class TokenizerInterface(Protocol):
    """Protocol for tokenizer functionality."""

    def encode(self, text: str) -> List[int]:
        """Encode text to token ids."""
        ...

    def decode(self, token_ids: List[int]) -> str:
        """Decode token ids to text."""
        ...

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        ...


@dataclass
class BenchmarkResult:
    """Standard result format for all benchmarks."""

    overall: Dict[str, Any]
    config: Dict[str, Any]
    metadata: Dict[str, Any]
    details: Optional[Dict[str, Any]] = None
    errors: Optional[List[Dict[str, Any]]] = None


class BenchmarkInterface(Protocol):
    """Protocol defining the interface that all benchmarks must implement."""

    @property
    def name(self) -> str:
        """Get benchmark name."""
        ...

    @property
    def config(self) -> Dict[str, Any]:
        """Get benchmark configuration."""
        ...

    def load_data(self) -> Any:
        """Load benchmark dataset."""
        ...

    def evaluate(self, model: ModelInterface) -> BenchmarkResult:
        """Evaluate model on this benchmark."""
        ...


class ModelLoaderInterface(Protocol):
    """Protocol for model loading functionality."""

    def load_model(self, **kwargs) -> ModelInterface:
        """Load a model with given parameters."""
        ...

    def supports_model_type(self, model_type: str) -> bool:
        """Check if this loader supports the given model type."""
        ...


class AbstractModelWrapper(ABC):
    """Abstract base class for model wrappers with common functionality."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._device = None
        self._model_info = None

    @property
    def model_info(self) -> ModelInfo:
        """Get model information."""
        if self._model_info is None:
            self._model_info = self._create_model_info()
        return self._model_info

    @abstractmethod
    def _create_model_info(self) -> ModelInfo:
        """Create model information object."""
        pass

    @abstractmethod
    def _load_model_and_tokenizer(self, **kwargs):
        """Load the actual model and tokenizer."""
        pass

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt with error handling."""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(
                f"Model {self.model_name} not loaded. Call load() first."
            )

        try:
            return self._generate_impl(prompt, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Generation failed for {self.model_name}: {str(e)}")

    @abstractmethod
    def _generate_impl(self, prompt: str, **kwargs) -> str:
        """Implementation-specific generation logic."""
        pass

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts."""
        # Default implementation - can be overridden for batch optimization
        return [self.generate(prompt, **kwargs) for prompt in prompts]

    def cleanup(self) -> None:
        """Clean up model resources."""
        if hasattr(self, "_model") and self._model is not None:
            del self._model
        if hasattr(self, "_tokenizer") and self._tokenizer is not None:
            del self._tokenizer

        # Clear GPU memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        import gc

        gc.collect()


class AbstractBenchmark(ABC):
    """Abstract base class for benchmarks with common functionality."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__.replace("Benchmark", "").lower()
        self._data = None
        self._logger = None

    @property
    def logger(self):
        """Get logger instance."""
        if self._logger is None:
            import logging

            self._logger = logging.getLogger(f"gemma_benchmark.tasks.{self.name}")
        return self._logger

    @abstractmethod
    def load_data(self) -> Any:
        """Load benchmark dataset."""
        pass

    @abstractmethod
    def _evaluate_impl(self, model: ModelInterface) -> Dict[str, Any]:
        """Implementation-specific evaluation logic."""
        pass

    def evaluate(self, model: ModelInterface) -> BenchmarkResult:
        """Evaluate model with standardized error handling and result format."""
        if self._data is None:
            self.logger.info(f"Loading data for {self.name} benchmark...")
            self._data = self.load_data()

        self.logger.info(f"Starting {self.name} evaluation for {model.model_name}")

        try:
            # Run the implementation-specific evaluation
            raw_results = self._evaluate_impl(model)

            # Standardize the result format
            result = BenchmarkResult(
                overall=raw_results.get("overall", {}),
                config=self.config.copy(),
                metadata={
                    "benchmark_name": self.name,
                    "model_name": model.model_name,
                    "model_info": model.model_info.__dict__,
                },
                details=raw_results.get("details"),
                errors=raw_results.get("errors", []),
            )

            self.logger.info(f"Completed {self.name} evaluation for {model.model_name}")
            return result

        except Exception as e:
            self.logger.error(f"Evaluation failed for {self.name}: {str(e)}")
            # Return a failed result instead of crashing
            return BenchmarkResult(
                overall={"error": str(e), "success": False},
                config=self.config.copy(),
                metadata={
                    "benchmark_name": self.name,
                    "model_name": model.model_name,
                    "error_type": type(e).__name__,
                },
                errors=[{"type": type(e).__name__, "message": str(e)}],
            )


class ModelFactory:
    """Factory for creating model loaders without circular dependencies."""

    _loaders: Dict[str, type] = {}

    @classmethod
    def register_loader(cls, model_type: str, loader_class: type):
        """Register a model loader class for a specific type."""
        cls._loaders[model_type] = loader_class

    @classmethod
    def create_loader(cls, model_type: str) -> ModelLoaderInterface:
        """Create a model loader instance for the given type."""
        if model_type not in cls._loaders:
            raise ValueError(f"No loader registered for model type: {model_type}")

        loader_class = cls._loaders[model_type]
        return loader_class()

    @classmethod
    def get_supported_types(cls) -> List[str]:
        """Get list of supported model types."""
        return list(cls._loaders.keys())


class BenchmarkFactory:
    """Factory for creating benchmark instances without circular dependencies."""

    _benchmarks: Dict[str, type] = {}

    @classmethod
    def register_benchmark(cls, benchmark_type: str, benchmark_class: type):
        """Register a benchmark class for a specific type."""
        cls._benchmarks[benchmark_type] = benchmark_class

    @classmethod
    def create_benchmark(
        cls, benchmark_type: str, config: Dict[str, Any]
    ) -> BenchmarkInterface:
        """Create a benchmark instance for the given type."""
        if benchmark_type not in cls._benchmarks:
            raise ValueError(f"No benchmark registered for type: {benchmark_type}")

        benchmark_class = cls._benchmarks[benchmark_type]
        return benchmark_class(config)

    @classmethod
    def get_supported_types(cls) -> List[str]:
        """Get list of supported benchmark types."""
        return list(cls._benchmarks.keys())


# Exception classes for better error handling
class ModelLoadingError(Exception):
    """Raised when model loading fails."""

    pass


class BenchmarkExecutionError(Exception):
    """Raised when benchmark execution fails."""

    pass


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""

    pass


class AuthenticationError(Exception):
    """Raised when authentication fails."""

    pass


# Utility functions for common operations
def validate_generation_config(config: Dict[str, Any]) -> GenerationConfig:
    """Validate and create a GenerationConfig from a dictionary."""
    valid_keys = {
        "max_new_tokens",
        "temperature",
        "do_sample",
        "top_p",
        "top_k",
        "repetition_penalty",
        "pad_token_id",
        "eos_token_id",
    }

    # Filter out invalid keys
    filtered_config = {k: v for k, v in config.items() if k in valid_keys}

    # Validate ranges
    if "temperature" in filtered_config:
        temp = filtered_config["temperature"]
        if not 0.0 <= temp <= 2.0:
            raise ValueError(f"Temperature must be between 0.0 and 2.0, got {temp}")

    if "top_p" in filtered_config:
        top_p = filtered_config["top_p"]
        if not 0.0 <= top_p <= 1.0:
            raise ValueError(f"top_p must be between 0.0 and 1.0, got {top_p}")

    return GenerationConfig(**filtered_config)


def estimate_memory_usage(model_size: str) -> float:
    """Estimate memory usage in GB based on model size."""
    size_map = {
        "1b": 2.0,
        "2b": 4.0,
        "7b": 14.0,
        "9b": 18.0,
        "13b": 26.0,
        "27b": 54.0,
        "70b": 140.0,
    }

    size_lower = model_size.lower()
    if size_lower in size_map:
        return size_map[size_lower]

    # Try to extract number from size string
    import re

    match = re.search(r"(\d+)b", size_lower)
    if match:
        size_num = int(match.group(1))
        return size_num * 2.0  # Rough estimate: 2GB per billion parameters

    return 4.0  # Default fallback
