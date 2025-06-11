"""
Model loading and management for the Gemma Benchmarking Suite.
"""

import logging
import torch
from typing import Dict, Any, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub.utils import HfHubHTTPError

# Core interfaces for model wrapping
# FIX: Added ModelInterface to the import statement
from .interfaces import ModelWrapper, ModelInterface


# --- Custom Exception for Model Loading ---
class ModelLoadingError(Exception):
    """Custom exception for errors during model loading."""

    pass


# --- Abstract Base Loader ---


class BaseModelLoader:
    """Abstract base class for model loaders."""



    def __init__(self, model_name: str, config: Dict[str, Any]):
        self.model_name = model_name
        self.config = config
        self.logger = logging.getLogger(
            f"gemma_benchmark.model_loader.{self.__class__.__name__}"
        )

    def _get_model_id(self) -> str:
        """Construct the HuggingFace model ID."""
        raise NotImplementedError

    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get the quantization configuration if enabled."""
        if self.config.get("quantization", False):
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        return None

    def load(self) -> Tuple[Any, Any]:
        """Load and return the model and tokenizer."""
        model_id = self._get_model_id()
        self.logger.info(f"Loading model: {model_id}")

        quant_config = self._get_quantization_config()
        torch_dtype = getattr(torch, self.config.get("torch_dtype", "bfloat16"))

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=self.config.get("cache_dir")
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            torch_dtype=torch_dtype,
            device_map=self.config.get("device_map", "auto"),
            cache_dir=self.config.get("cache_dir"),
        )
        return model, tokenizer


# --- Specific Model Loaders ---


class GemmaLoader(BaseModelLoader):
    """Loader for Gemma models."""

    def _get_model_id(self) -> str:
        size = self.config["size"]
        variant = self.config.get("variant", "it")
        return f"google/gemma-{size.replace('b', '')}-{variant}"


class MistralLoader(BaseModelLoader):
    """Loader for Mistral models."""

    def _get_model_id(self) -> str:
        size = self.config["size"]
        return f"mistralai/Mistral-{size.upper()}-v0.1"


class LlamaLoader(BaseModelLoader):
    """Loader for Llama models."""

    def _get_model_id(self) -> str:
        size = self.config["size"]
        return f"meta-llama/Llama-2-{size}-hf"


class HuggingFaceGenericLoader(BaseModelLoader):
    """Generic loader for any HuggingFace model."""

    def _get_model_id(self) -> str:
        model_id = self.config.get("model_id")
        if not model_id:
            raise ValueError(
                "model_id must be specified for 'huggingface' type models."
            )
        return model_id


# --- Model Manager ---


class ModelManager:
    """Manages the loading and retrieval of models."""

    def __init__(self):
        self.logger = logging.getLogger("gemma_benchmark.model_manager")
        self._loaders = {
            "gemma": GemmaLoader,
            "mistral": MistralLoader,
            "llama": LlamaLoader,
            "huggingface": HuggingFaceGenericLoader,
        }
        self._loaded_models: Dict[str, ModelInterface] = {}

    def _get_loader_class(self, model_type: str):
        """Get the loader class for a given model type."""
        loader = self._loaders.get(model_type.lower())
        if not loader:
            raise ValueError(f"Unsupported model type: {model_type}")
        return loader

    def load_model(self, model_name: str, config: Dict[str, Any]) -> ModelInterface:
        """
        Load a single model with robust error handling.

        Args:
            model_name: The friendly name of the model from the config.
            config: The configuration dictionary for this model.

        Returns:
            A ModelWrapper instance.

        Raises:
            ModelLoadingError: If the model fails to load for a known reason.
        """
        if model_name in self._loaded_models:
            self.logger.info(f"Model '{model_name}' is already loaded.")
            return self._loaded_models[model_name]

        self.logger.info(f"Attempting to load model '{model_name}'...")
        try:
            loader_class = self._get_loader_class(config["type"])
            loader = loader_class(model_name, config)

            model, tokenizer = loader.load()

            # Verify model is loaded
            if model is None or tokenizer is None:
                raise ModelLoadingError("Loader returned None for model or tokenizer.")

            self.logger.info(
                f"Successfully loaded model and tokenizer for '{model_name}'."
            )
            wrapped_model = ModelWrapper(model, tokenizer, model_name)
            self._loaded_models[model_name] = wrapped_model
            return wrapped_model

        except torch.cuda.OutOfMemoryError as e:
            self.logger.error(
                f"GPU out of memory for model '{model_name}'. Try enabling quantization or using a smaller model."
            )
            raise ModelLoadingError(f"Out of Memory: {e}") from e

        except HfHubHTTPError as e:
            if e.response.status_code == 401:
                self.logger.error(
                    f"Authentication failed for model '{model_name}'. Please check your HuggingFace token (HF_TOKEN)."
                )
                raise ModelLoadingError("HuggingFace authentication failed.") from e
            else:
                self.logger.error(
                    f"HTTP error when downloading model '{model_name}': {e.response.status_code}"
                )
                raise ModelLoadingError(f"HTTP Error: {e}") from e

        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred while loading model '{model_name}': {e}",
                exc_info=True,
            )
            raise ModelLoadingError(f"Unexpected error: {e}") from e


# --- Singleton Pattern for ModelManager ---

_model_manager_instance: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """
    Get the singleton instance of the ModelManager.

    This ensures that all parts of the application use the same
    model manager, preventing models from being loaded multiple times.
    """
    global _model_manager_instance
    if _model_manager_instance is None:
        _model_manager_instance = ModelManager()
    return _model_manager_instance