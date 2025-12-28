"""
Model loading and management for the OpenEvalsing Suite.

Supports modern features including:
- Flash Attention 2 for faster inference
- SDPA (Scaled Dot-Product Attention) as fallback
- torch.compile() for PyTorch 2.0+ optimization
- BitsAndBytes quantization (4-bit and 8-bit)
"""

import logging
from typing import Any

import torch
from huggingface_hub.utils import HfHubHTTPError
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .interfaces import ModelInterface, ModelWrapper


# --- Custom Exception for Model Loading ---
class ModelLoadingError(Exception):
    """Custom exception for errors during model loading."""

    pass


def _check_flash_attention_available() -> bool:
    """Check if Flash Attention 2 is available."""
    try:
        import flash_attn  # noqa: F401

        return True
    except ImportError:
        return False


def _get_best_attention_implementation() -> str:
    """Get the best available attention implementation."""
    if _check_flash_attention_available():
        return "flash_attention_2"
    # SDPA is available in PyTorch 2.0+ and transformers 4.36+
    return "sdpa"


# --- Abstract Base Loader ---


class BaseModelLoader:
    """Abstract base class for model loaders with modern optimizations."""

    def __init__(self, model_name: str, config: dict[str, Any]):
        self.model_name = model_name
        self.config = config
        self.logger = logging.getLogger(
            f"openevals.model_loader.{self.__class__.__name__}"
        )
        self._use_flash_attention = False

    def _get_model_id(self) -> str:
        """Construct the HuggingFace model ID."""
        raise NotImplementedError

    def _get_quantization_config(self) -> BitsAndBytesConfig | None:
        """Get the quantization configuration if enabled."""
        if self.config.get("quantization", False):
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        return None

    def _get_attention_implementation(self) -> str | None:
        """Get the attention implementation to use."""
        if not self.config.get("use_flash_attention", True):
            return None

        attn_impl = _get_best_attention_implementation()
        if attn_impl == "flash_attention_2":
            self._use_flash_attention = True
            self.logger.info("Using Flash Attention 2 for faster inference")
        else:
            self.logger.info("Using SDPA (Scaled Dot-Product Attention)")
        return attn_impl

    def load(self) -> tuple[Any, Any]:
        """Load and return the model and tokenizer."""
        model_id = self._get_model_id()
        self.logger.info(f"Loading model: {model_id}")

        quant_config = self._get_quantization_config()
        torch_dtype = getattr(torch, self.config.get("torch_dtype", "bfloat16"))
        attn_implementation = self._get_attention_implementation()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=self.config.get("cache_dir")
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Build model kwargs
        model_kwargs: dict[str, Any] = {
            "quantization_config": quant_config,
            "torch_dtype": torch_dtype,
            "device_map": self.config.get("device_map", "auto"),
            "cache_dir": self.config.get("cache_dir"),
        }

        # Add attention implementation if available
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

        return model, tokenizer


# --- Specific Model Loaders ---


class GemmaLoader(BaseModelLoader):
    """Loader for Gemma models (supports Gemma 1 and Gemma 2)."""

    def _get_model_id(self) -> str:
        size = self.config["size"]
        variant = self.config.get("variant", "it")
        # Support both gemma and gemma-2 naming conventions
        size_clean = size.lower().replace("b", "")
        return f"google/gemma-{size_clean}-{variant}"


class MistralLoader(BaseModelLoader):
    """Loader for Mistral models."""

    def _get_model_id(self) -> str:
        size = self.config["size"]
        return f"mistralai/Mistral-{size.upper()}-v0.3"


class LlamaLoader(BaseModelLoader):
    """Loader for Llama models (supports Llama 2 and Llama 3)."""

    def _get_model_id(self) -> str:
        size = self.config["size"]
        version = self.config.get("version", "3.1")
        if version.startswith("3"):
            return f"meta-llama/Llama-{version}-{size}"
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
    """Manages the loading and retrieval of models with modern optimizations."""

    def __init__(self) -> None:
        self.logger = logging.getLogger("openevals.model_manager")
        self._loaders: dict[str, type[BaseModelLoader]] = {
            "gemma": GemmaLoader,
            "mistral": MistralLoader,
            "llama": LlamaLoader,
            "huggingface": HuggingFaceGenericLoader,
        }
        self._loaded_models: dict[str, ModelInterface] = {}

    def _get_loader_class(self, model_type: str) -> type[BaseModelLoader]:
        """Get the loader class for a given model type."""
        loader = self._loaders.get(model_type.lower())
        if not loader:
            raise ValueError(f"Unsupported model type: {model_type}")
        return loader

    def load_model(self, model_name: str, config: dict[str, Any]) -> ModelInterface:
        """
        Load a single model with robust error handling and modern optimizations.

        Args:
            model_name: The friendly name of the model from the config.
            config: The configuration dictionary for this model.

        Returns:
            A ModelWrapper instance with optional torch.compile and Flash Attention.

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

            # Create wrapped model with optional torch.compile
            use_compile = config.get("use_compile", False)
            use_flash_attention = loader._use_flash_attention

            wrapped_model = ModelWrapper(
                model,
                tokenizer,
                model_name,
                use_compile=use_compile,
                use_flash_attention=use_flash_attention,
            )
            self._loaded_models[model_name] = wrapped_model

            if wrapped_model._compiled:
                self.logger.info(f"Model '{model_name}' compiled with torch.compile()")

            return wrapped_model

        except torch.cuda.OutOfMemoryError as e:
            self.logger.error(
                f"GPU out of memory for model '{model_name}'. "
                "Try enabling quantization or using a smaller model."
            )
            raise ModelLoadingError(f"Out of Memory: {e}") from e

        except HfHubHTTPError as e:
            if e.response.status_code == 401:
                self.logger.error(
                    f"Authentication failed for model '{model_name}'. "
                    "Please check your HuggingFace token (HF_TOKEN)."
                )
                raise ModelLoadingError("HuggingFace authentication failed.") from e
            self.logger.error(
                f"HTTP error when downloading model '{model_name}': "
                f"{e.response.status_code}"
            )
            raise ModelLoadingError(f"HTTP Error: {e}") from e

        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred while loading model '{model_name}': {e}",
                exc_info=True,
            )
            raise ModelLoadingError(f"Unexpected error: {e}") from e

    def unload_model(self, model_name: str) -> None:
        """Unload a model and free memory."""
        if model_name in self._loaded_models:
            self._loaded_models[model_name].cleanup()
            del self._loaded_models[model_name]
            self.logger.info(f"Unloaded model '{model_name}'")

    def unload_all(self) -> None:
        """Unload all models and free memory."""
        for model_name in list(self._loaded_models.keys()):
            self.unload_model(model_name)


# --- Singleton Pattern for ModelManager ---

_model_manager_instance: ModelManager | None = None


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
