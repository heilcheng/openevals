"""
Fixed model loading utilities without circular dependencies.

This module provides model loading functionality using dependency injection
and interface-based design to prevent circular import issues.
"""

import os
import logging
import torch
from typing import Dict, Any, Optional, List, Union
import warnings
from pathlib import Path

# Import interfaces instead of concrete implementations
from .interfaces import (
    ModelInterface, 
    AbstractModelWrapper, 
    ModelLoaderInterface,
    ModelFactory,
    ModelInfo,
    GenerationConfig,
    validate_generation_config,
    estimate_memory_usage,
    ModelLoadingError
)

# Core ML imports
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from huggingface_hub import HfApi
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.getLogger(__name__).error("transformers not available")

# Suppress common warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


class HuggingFaceModelWrapper(AbstractModelWrapper):
    """
    Model wrapper for HuggingFace transformers without circular dependencies.
    """
    
    def __init__(self, model_name: str, model_type: str, **kwargs):
        super().__init__(model_name)
        self.model_type = model_type
        self.logger = logging.getLogger(f"gemma_benchmark.model.{model_name}")
        self.load_kwargs = kwargs
        
        # Load the model immediately
        self._load_model_and_tokenizer(**kwargs)
    
    def _create_model_info(self) -> ModelInfo:
        """Create model information object."""
        # Extract size from model name if possible
        size = self._extract_size_from_name(self.model_name)
        
        # Get device info
        device = None
        if self._model is not None:
            try:
                device = str(next(self._model.parameters()).device)
            except:
                device = "unknown"
        
        # Estimate memory usage
        memory_usage = estimate_memory_usage(size) if size else None
        
        return ModelInfo(
            name=self.model_name,
            type=self.model_type,
            size=size,
            device=device,
            memory_usage=memory_usage
        )
    
    def _extract_size_from_name(self, name: str) -> Optional[str]:
        """Extract model size from name."""
        import re
        # Look for patterns like "2b", "7b", "13b" etc.
        match = re.search(r'(\d+[bm])', name.lower())
        return match.group(1) if match else None
    
    def _load_model_and_tokenizer(self, **kwargs):
        """Load the model and tokenizer from HuggingFace."""
        if not TRANSFORMERS_AVAILABLE:
            raise ModelLoadingError("transformers library not available")
        
        try:
            # Extract loading parameters
            cache_dir = kwargs.get('cache_dir')
            quantization = kwargs.get('quantization', True)
            device_map = kwargs.get('device_map', 'auto')
            max_memory = kwargs.get('max_memory')
            torch_dtype = kwargs.get('torch_dtype', torch.bfloat16)
            
            self.logger.info(f"Loading model: {self.model_name}")
            
            # Setup quantization config if requested
            quantization_config = None
            if quantization:
                self.logger.info("Using 4-bit quantization")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True,
                )
            
            # Load tokenizer first
            self.logger.info("Loading tokenizer...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            
            # Ensure pad token is set
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            # Prepare model loading arguments
            model_kwargs = {
                "device_map": device_map,
                "cache_dir": cache_dir,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            
            # Add dtype if not using quantization
            if not quantization:
                model_kwargs["torch_dtype"] = torch_dtype
            
            # Add quantization config if using it
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            
            # Add memory limits if specified
            if max_memory:
                model_kwargs["max_memory"] = max_memory
            
            # Try to use optimized attention
            try:
                if torch.cuda.is_available():
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    self.logger.info("Using Flash Attention 2")
            except Exception:
                self.logger.debug("Flash attention not available, using default")
            
            # Load the model
            self.logger.info("Loading model weights...")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Set to eval mode
            self._model.eval()
            
            # Store device info
            try:
                self._device = str(next(self._model.parameters()).device)
            except:
                self._device = "unknown"
            
            self.logger.info(f"Successfully loaded {self.model_name} on {self._device}")
            
            # Log memory info if available
            if torch.cuda.is_available() and 'cuda' in self._device:
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                self.logger.info(f"GPU memory allocated: {memory_allocated:.2f} GB")
            
        except Exception as e:
            self.logger.error(f"Failed to load model {self.model_name}: {e}")
            raise ModelLoadingError(f"Model loading failed: {str(e)}")
    
    def _generate_impl(self, prompt: str, **kwargs) -> str:
        """Implementation-specific generation logic."""
        try:
            # Validate and setup generation config
            generation_config = self._prepare_generation_config(**kwargs)
            
            # Tokenize input
            inputs = self._tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048
            )
            
            # Move inputs to same device as model
            if hasattr(self._model, 'device'):
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            elif torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    **generation_config.__dict__
                )
            
            # Decode only the new tokens
            input_length = inputs['input_ids'].shape[1]
            new_tokens = outputs[0][input_length:]
            generated_text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            return generated_text.strip()
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return f"ERROR: {str(e)}"
    
    def _prepare_generation_config(self, **kwargs) -> GenerationConfig:
        """Prepare generation configuration with defaults."""
        # Set defaults for this model type
        defaults = {
            'max_new_tokens': 100,
            'temperature': 0.0,
            'do_sample': False,
            'pad_token_id': self._tokenizer.eos_token_id,
            'eos_token_id': self._tokenizer.eos_token_id,
        }
        
        # Override with provided kwargs
        config_dict = {**defaults, **kwargs}
        
        return validate_generation_config(config_dict)
    
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Optimized batch generation."""
        if len(prompts) == 1:
            return [self.generate(prompts[0], **kwargs)]
        
        try:
            generation_config = self._prepare_generation_config(**kwargs)
            
            # Tokenize all prompts
            inputs = self._tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            
            # Move to device
            if hasattr(self._model, 'device'):
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            elif torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    **generation_config.__dict__
                )
            
            # Decode results
            results = []
            input_length = inputs['input_ids'].shape[1]
            
            for i, output in enumerate(outputs):
                new_tokens = output[input_length:]
                generated_text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
                results.append(generated_text.strip())
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch generation failed: {e}")
            # Fallback to individual generation
            return [self.generate(prompt, **kwargs) for prompt in prompts]


class GemmaLoader:
    """Loader for Gemma models without circular dependencies."""
    
    def __init__(self):
        self.logger = logging.getLogger("gemma_benchmark.model_loader.gemma")
        self.supported_sizes = ["2b", "9b", "27b"]
        self.supported_variants = ["it"]  # instruction-tuned
    
    def supports_model_type(self, model_type: str) -> bool:
        """Check if this loader supports the given model type."""
        return model_type.lower() == "gemma"
    
    def load_model(self, **kwargs) -> ModelInterface:
        """Load a Gemma model."""
        # Extract Gemma-specific parameters
        size = kwargs.get('size', '2b')
        variant = kwargs.get('variant', 'it')
        
        # Validate parameters
        if size not in self.supported_sizes:
            raise ModelLoadingError(f"Unsupported Gemma size: {size}. Supported: {self.supported_sizes}")
        
        if variant not in self.supported_variants:
            raise ModelLoadingError(f"Unsupported Gemma variant: {variant}. Supported: {self.supported_variants}")
        
        # Map to actual model ID
        model_id = f"google/gemma-2-{size}-{variant}"
        
        self.logger.info(f"Loading Gemma model: {model_id}")
        
        # Check authentication and access
        self._verify_model_access(model_id)
        
        # Create wrapper with Gemma-specific settings
        wrapper = HuggingFaceModelWrapper(
            model_name=model_id,
            model_type="gemma",
            **kwargs
        )
        
        return wrapper
    
    def _verify_model_access(self, model_id: str):
        """Verify access to Gemma model."""
        try:
            from ..auth import get_auth_manager
            auth_manager = get_auth_manager()
            
            access_result = auth_manager.check_model_access(model_id)
            if not access_result.has_access:
                error_msg = f"Cannot access {model_id}: {access_result.error_message}"
                if access_result.suggestions:
                    error_msg += f"\nSuggestions: {'; '.join(access_result.suggestions[:2])}"
                raise ModelLoadingError(error_msg)
                
        except ImportError:
            self.logger.warning("Authentication module not available, skipping access check")


class MistralLoader:
    """Loader for Mistral models without circular dependencies."""
    
    def __init__(self):
        self.logger = logging.getLogger("gemma_benchmark.model_loader.mistral")
        self.model_map = {
            "7b": "mistralai/Mistral-7B-Instruct-v0.3"
        }
    
    def supports_model_type(self, model_type: str) -> bool:
        """Check if this loader supports the given model type."""
        return model_type.lower() == "mistral"
    
    def load_model(self, **kwargs) -> ModelInterface:
        """Load a Mistral model."""
        size = kwargs.get('size', '7b')
        
        if size not in self.model_map:
            raise ModelLoadingError(f"Unsupported Mistral size: {size}. Supported: {list(self.model_map.keys())}")
        
        model_id = self.model_map[size]
        self.logger.info(f"Loading Mistral model: {model_id}")
        
        wrapper = HuggingFaceModelWrapper(
            model_name=model_id,
            model_type="mistral",
            **kwargs
        )
        
        return wrapper


class HuggingFaceGenericLoader:
    """Generic loader for any HuggingFace model without circular dependencies."""
    
    def __init__(self):
        self.logger = logging.getLogger("gemma_benchmark.model_loader.generic")
    
    def supports_model_type(self, model_type: str) -> bool:
        """Check if this loader supports the given model type."""
        return model_type.lower() == "huggingface"
    
    def load_model(self, **kwargs) -> ModelInterface:
        """Load any HuggingFace model."""
        model_id = kwargs.get('model_id')
        if not model_id:
            raise ModelLoadingError("model_id is required for generic HuggingFace loader")
        
        self.logger.info(f"Loading HuggingFace model: {model_id}")
        
        # Extract model name for wrapper
        model_name = model_id.split('/')[-1]
        
        wrapper = HuggingFaceModelWrapper(
            model_name=model_id,
            model_type="huggingface",
            **kwargs
        )
        
        return wrapper


class ModelManager:
    """
    High-level model management without circular dependencies.
    
    This class orchestrates model loading using the factory pattern
    and provides a clean interface for the benchmark system.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("gemma_benchmark.model_manager")
        self.loaded_models: Dict[str, ModelInterface] = {}
        self._register_default_loaders()
    
    def _register_default_loaders(self):
        """Register default model loaders."""
        ModelFactory.register_loader("gemma", GemmaLoader)
        ModelFactory.register_loader("mistral", MistralLoader)
        ModelFactory.register_loader("huggingface", HuggingFaceGenericLoader)
        
        self.logger.info(f"Registered loaders for: {ModelFactory.get_supported_types()}")
    
    def load_model(self, model_name: str, model_config: Dict[str, Any]) -> ModelInterface:
        """
        Load a model based on configuration.
        
        Args:
            model_name: Unique name for this model instance
            model_config: Configuration dictionary with 'type' and other parameters
            
        Returns:
            ModelInterface instance
        """
        model_type = model_config.get('type')
        if not model_type:
            raise ModelLoadingError(f"Model type not specified for {model_name}")
        
        if model_name in self.loaded_models:
            self.logger.info(f"Model {model_name} already loaded, returning cached instance")
            return self.loaded_models[model_name]
        
        try:
            # Create loader for this model type
            loader = ModelFactory.create_loader(model_type)
            
            # Load the model
            self.logger.info(f"Loading {model_type} model: {model_name}")
            model = loader.load_model(**model_config)
            
            # Cache the loaded model
            self.loaded_models[model_name] = model
            
            self.logger.info(f"Successfully loaded and cached model: {model_name}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            raise ModelLoadingError(f"Model loading failed for {model_name}: {str(e)}")
    
    def get_model(self, model_name: str) -> Optional[ModelInterface]:
        """Get a previously loaded model."""
        return self.loaded_models.get(model_name)
    
    def cleanup_model(self, model_name: str) -> bool:
        """Clean up a specific model."""
        if model_name in self.loaded_models:
            model = self.loaded_models[model_name]
            model.cleanup()
            del self.loaded_models[model_name]
            self.logger.info(f"Cleaned up model: {model_name}")
            return True
        return False
    
    def cleanup_all_models(self):
        """Clean up all loaded models."""
        model_names = list(self.loaded_models.keys())
        for model_name in model_names:
            self.cleanup_model(model_name)
        self.logger.info("Cleaned up all models")
    
    def get_loaded_models(self) -> List[str]:
        """Get list of loaded model names."""
        return list(self.loaded_models.keys())
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a loaded model."""
        model = self.get_model(model_name)
        return model.model_info if model else None


# Create a global model manager instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


# Convenience functions for backward compatibility
def load_model(model_name: str, model_config: Dict[str, Any]) -> ModelInterface:
    """Load a model using the global manager."""
    manager = get_model_manager()
    return manager.load_model(model_name, model_config)


def cleanup_models():
    """Clean up all loaded models."""
    manager = get_model_manager()
    manager.cleanup_all_models()