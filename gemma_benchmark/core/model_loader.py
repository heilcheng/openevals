"""
Model loading utilities for Gemma and other models.
"""

import os
import logging
import torch
from typing import Dict, Any, Optional, List, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import HfApi
import warnings

# Suppress some common warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

class ModelWrapper:
    """Base wrapper class for language models."""
    
    def __init__(self, model_name: str, model=None, tokenizer=None):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(f"gemma_benchmark.model.{model_name}")
    
    def generate(self, prompt: str, max_new_tokens: int = 100, **kwargs) -> str:
        """Generate text based on a prompt."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model or tokenizer not loaded")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            
            # Move to GPU if available and model is on GPU
            if torch.cuda.is_available() and next(self.model.parameters()).device.type == 'cuda':
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Set default generation parameters
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "temperature": 1.0,
                "top_p": 1.0,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                **kwargs
            }
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs
                )
            
            # Decode only the new tokens
            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            return generated_text.strip()
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return f"ERROR: {str(e)}"


class GemmaLoader:
    """Loader for Gemma models with enhanced authentication and error handling."""
    
    def __init__(self):
        self.logger = logging.getLogger("gemma_benchmark.model_loader.gemma")
    
    def _check_authentication(self):
        """Check if user is authenticated with HuggingFace."""
        try:
            api = HfApi()
            user_info = api.whoami()
            self.logger.info(f"Authenticated as: {user_info['name']}")
            return True
        except Exception:
            self.logger.error("Not authenticated with HuggingFace. Please run authentication setup.")
            return False
    
    def _check_model_access(self, model_id: str) -> bool:
        """Check if user has access to the model."""
        try:
            api = HfApi()
            model_info = api.model_info(model_id)
            return True
        except Exception as e:
            self.logger.error(f"Cannot access model {model_id}: {e}")
            return False
    
    def load_model(self, 
                  size: str = "2b", 
                  variant: str = "it", 
                  cache_dir: Optional[str] = None,
                  quantization: bool = True,
                  device_map: Optional[Union[str, Dict]] = "auto",
                  max_memory: Optional[Dict[str, str]] = None) -> ModelWrapper:
        """
        Load a Gemma model from HuggingFace with enhanced error handling.
        
        Args:
            size: Model size ("2b", "9b", "27b")
            variant: Model variant ("it" for instruction-tuned)
            cache_dir: Directory to cache model weights
            quantization: Whether to use 4-bit quantization
            device_map: Device mapping strategy
            max_memory: Maximum memory per device
        """
        # Check authentication first
        if not self._check_authentication():
            raise RuntimeError("HuggingFace authentication required. Please set up authentication.")
        
        # Map size to actual model names
        model_mapping = {
            "2b": "google/gemma-2-2b-it",
            "9b": "google/gemma-2-9b-it", 
            "27b": "google/gemma-2-27b-it"
        }
        
        if size not in model_mapping:
            raise ValueError(f"Unsupported Gemma size: {size}. Available: {list(model_mapping.keys())}")
        
        model_id = model_mapping[size]
        self.logger.info(f"Loading {model_id}")
        
        # Check model access
        if not self._check_model_access(model_id):
            raise RuntimeError(f"Cannot access {model_id}. Please check permissions and license acceptance.")
        
        try:
            # Setup quantization if requested
            quantization_config = None
            if quantization:
                self.logger.info("Using 4-bit quantization")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
            
            # Load tokenizer
            self.logger.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, 
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            
            # Ensure pad token is set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Prepare model loading arguments
            model_kwargs = {
                "device_map": device_map,
                "torch_dtype": torch.bfloat16 if not quantization else None,
                "cache_dir": cache_dir,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            
            if max_memory:
                model_kwargs["max_memory"] = max_memory
            
            # Try to use flash attention if available
            try:
                if torch.cuda.is_available():
                    model_kwargs["attn_implementation"] = "flash_attention_2"
            except Exception:
                self.logger.warning("Flash attention not available, using default attention")
            
            # Load model
            self.logger.info("Loading model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **model_kwargs
            )
            
            # Put model in eval mode
            model.eval()
            
            model_wrapper = ModelWrapper(f"gemma-{size}-{variant}", model, tokenizer)
            self.logger.info(f"Successfully loaded {model_id}")
            
            # Log model info
            if hasattr(model, 'config'):
                param_count = sum(p.numel() for p in model.parameters())
                self.logger.info(f"Model has {param_count:,} parameters")
            
            return model_wrapper
            
        except Exception as e:
            self.logger.error(f"Failed to load Gemma model: {e}")
            self.logger.info("Troubleshooting tips:")
            self.logger.info("1. Ensure you have accepted the Gemma license on HuggingFace")
            self.logger.info("2. Check your HuggingFace token has the correct permissions")
            self.logger.info("3. Verify you have sufficient GPU memory for the model")
            raise


class MistralLoader:
    """Loader for Mistral models with enhanced error handling."""
    
    def __init__(self):
        self.logger = logging.getLogger("gemma_benchmark.model_loader.mistral")
    
    def load_model(self, 
                  size: str = "7b", 
                  variant: str = "instruct", 
                  cache_dir: Optional[str] = None,
                  quantization: bool = True,
                  device_map: Optional[Union[str, Dict]] = "auto",
                  max_memory: Optional[Dict[str, str]] = None) -> ModelWrapper:
        """Load a Mistral model with enhanced error handling."""
        
        # Map size to model names
        size_mapping = {
            "7b": "mistralai/Mistral-7B-Instruct-v0.3"
        }
        
        if size not in size_mapping:
            raise ValueError(f"Unsupported Mistral size: {size}. Available: {list(size_mapping.keys())}")
        
        model_id = size_mapping[size]
        self.logger.info(f"Loading {model_id}")
        
        try:
            quantization_config = None
            if quantization:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, 
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model_kwargs = {
                "device_map": device_map,
                "torch_dtype": torch.bfloat16 if not quantization else None,
                "cache_dir": cache_dir,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            
            if max_memory:
                model_kwargs["max_memory"] = max_memory
            
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **model_kwargs
            )
            
            model.eval()
            
            model_wrapper = ModelWrapper(f"mistral-{size}-{variant}", model, tokenizer)
            self.logger.info(f"Successfully loaded {model_id}")
            return model_wrapper
            
        except Exception as e:
            self.logger.error(f"Failed to load Mistral model: {e}")
            raise


class HuggingFaceLoader:
    """Generic loader for any HuggingFace model."""
    
    def __init__(self):
        self.logger = logging.getLogger("gemma_benchmark.model_loader.huggingface")
    
    def load_model(self,
                  model_id: str,
                  tokenizer_id: Optional[str] = None,
                  cache_dir: Optional[str] = None,
                  quantization: bool = True,
                  device_map: Optional[Union[str, Dict]] = "auto",
                  max_memory: Optional[Dict[str, str]] = None) -> ModelWrapper:
        """Load any HuggingFace model."""
        
        tokenizer_id = tokenizer_id or model_id
        self.logger.info(f"Loading model: {model_id}, tokenizer: {tokenizer_id}")
        
        try:
            quantization_config = None
            if quantization:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
            
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_id,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model_kwargs = {
                "device_map": device_map,
                "torch_dtype": torch.bfloat16 if not quantization else None,
                "cache_dir": cache_dir,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            
            if max_memory:
                model_kwargs["max_memory"] = max_memory
            
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **model_kwargs
            )
            
            model.eval()
            
            # Extract model name from ID
            model_name = model_id.split('/')[-1]
            model_wrapper = ModelWrapper(model_name, model, tokenizer)
            self.logger.info(f"Successfully loaded {model_id}")
            return model_wrapper
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_id}: {e}")
            raise