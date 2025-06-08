"""
Model loading utilities for Gemma and other models.
"""

import os
import logging
import torch
from typing import Dict, Any, Optional, List, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

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
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode only the new tokens
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return generated_text.strip()

class GemmaLoader:
    """Loader for Gemma models."""
    
    def __init__(self):
        self.logger = logging.getLogger("gemma_benchmark.model_loader.gemma")
    
    def load_model(self, 
                  size: str = "2b", 
                  variant: str = "it", 
                  cache_dir: Optional[str] = None,
                  quantization: bool = True) -> ModelWrapper:
        """
        Load a Gemma model from HuggingFace.
        
        Args:
            size: Model size ("2b", "9b", "27b")
            variant: Model variant ("it" for instruction-tuned)
            cache_dir: Directory to cache model weights
            quantization: Whether to use 4-bit quantization
        """
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
        
        try:
            # Setup quantization if requested
            quantization_config = None
            if quantization:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, 
                cache_dir=cache_dir
            )
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16 if not quantization else None,
                cache_dir=cache_dir,
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
            )
            
            model_wrapper = ModelWrapper(f"gemma-{size}-{variant}", model, tokenizer)
            self.logger.info(f"Successfully loaded {model_id}")
            return model_wrapper
            
        except Exception as e:
            self.logger.error(f"Failed to load Gemma model: {e}")
            raise

class MistralLoader:
    """Loader for Mistral models."""
    
    def load_model(self, 
                  size: str = "7b", 
                  variant: str = "instruct", 
                  cache_dir: Optional[str] = None,
                  quantization: bool = True) -> ModelWrapper:
        """Load a Mistral model."""
        model_id = f"mistralai/Mistral-{size.upper()}-Instruct-v0.3"
        
        self.logger = logging.getLogger("gemma_benchmark.model_loader.mistral")
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
            
            tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto", 
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16 if not quantization else None,
                cache_dir=cache_dir
            )
            
            return ModelWrapper(f"mistral-{size}-{variant}", model, tokenizer)
            
        except Exception as e:
            self.logger.error(f"Failed to load Mistral model: {e}")
            raise