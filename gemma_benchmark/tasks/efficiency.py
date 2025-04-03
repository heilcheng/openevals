"""
Efficiency benchmarking for language models.
"""

import time
import logging
import psutil
import platform
from typing import Dict, List, Any, Optional

from ..core.model_loader import ModelWrapper

class EfficiencyBenchmark:
    """
    Benchmark for measuring model efficiency metrics like 
    latency, memory usage, and tokens per second.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the efficiency benchmark.
        
        Args:
            config: Configuration dictionary for the benchmark
        """
        self.logger = logging.getLogger("gemma_benchmark.tasks.efficiency")
        self.config = config
        self.sample_prompts = config.get("sample_prompts", [
            "Explain the theory of relativity",
            "Write a short story about a robot who discovers emotions",
            "Summarize the key events of World War II",
            "Describe the process of photosynthesis in plants",
        ])
        self.output_lengths = config.get("output_lengths", [128, 256, 512, 1024])
        
        # Detect hardware information
        self.system_info = self._get_system_info()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system hardware information."""
        info = {
            "os": platform.system(),
            "python_version": platform.python_version(),
            "cpu": platform.processor(),
            "cpu_count": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total": psutil.virtual_memory().total / (1024 ** 3),  # GB
        }
        
        # Try to detect GPU if available
        try:
            import torch
            info["torch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["cuda_version"] = torch.version.cuda
                info["gpu_count"] = torch.cuda.device_count()
                info["gpu_name"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                info["gpu_memory"] = [torch.cuda.get_device_properties(i).total_memory / (1024 ** 3) for i in range(torch.cuda.device_count())]
        except (ImportError, AttributeError):
            info["torch_available"] = False
        
        return info
    
    def evaluate(self, model: ModelWrapper) -> Dict[str, Any]:
        """
        Evaluate the model's efficiency metrics.
        
        Args:
            model: The model to evaluate
            
        Returns:
            Dictionary containing efficiency metrics
        """
        self.logger.info(f"Evaluating efficiency metrics for {model.model_name}")
        
        results = {
            "latency": {},
            "memory_usage": {},
            "tokens_per_second": {},
            "system_info": self.system_info
        }
        
        # Warm up the model
        self.logger.info("Warming up model...")
        _ = model.generate(self.sample_prompts[0], max_new_tokens=10)
        
        # Test latency at different output lengths
        for length in self.output_lengths:
            self.logger.info(f"Testing with output length: {length}")
            latencies = []
            memory_usages = []
            
            for prompt in self.sample_prompts:
                # Record memory before
                mem_before = psutil.Process().memory_info().rss / (1024 ** 3)  # GB
                
                # Record start time
                start_time = time.time()
                
                # Generate text
                _ = model.generate(prompt, max_new_tokens=length)
                
                # Record end time
                end_time = time.time()
                
                # Record memory after
                mem_after = psutil.Process().memory_info().rss / (1024 ** 3)  # GB
                memory_usage = mem_after - mem_before
                
                latency = end_time - start_time
                latencies.append(latency)
                memory_usages.append(memory_usage)
                
                self.logger.debug(f"Prompt: '{prompt[:20]}...', Latency: {latency:.4f}s, Memory: {memory_usage:.4f}GB")
            
            # Calculate average metrics
            avg_latency = sum(latencies) / len(latencies)
            avg_memory = sum(memory_usages) / len(memory_usages)
            tokens_per_sec = length / avg_latency
            
            results["latency"][f"tokens_{length}"] = avg_latency
            results["memory_usage"][f"tokens_{length}"] = avg_memory
            results["tokens_per_second"][f"tokens_{length}"] = tokens_per_sec
            
            self.logger.info(f"Length {length}: Avg Latency: {avg_latency:.4f}s, Avg Memory: {avg_memory:.4f}GB, Tokens/sec: {tokens_per_sec:.2f}")
        
        return results