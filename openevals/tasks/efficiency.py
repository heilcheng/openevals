"""
Efficiency benchmarking for language models with enhanced system compatibility.
"""

import gc
import logging
import platform
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "psutil not available - memory monitoring will be limited"
    )

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "PyTorch not available - GPU monitoring will be disabled"
    )

try:
    import nvidia_ml_py3 as nvml

    nvml.nvmlInit()
    NVML_AVAILABLE = True
except (ImportError, Exception):
    NVML_AVAILABLE = False

from ..core.model_loader import ModelWrapper


@dataclass
class EfficiencyMetrics:
    """Container for efficiency metrics."""

    latency: float
    tokens_per_second: float
    memory_usage_mb: float
    gpu_memory_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    error: Optional[str] = None


class SystemMonitor:
    """Cross-platform system monitoring utilities."""

    def __init__(self):
        self.logger = logging.getLogger("openevals.efficiency.monitor")

    def get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                return memory_info.rss / (1024**3)
            except Exception as e:
                self.logger.warning(f"Failed to get memory usage with psutil: {e}")

        # Fallback implementation
        try:
            import resource

            ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # On Linux, ru_maxrss is in KB; on macOS, it's in bytes
            import platform

            if platform.system() == "Darwin":  # macOS
                return ru_maxrss / (1024**3)
            else:  # Linux
                return ru_maxrss / (1024**2)
        except Exception as e:
            self.logger.warning(f"Failed to get memory usage with resource: {e}")
            return 0.0

    def get_cpu_usage(self) -> Optional[float]:
        """Get current CPU usage percentage."""
        if PSUTIL_AVAILABLE:
            try:
                return psutil.cpu_percent(interval=0.1)
            except Exception as e:
                self.logger.warning(f"Failed to get CPU usage: {e}")
        return None

    def get_gpu_memory_usage(self) -> Optional[float]:
        """Get GPU memory usage in GB."""
        if not TORCH_AVAILABLE:
            return None

        try:
            if torch.cuda.is_available():
                # PyTorch method
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                return memory_allocated
        except Exception as e:
            self.logger.warning(f"Failed to get GPU memory with PyTorch: {e}")

        # NVML method as fallback
        if NVML_AVAILABLE:
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(0)
                info = nvml.nvmlDeviceGetMemoryInfo(handle)
                return info.used / (1024**3)
            except Exception as e:
                self.logger.warning(f"Failed to get GPU memory with NVML: {e}")

        return None

    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        info = {
            "os": platform.system(),
            "os_version": platform.release(),
            "python_version": platform.python_version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
        }

        # CPU information
        if PSUTIL_AVAILABLE:
            try:
                info.update(
                    {
                        "cpu_count_physical": psutil.cpu_count(logical=False),
                        "cpu_count_logical": psutil.cpu_count(logical=True),
                        "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                        "memory_available_gb": psutil.virtual_memory().available
                        / (1024**3),
                    }
                )
            except Exception as e:
                self.logger.warning(f"Failed to get detailed system info: {e}")

        # PyTorch and GPU information
        if TORCH_AVAILABLE:
            try:
                info.update(
                    {
                        "torch_version": torch.__version__,
                        "cuda_available": torch.cuda.is_available(),
                    }
                )

                if torch.cuda.is_available():
                    info.update(
                        {
                            "cuda_version": torch.version.cuda,
                            "cudnn_version": torch.backends.cudnn.version(),
                            "gpu_count": torch.cuda.device_count(),
                            "gpu_names": [
                                torch.cuda.get_device_name(i)
                                for i in range(torch.cuda.device_count())
                            ],
                            "gpu_memory_total_gb": [
                                torch.cuda.get_device_properties(i).total_memory
                                / (1024**3)
                                for i in range(torch.cuda.device_count())
                            ],
                        }
                    )
            except Exception as e:
                self.logger.warning(f"Failed to get GPU info: {e}")

        return info


class EfficiencyBenchmark:
    """
    Enhanced benchmark for measuring model efficiency metrics.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the efficiency benchmark.

        Args:
            config: Configuration dictionary for the benchmark
        """
        self.logger = logging.getLogger("openevals.tasks.efficiency")
        self.config = config
        self.sample_prompts = config.get(
            "sample_prompts",
            [
                "Explain the theory of relativity in simple terms",
                "Write a short story about a robot who discovers emotions",
                "Summarize the key events of World War II",
                "Describe the process of photosynthesis in plants",
                "What are the benefits of renewable energy?",
            ],
        )
        self.output_lengths = config.get("output_lengths", [64, 128, 256, 512])
        self.warmup_runs = config.get("warmup_runs", 2)
        self.measurement_runs = config.get("measurement_runs", 3)

        # Initialize system monitor
        self.monitor = SystemMonitor()
        self.system_info = self.monitor.get_system_info()

        self.logger.info(
            f"Initialized efficiency benchmark with {len(self.sample_prompts)} prompts"
        )
        self.logger.info(f"Output lengths: {self.output_lengths}")

    def _cleanup_memory(self):
        """Clean up memory between measurements."""
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception as e:
                self.logger.warning(f"Failed to clear CUDA cache: {e}")

    def _measure_single_generation(
        self, model: ModelWrapper, prompt: str, max_tokens: int
    ) -> EfficiencyMetrics:
        """
        Measure efficiency metrics for a single text generation.

        Args:
            model: The model to benchmark
            prompt: Input prompt
            max_tokens: Maximum tokens to generate

        Returns:
            EfficiencyMetrics object with measurements
        """
        try:
            # Clean up before measurement
            self._cleanup_memory()

            # Record initial state
            memory_before = self.monitor.get_memory_usage()
            gpu_memory_before = self.monitor.get_gpu_memory_usage()

            # Synchronize if using CUDA
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.synchronize()

            # Time the generation
            start_time = time.perf_counter()

            generated_text = model.generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=0.0,  # Deterministic for consistent timing
                do_sample=False,
            )

            # Synchronize if using CUDA
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.perf_counter()

            # Record final state
            memory_after = self.monitor.get_memory_usage()
            gpu_memory_after = self.monitor.get_gpu_memory_usage()
            cpu_usage = self.monitor.get_cpu_usage()

            # Calculate metrics
            latency = end_time - start_time

            # Count actual tokens generated using the tokenizer
            actual_tokens = 0
            if hasattr(model, "tokenizer") and model.tokenizer is not None:
                try:
                    # Tokenize the generated text to get exact token count
                    tokens = model.tokenizer.encode(generated_text)
                    actual_tokens = len(tokens)
                except Exception as e:
                    self.logger.warning(f"Failed to tokenize for accurate count: {e}")
                    # Fallback to estimation only if tokenization fails
                    actual_tokens = min(len(generated_text.split()) * 1.3, max_tokens)
            else:
                # Fallback if no tokenizer available
                actual_tokens = min(len(generated_text.split()) * 1.3, max_tokens)

            tokens_per_second = actual_tokens / latency if latency > 0 else 0

            memory_delta = (memory_after - memory_before) * 1024  # Convert to MB
            gpu_memory_delta = None
            if gpu_memory_before is not None and gpu_memory_after is not None:
                gpu_memory_delta = (
                    gpu_memory_after - gpu_memory_before
                ) * 1024  # Convert to MB

            return EfficiencyMetrics(
                latency=latency,
                tokens_per_second=tokens_per_second,
                memory_usage_mb=max(
                    0, memory_delta
                ),  # Don't report negative memory usage
                gpu_memory_mb=gpu_memory_delta,
                cpu_usage_percent=cpu_usage,
            )

        except Exception as e:
            self.logger.error(f"Error during efficiency measurement: {e}")
            return EfficiencyMetrics(
                latency=0.0, tokens_per_second=0.0, memory_usage_mb=0.0, error=str(e)
            )

    def _run_warmup(self, model: ModelWrapper):
        """Run warmup iterations to stabilize performance."""
        self.logger.info("Running warmup iterations...")

        warmup_prompt = self.sample_prompts[0]
        warmup_length = min(self.output_lengths)

        for i in range(self.warmup_runs):
            try:
                _ = model.generate(
                    warmup_prompt, max_new_tokens=warmup_length, temperature=0.0
                )
                self._cleanup_memory()
            except Exception as e:
                self.logger.warning(f"Warmup iteration {i+1} failed: {e}")

        self.logger.info("Warmup complete")

    def evaluate(self, model: ModelWrapper) -> Dict[str, Any]:
        """
        Evaluate the model's efficiency metrics.

        Args:
            model: The model to evaluate

        Returns:
            Dictionary containing efficiency metrics
        """
        self.logger.info(f"Evaluating efficiency metrics for {model.model_name}")

        # Run warmup
        self._run_warmup(model)

        results = {
            "latency": {},
            "tokens_per_second": {},
            "memory_usage": {},
            "gpu_memory_usage": {},
            "cpu_usage": {},
            "system_info": self.system_info,
            "config": {
                "measurement_runs": self.measurement_runs,
                "warmup_runs": self.warmup_runs,
                "num_prompts": len(self.sample_prompts),
            },
        }

        # Test each output length
        for length in self.output_lengths:
            self.logger.info(f"Testing with output length: {length}")

            # Collect measurements across multiple runs and prompts
            all_metrics = []

            for run in range(self.measurement_runs):
                for prompt_idx, prompt in enumerate(self.sample_prompts):
                    self.logger.debug(
                        f"Run {run+1}/{self.measurement_runs}, Prompt {prompt_idx+1}/{len(self.sample_prompts)}"
                    )

                    metrics = self._measure_single_generation(model, prompt, length)
                    if metrics.error is None:
                        all_metrics.append(metrics)
                    else:
                        self.logger.warning(f"Measurement failed: {metrics.error}")

            if not all_metrics:
                self.logger.error(f"No successful measurements for length {length}")
                continue

            # Calculate averages
            avg_latency = sum(m.latency for m in all_metrics) / len(all_metrics)
            avg_tps = sum(m.tokens_per_second for m in all_metrics) / len(all_metrics)
            avg_memory = sum(m.memory_usage_mb for m in all_metrics) / len(all_metrics)

            # GPU memory (if available)
            gpu_memory_values = [
                m.gpu_memory_mb for m in all_metrics if m.gpu_memory_mb is not None
            ]
            avg_gpu_memory = (
                sum(gpu_memory_values) / len(gpu_memory_values)
                if gpu_memory_values
                else None
            )

            # CPU usage (if available)
            cpu_usage_values = [
                m.cpu_usage_percent
                for m in all_metrics
                if m.cpu_usage_percent is not None
            ]
            avg_cpu_usage = (
                sum(cpu_usage_values) / len(cpu_usage_values)
                if cpu_usage_values
                else None
            )

            # Store results
            length_key = f"tokens_{length}"
            results["latency"][length_key] = avg_latency
            results["tokens_per_second"][length_key] = avg_tps
            results["memory_usage"][length_key] = avg_memory / 1024  # Convert to GB

            if avg_gpu_memory is not None:
                results["gpu_memory_usage"][length_key] = (
                    avg_gpu_memory / 1024
                )  # Convert to GB

            if avg_cpu_usage is not None:
                results["cpu_usage"][length_key] = avg_cpu_usage

            self.logger.info(
                f"Length {length}: "
                f"Latency: {avg_latency:.3f}s, "
                f"TPS: {avg_tps:.1f}, "
                f"Memory: {avg_memory/1024:.3f}GB"
            )

        # Add summary statistics
        if results["latency"]:
            all_latencies = list(results["latency"].values())
            all_tps = list(results["tokens_per_second"].values())

            results["summary"] = {
                "avg_latency": sum(all_latencies) / len(all_latencies),
                "avg_tokens_per_second": sum(all_tps) / len(all_tps),
                "total_measurements": len(all_metrics),
                "successful_measurements": len(
                    [m for m in all_metrics if m.error is None]
                ),
            }

        self.logger.info(f"Efficiency evaluation complete for {model.model_name}")
        return results
