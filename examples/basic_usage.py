#!/usr/bin/env python3
"""
Basic usage example for the OpenEvalsing Suite.

This script demonstrates how to:
1. Set up authentication
2. Run a simple benchmark
3. Generate visualizations
4. Access results
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openevals.auth import AuthManager
from openevals.core.benchmark import GemmaBenchmark
from openevals.visualization.charts import ChartGenerator


def setup_logging():
    """Set up logging for the example."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main():
    """Run basic benchmark example."""
    setup_logging()
    logger = logging.getLogger("examples.basic_usage")

    logger.info("Starting OpenEvals Example")

    # 1. Authentication
    logger.info("Checking for HuggingFace authentication token...")
    if not AuthManager().get_token():
        logger.error(
            "Authentication failed. Please set the HF_TOKEN environment "
            "variable or run `huggingface-cli login`."
        )
        return
    logger.info("HuggingFace token found.")

    # 2. Create a simple configuration
    config_path = "examples/simple_config.yaml"
    if not os.path.exists(config_path):
        logger.info("Creating example configuration...")
        create_example_config(config_path)

    # 3. Initialize benchmark
    logger.info("Initializing benchmark...")
    try:
        benchmark = GemmaBenchmark(config_path)

        # 4. Load models and tasks
        logger.info("Loading models...")
        benchmark.load_models(["gemma-2b"])

        logger.info("Loading tasks...")
        benchmark.load_tasks(["efficiency"])

        # 5. Run benchmarks
        logger.info("Running benchmarks...")
        results = benchmark.run_benchmarks()

        # 6. Save results
        logger.info("Saving results...")
        results_path = benchmark.save_results("examples/results.yaml")
        logger.info(f"Results saved to: {results_path}")

        # 7. Generate visualizations
        logger.info("Generating visualizations...")
        output_dir = os.path.dirname(results_path)
        chart_generator = ChartGenerator(os.path.join(output_dir, "charts"))

        efficiency_charts = chart_generator.create_efficiency_comparison_chart(results)
        for chart_type, path in efficiency_charts.items():
            logger.info(f"Generated {chart_type} chart: {path}")

        # 8. Display summary
        display_results_summary(results)

        logger.info("Benchmark completed successfully!")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise


def create_example_config(config_path: str):
    """Create a simple configuration file for the example."""
    import yaml

    config = {
        "models": {
            "gemma-2b": {
                "type": "gemma",
                "size": "2b",
                "variant": "it",
                "cache_dir": "cache/models",
                "quantization": True,
            }
        },
        "tasks": {
            "efficiency": {
                "type": "efficiency",
                "sample_prompts": [
                    "Explain quantum computing in simple terms",
                    "Write a haiku about artificial intelligence",
                    "Summarize the benefits of renewable energy",
                ],
                "output_lengths": [64, 128, 256],
            }
        },
        "output": {"path": "examples/results", "visualize": True},
        "hardware": {"device": "auto", "precision": "bfloat16", "quantization": True},
    }

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def display_results_summary(results):
    """Display a summary of the benchmark results."""
    logger = logging.getLogger("examples.basic_usage")

    logger.info("\n" + "=" * 50)
    logger.info("BENCHMARK RESULTS SUMMARY")
    logger.info("=" * 50)

    for model_name, model_results in results.items():
        logger.info(f"\nModel: {model_name}")

        for task_name, task_results in model_results.items():
            logger.info(f"  Task: {task_name}")

            if task_name == "efficiency":
                if "tokens_per_second" in task_results:
                    logger.info("    Performance:")
                    for length, tps in task_results["tokens_per_second"].items():
                        logger.info(f"      {length}: {tps:.2f} tokens/sec")

                if "latency" in task_results:
                    logger.info("    Latency:")
                    for length, latency in task_results["latency"].items():
                        logger.info(f"      {length}: {latency:.3f} seconds")

                if "system_info" in task_results:
                    sys_info = task_results["system_info"]
                    logger.info(
                        f"    System: {sys_info.get('os', 'Unknown')} | "
                        f"CPU: {sys_info.get('cpu_count', 'Unknown')} cores | "
                        f"RAM: {sys_info.get('memory_total', 'Unknown'):.1f}GB"
                    )

                    if sys_info.get("cuda_available"):
                        gpu_info = sys_info.get("gpu_name", ["Unknown"])
                        logger.info(
                            f"    GPU: {gpu_info[0] if gpu_info else 'Unknown'}"
                        )

            else:
                if "overall" in task_results and "accuracy" in task_results["overall"]:
                    accuracy = task_results["overall"]["accuracy"]
                    logger.info(f"    Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
