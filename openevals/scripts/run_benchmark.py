#!/usr/bin/env python3
"""
Main script for running the Gemma benchmarking suite.
"""

import argparse
import datetime
import importlib
import logging
import os
import pkgutil
import sys
from pathlib import Path

# Add parent directory to path to make imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from openevals.core.benchmark import Benchmark

# Import for task registration
from openevals.core.interfaces import AbstractBenchmark, BenchmarkFactory
from openevals.visualization.charts import ChartGenerator


def register_available_tasks():
    """
    Dynamically discover and register all benchmark tasks.
    This allows new tasks to be added to the tasks folder without
    changing this script.
    """
    logger = logging.getLogger("openevals")
    logger.info("Registering available benchmark tasks...")

    registered_count = 0
    failed_modules = []

    try:
        import openevals.tasks

        package = openevals.tasks

        # Walk through all modules in the tasks package
        for _, module_name, _ in pkgutil.walk_packages(
            package.__path__, package.__name__ + "."
        ):
            try:
                # Import the module
                module = importlib.import_module(module_name)

                # Look for benchmark classes in the module
                for attribute_name in dir(module):
                    attribute = getattr(module, attribute_name)

                    # Check if it's a class that inherits from AbstractBenchmark
                    if (
                        isinstance(attribute, type)
                        and issubclass(attribute, AbstractBenchmark)
                        and attribute is not AbstractBenchmark
                    ):
                        # Extract task type from class name
                        # E.g., MMLUBenchmark -> mmlu, GSM8KBenchmark -> gsm8k
                        task_type = attribute.__name__.replace("Benchmark", "").lower()

                        # Handle special cases for naming
                        if task_type == "gsm8k":
                            task_type = "gsm8k"  # Keep as is
                        elif task_type == "humaneval":
                            task_type = "humaneval"  # Keep as is
                        elif task_type == "truthfulqa":
                            task_type = "truthfulqa"  # Keep as is
                        elif task_type == "arc":
                            task_type = "arc"  # Keep as is

                        # Register the benchmark
                        BenchmarkFactory.register_benchmark(task_type, attribute)
                        logger.debug(
                            f"Registered benchmark '{task_type}' from {attribute.__name__}"
                        )
                        registered_count += 1

            except Exception as e:
                logger.warning(
                    f"Could not register tasks from module {module_name}: {e}"
                )
                failed_modules.append((module_name, str(e)))

    except Exception as e:
        logger.error(f"Failed to walk packages in openevals.tasks: {e}")

    # Log summary
    logger.info(f"Successfully registered {registered_count} benchmark tasks")

    if failed_modules:
        logger.warning(f"Failed to load {len(failed_modules)} modules:")
        for module_name, error in failed_modules:
            logger.warning(f"  - {module_name}: {error}")

    # Log all registered tasks
    try:
        registered_tasks = BenchmarkFactory.get_supported_types()
        if registered_tasks:
            logger.info(f"Available tasks: {', '.join(sorted(registered_tasks))}")
        else:
            logger.warning("No tasks were registered!")
    except Exception as e:
        logger.error(f"Could not retrieve registered tasks: {e}")

    # Manual fallback registration if automatic registration failed
    if registered_count == 0:
        logger.warning(
            "Automatic registration failed, attempting manual registration..."
        )

        manual_tasks = [
            ("openevals.tasks.mmlu", "MMLUBenchmark", "mmlu"),
            ("openevals.tasks.gsm8k", "Gsm8kBenchmark", "gsm8k"),
            ("openevals.tasks.humaneval", "HumanevalBenchmark", "humaneval"),
            ("openevals.tasks.efficiency", "EfficiencyBenchmark", "efficiency"),
            ("openevals.tasks.arc", "ArcBenchmark", "arc"),
            ("openevals.tasks.truthfulqa", "TruthfulqaBenchmark", "truthfulqa"),
        ]

        for module_path, class_name, task_type in manual_tasks:
            try:
                module = importlib.import_module(module_path)
                task_class = getattr(module, class_name)
                BenchmarkFactory.register_benchmark(task_type, task_class)
                logger.info(f"Manually registered '{task_type}'")
            except Exception as e:
                logger.debug(f"Could not manually register {task_type}: {e}")


def setup_logging(log_level: str = "INFO") -> None:
    """
    Set up logging configuration.

    Args:
        log_level: Logging level (default: INFO)
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run Gemma benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all models and tasks in config
  %(prog)s --config configs/benchmark_config.yaml

  # Run specific models and tasks
  %(prog)s --config configs/benchmark_config.yaml --models gemma-2b --tasks mmlu

  # Run with visualization
  %(prog)s --config configs/benchmark_config.yaml --visualize
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to benchmark configuration YAML file",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific models to benchmark (if not specified, all models in config will be used)",
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Specific tasks to run (if not specified, all tasks in config will be used)",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for results (if not specified, a timestamp-based directory will be created)",
    )

    parser.add_argument(
        "--visualize", action="store_true", help="Generate visualizations from results"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    parser.add_argument(
        "--list-tasks", action="store_true", help="List available tasks and exit"
    )

    parser.add_argument(
        "--list-models", action="store_true", help="List models in config and exit"
    )

    return parser.parse_args()


def list_available_tasks():
    """List all registered benchmark tasks."""
    print("\nAvailable benchmark tasks:")
    try:
        tasks = BenchmarkFactory.get_supported_types()
        if tasks:
            for task in sorted(tasks):
                print(f"  - {task}")
        else:
            print("  No tasks registered")
    except Exception as e:
        print(f"  Error retrieving tasks: {e}")


def list_config_models(config_path: str):
    """List all models defined in the configuration."""
    try:
        import yaml

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        print(f"\nModels defined in {config_path}:")
        if "models" in config:
            for model_name, model_config in config["models"].items():
                model_type = model_config.get("type", "unknown")
                print(f"  - {model_name} (type: {model_type})")
        else:
            print("  No models defined")
    except Exception as e:
        print(f"  Error reading config: {e}")


def main() -> None:
    """Main function."""
    args = parse_args()
    setup_logging(args.log_level)

    # Register tasks before doing anything else
    register_available_tasks()

    logger = logging.getLogger("openevals")

    # Handle list commands
    if args.list_tasks:
        list_available_tasks()
        sys.exit(0)

    if args.list_models:
        list_config_models(args.config)
        sys.exit(0)

    logger.info("🚀 Starting OpenEvals Suite")

    # Create timestamp for results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join("results", timestamp)

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {output_dir}")

    # Initialize benchmark
    try:
        logger.info(f"Loading configuration from: {args.config}")
        benchmark = Benchmark(args.config)

        # Load models
        if args.models:
            logger.info(f"Loading specified models: {', '.join(args.models)}")
            benchmark.load_models(args.models)
        else:
            logger.info("Loading all models from configuration")
            benchmark.load_models()

        # Load tasks
        if args.tasks:
            logger.info(f"Loading specified tasks: {', '.join(args.tasks)}")
            benchmark.load_tasks(args.tasks)
        else:
            logger.info("Loading all tasks from configuration")
            benchmark.load_tasks()

        # Check that we have something to run
        if not benchmark.models:
            logger.error("No models were loaded successfully")
            sys.exit(1)

        if not benchmark.tasks:
            logger.error("No tasks were loaded successfully")
            sys.exit(1)

        logger.info(
            f"Loaded {len(benchmark.models)} model(s) and {len(benchmark.tasks)} task(s)"
        )

        # Run benchmarks
        logger.info("Starting benchmark evaluation...")
        results = benchmark.run_benchmarks()

        # Save results
        results_file = os.path.join(output_dir, "results.yaml")
        saved_path = benchmark.save_results(results_file)
        logger.info(f"Results saved to: {saved_path}")

        # Save configuration used
        config_copy = os.path.join(output_dir, "config_used.yaml")
        import shutil

        shutil.copy2(args.config, config_copy)
        logger.info(f"Configuration saved to: {config_copy}")

        # Generate visualizations if requested
        if args.visualize:
            logger.info("Generating visualizations...")
            vis_dir = os.path.join(output_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)

            chart_generator = ChartGenerator(vis_dir)

            try:
                # Create performance heatmap
                heatmap_path = chart_generator.create_performance_heatmap(results)
                if heatmap_path:
                    logger.info(f"Generated performance heatmap: {heatmap_path}")

                # Create task-specific visualizations
                for task_name in benchmark.tasks.keys():
                    try:
                        comparison_path = chart_generator.create_model_comparison_chart(
                            results, task_name
                        )
                        if comparison_path:
                            logger.info(
                                f"Generated {task_name} comparison: {comparison_path}"
                            )
                    except Exception as e:
                        logger.warning(
                            f"Could not create comparison chart for {task_name}: {e}"
                        )

                    # Create subject breakdown charts for MMLU
                    if task_name.lower().startswith("mmlu"):
                        for model_name in benchmark.models.keys():
                            if (
                                model_name in results
                                and task_name in results[model_name]
                                and "subjects" in results[model_name][task_name]
                            ):
                                try:
                                    subject_path = (
                                        chart_generator.create_subject_breakdown_chart(
                                            results, model_name, task_name
                                        )
                                    )
                                    if subject_path:
                                        logger.info(
                                            f"Generated subject breakdown: {subject_path}"
                                        )
                                except Exception as e:
                                    logger.warning(
                                        f"Could not create subject breakdown: {e}"
                                    )

                # Create efficiency comparison charts
                if any(
                    "efficiency" in task_name.lower()
                    for task_name in benchmark.tasks.keys()
                ):
                    try:
                        efficiency_charts = (
                            chart_generator.create_efficiency_comparison_chart(results)
                        )
                        for chart_type, path in efficiency_charts.items():
                            logger.info(
                                f"Generated efficiency chart ({chart_type}): {path}"
                            )
                    except Exception as e:
                        logger.warning(f"Could not create efficiency charts: {e}")

                # Create model size vs performance charts
                for task_name in benchmark.tasks.keys():
                    try:
                        size_perf_path = (
                            chart_generator.create_model_size_vs_performance_chart(
                                results, task_name
                            )
                        )
                        if size_perf_path:
                            logger.info(
                                f"Generated size vs performance chart: {size_perf_path}"
                            )
                    except Exception as e:
                        logger.warning(
                            f"Could not create size vs performance chart: {e}"
                        )

            except Exception as e:
                logger.error(
                    f"Error during visualization generation: {e}", exc_info=True
                )

            logger.info(f"Visualizations saved to: {vis_dir}")

        # Print summary
        print("\n" + "=" * 60)
        print("BENCHMARK COMPLETE")
        print("=" * 60)
        print(f"Results saved to: {output_dir}")
        print(f"\nModels evaluated: {', '.join(benchmark.models.keys())}")
        print(f"Tasks completed: {', '.join(benchmark.tasks.keys())}")

        if args.visualize:
            print(f"Visualizations: {vis_dir}")

        print("\n✅ Benchmark completed successfully!")

    except KeyboardInterrupt:
        logger.warning("Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running benchmark: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
