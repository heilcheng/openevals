#!/usr/bin/env python3
"""
Main script for running the Gemma benchmarking suite.
"""

import os
import sys
import argparse
import logging
import datetime
from pathlib import Path
import pkgutil
import importlib

# Add parent directory to path to make imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from gemma_benchmark.core.benchmark import GemmaBenchmark
from gemma_benchmark.visualization.charts import ChartGenerator
# Import for task registration
from gemma_benchmark.core.interfaces import BenchmarkFactory, AbstractBenchmark

def register_available_tasks():
    """
    Dynamically discover and register all benchmark tasks.
    This allows new tasks to be added to the tasks folder without
    changing this script.
    """
    logger = logging.getLogger("gemma_benchmark")
    logger.info("Registering available benchmark tasks...")
    import gemma_benchmark.tasks

    package = gemma_benchmark.tasks
    for _, module_name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
        try:
            module = importlib.import_module(module_name)
            for attribute_name in dir(module):
                attribute = getattr(module, attribute_name)
                # Register class if it's a subclass of AbstractBenchmark
                if (isinstance(attribute, type) and
                        issubclass(attribute, AbstractBenchmark) and
                        attribute is not AbstractBenchmark):
                    
                    task_type = attribute.__name__.replace('Benchmark', '').lower()
                    BenchmarkFactory.register_benchmark(task_type, attribute)
                    logger.debug(f"Registered benchmark '{task_type}'")
        except Exception as e:
            logger.warning(f"Could not register tasks from module {module_name}: {e}")

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
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Run Gemma benchmarks")
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to benchmark configuration YAML file"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific models to benchmark (if not specified, all models in config will be used)"
    )
    
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Specific tasks to run (if not specified, all tasks in config will be used)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for results (if not specified, a timestamp-based directory will be created)"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations from results"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    
    return parser.parse_args()

def main() -> None:
    """Main function."""
    args = parse_args()
    setup_logging(args.log_level)
    
    # Register tasks before doing anything else
    register_available_tasks()
    
    logger = logging.getLogger("gemma_benchmark")
    
    # Create timestamp for results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join("results", timestamp)
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Results will be saved to {output_dir}")
    
    # Initialize benchmark
    try:
        benchmark = GemmaBenchmark(args.config)
        
        # Load models - this call now works as intended
        benchmark.load_models(args.models)
        
        # Load tasks - this call now works as intended
        benchmark.load_tasks(args.tasks)
        
        # Run benchmarks
        results = benchmark.run_benchmarks()
        
        # Save results
        results_file = os.path.join(output_dir, "results.yaml")
        benchmark.save_results(results_file)
        logger.info(f"Saved results to {results_file}")
        
        # Generate visualizations if requested
        if args.visualize:
            logger.info("Generating visualizations...")
            vis_dir = os.path.join(output_dir, "visualizations")
            chart_generator = ChartGenerator(vis_dir)
            
            # Create performance heatmap
            chart_generator.create_performance_heatmap(results)
            
            # Create task-specific visualizations
            for task_name in benchmark.tasks.keys():
                chart_generator.create_model_comparison_chart(results, task_name)
                
                # Create subject breakdown charts for each model
                for model_name in benchmark.models.keys():
                    if (model_name in results and 
                        task_name in results[model_name] and 
                        "subjects" in results[model_name][task_name]):
                        chart_generator.create_subject_breakdown_chart(results, model_name, task_name)
            
            # Create efficiency comparison charts
            chart_generator.create_efficiency_comparison_chart(results)
            
            # Create model size vs performance charts
            for task_name in benchmark.tasks.keys():
                chart_generator.create_model_size_vs_performance_chart(results, task_name)
            
            logger.info(f"Visualizations saved to {vis_dir}")
        
        logger.info("Benchmark complete!")
        
    except Exception as e:
        logger.error(f"Error running benchmark: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()