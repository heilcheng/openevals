#!/usr/bin/env python3
"""
Advanced usage example for the OpenEvalsing Suite.

This script demonstrates advanced features:
1. Multi-model comparison
2. Multiple benchmark tasks
3. Statistical analysis
4. Custom configuration
5. Results comparison and analysis
"""

import os
import sys
import yaml
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openevals.utils.auth import AuthManager
from openevals.core.benchmark import GemmaBenchmark
from openevals.visualization.charts import ChartGenerator
from openevals.utils.metrics import (
    aggregate_results,
    calculate_confidence_interval,
)


def setup_logging():
    """Set up detailed logging for the example."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def create_advanced_config(config_path: str):
    """Create an advanced configuration for comprehensive benchmarking."""
    config = {
        "models": {
            "gemma-2b": {
                "type": "gemma",
                "size": "2b",
                "variant": "it",
                "cache_dir": "cache/models",
                "quantization": True,
                "device_map": "auto",
            },
            "gemma-9b": {
                "type": "gemma",
                "size": "9b",
                "variant": "it",
                "cache_dir": "cache/models",
                "quantization": True,
                "device_map": "auto",
            },
        },
        "tasks": {
            "mmlu": {"type": "mmlu", "subset": "mathematics", "shot_count": 5},
            "gsm8k": {"type": "gsm8k", "shot_count": 5, "use_chain_of_thought": True},
            "efficiency": {
                "type": "efficiency",
                "sample_prompts": [
                    "Explain the theory of relativity in simple terms",
                    "Write a Python function to calculate fibonacci numbers",
                    "Describe the process of machine learning model training",
                    "Compare renewable vs fossil fuel energy sources",
                ],
                "output_lengths": [128, 256, 512],
            },
        },
        "evaluation": {
            "runs": 1,  # Script controls the number of runs
            "batch_size": 4,
            "statistical_tests": True,
            "confidence_level": 0.95,
        },
        "output": {
            "path": "examples/advanced_results",
            "save_predictions": True,
            "visualize": True,
            "export_formats": ["json", "yaml"],
        },
        "hardware": {
            "device": "auto",
            "precision": "bfloat16",
            "mixed_precision": True,
            "gradient_checkpointing": True,
        },
    }

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def run_multiple_evaluations(config_path: str, num_runs: int = 2) -> Dict[str, Any]:
    """Run multiple evaluation rounds for statistical analysis."""
    logger = logging.getLogger("examples.advanced_usage")

    all_results = []

    for run_idx in range(num_runs):
        logger.info(f"Starting evaluation run {run_idx + 1}/{num_runs}")

        benchmark = GemmaBenchmark(config_path)
        benchmark.load_models()
        benchmark.load_tasks()
        results = benchmark.run_benchmarks()

        run_results_path = f"examples/advanced_results/run_{run_idx + 1}_results.yaml"
        benchmark.save_results(run_results_path)

        all_results.append(results)
        logger.info(f"Completed run {run_idx + 1}")

    return all_results


def analyze_results(all_results: list) -> Dict[str, Any]:
    """Perform statistical analysis on multiple benchmark runs."""
    logger = logging.getLogger("examples.advanced_usage")

    logger.info("Performing statistical analysis...")

    analysis = {
        "summary": {},
        "statistical_significance": {},
        "confidence_intervals": {},
    }

    model_task_combinations = []
    if all_results:
        for model_name in all_results[0].keys():
            for task_name in all_results[0][model_name].keys():
                if task_name != "efficiency":
                    model_task_combinations.append((model_name, task_name))

    for model_name, task_name in model_task_combinations:
        combination_key = f"{model_name}_{task_name}"

        accuracies = []
        for run_result in all_results:
            if (
                model_name in run_result
                and task_name in run_result[model_name]
                and "overall" in run_result[model_name][task_name]
            ):
                accuracy = run_result[model_name][task_name]["overall"]["accuracy"]
                accuracies.append(accuracy)

        if accuracies:
            mean_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)

            n_samples = run_result[model_name][task_name]["overall"].get("total", 100)
            ci_lower, ci_upper = calculate_confidence_interval(mean_accuracy, n_samples)

            analysis["summary"][combination_key] = {
                "mean_accuracy": float(mean_accuracy),
                "std_accuracy": float(std_accuracy),
                "min_accuracy": float(min(accuracies)),
                "max_accuracy": float(max(accuracies)),
                "num_runs": len(accuracies),
            }

            analysis["confidence_intervals"][combination_key] = {
                "lower": float(ci_lower),
                "upper": float(ci_upper),
                "confidence_level": 0.95,
            }

    return analysis


def compare_models(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Compare models across different tasks."""
    logger = logging.getLogger("examples.advanced_usage")

    logger.info("Comparing model performance...")

    comparison = {"model_rankings": {}, "task_difficulty": {}, "model_strengths": {}}

    task_results = {}
    for combo_key, stats in analysis["summary"].items():
        model_name, task_name = combo_key.split("_", 1)
        if task_name not in task_results:
            task_results[task_name] = {}
        task_results[task_name][model_name] = stats["mean_accuracy"]

    for task_name, model_accuracies in task_results.items():
        sorted_models = sorted(
            model_accuracies.items(), key=lambda x: x[1], reverse=True
        )
        comparison["model_rankings"][task_name] = [
            {"model": model, "accuracy": acc} for model, acc in sorted_models
        ]

    for task_name, model_accuracies in task_results.items():
        avg_accuracy = np.mean(list(model_accuracies.values()))
        comparison["task_difficulty"][task_name] = {
            "average_accuracy": float(avg_accuracy),
            "difficulty_rank": None,
        }

    sorted_tasks = sorted(
        comparison["task_difficulty"].items(), key=lambda x: x[1]["average_accuracy"]
    )

    for rank, (task_name, task_info) in enumerate(sorted_tasks):
        comparison["task_difficulty"][task_name]["difficulty_rank"] = rank + 1

    return comparison


def generate_comprehensive_visualizations(all_results: list, analysis: Dict[str, Any]):
    """Generate comprehensive visualizations for the advanced analysis."""
    logger = logging.getLogger("examples.advanced_usage")

    logger.info("Generating comprehensive visualizations...")

    latest_results = all_results[-1]
    output_dir = "examples/advanced_results/visualizations"
    chart_generator = ChartGenerator(output_dir)

    heatmap_path = chart_generator.create_performance_heatmap(latest_results)
    logger.info(f"Generated performance heatmap: {heatmap_path}")

    for task_name in ["mmlu", "gsm8k"]:
        if any(task_name in model_results for model_results in latest_results.values()):
            comparison_path = chart_generator.create_model_comparison_chart(
                latest_results, task_name
            )
            logger.info(f"Generated {task_name} comparison: {comparison_path}")

    efficiency_charts = chart_generator.create_efficiency_comparison_chart(
        latest_results
    )
    for chart_type, path in efficiency_charts.items():
        logger.info(f"Generated efficiency chart ({chart_type}): {path}")

    for model_name in latest_results.keys():
        if "mmlu" in latest_results[model_name]:
            subject_path = chart_generator.create_subject_breakdown_chart(
                latest_results, model_name, "mmlu"
            )
            if subject_path:
                logger.info(
                    f"Generated subject breakdown for {model_name}: {subject_path}"
                )


def save_comprehensive_report(
    all_results: list, analysis: Dict[str, Any], comparison: Dict[str, Any]
):
    """Save a comprehensive analysis report."""
    logger = logging.getLogger("examples.advanced_usage")

    logger.info("Saving comprehensive report...")

    import datetime

    report = {
        "metadata": {
            "num_runs": len(all_results),
            "timestamp": str(datetime.datetime.now()),
            "models_evaluated": list(all_results[0].keys()) if all_results else [],
            "tasks_evaluated": (
                list(
                    set(
                        task
                        for result in all_results
                        for model_results in result.values()
                        for task in model_results.keys()
                    )
                )
                if all_results
                else []
            ),
        },
        "statistical_analysis": analysis,
        "model_comparison": comparison,
        "raw_results": all_results,
    }

    report_path = "examples/advanced_results/comprehensive_report.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Comprehensive report saved: {report_path}")

    summary_path = "examples/advanced_results/executive_summary.md"
    generate_executive_summary(report, summary_path)
    logger.info(f"Executive summary saved: {summary_path}")


def generate_executive_summary(report: Dict[str, Any], output_path: str):
    """Generate a human-readable executive summary."""

    summary_content = f"""# Gemma Model Benchmark Report

## Executive Summary

**Evaluation Overview:**
- Models Evaluated: {', '.join(report['metadata']['models_evaluated'])}
- Tasks Evaluated: {', '.join(report['metadata']['tasks_evaluated'])}
- Number of Runs: {report['metadata']['num_runs']}

## Key Findings

### Model Performance Rankings
"""

    if "model_comparison" in report and "model_rankings" in report["model_comparison"]:
        for task_name, rankings in report["model_comparison"]["model_rankings"].items():
            summary_content += f"**{task_name.upper()}:**\n"
            for i, ranking in enumerate(rankings):
                summary_content += (
                    f"{i+1}. {ranking['model']}: {ranking['accuracy']:.4f}\n"
                )
            summary_content += "\n"

    if "model_comparison" in report and "task_difficulty" in report["model_comparison"]:
        summary_content += "### Task Difficulty Analysis\n\n"

        difficulty_items = list(report["model_comparison"]["task_difficulty"].items())
        difficulty_items.sort(key=lambda x: x[1]["difficulty_rank"])

        for task_name, task_info in difficulty_items:
            difficulty = (
                "Easy"
                if task_info["average_accuracy"] > 0.7
                else "Medium" if task_info["average_accuracy"] > 0.5 else "Hard"
            )
            summary_content += f"- **{task_name}**: {task_info['average_accuracy']:.4f} avg accuracy ({difficulty})\n"

    summary_content += "\n### Statistical Notes\n\n"
    summary_content += (
        f"- All results based on {report['metadata']['num_runs']} independent runs\n"
    )
    summary_content += "- Confidence intervals calculated at 95% confidence level\n"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(summary_content)


def main():
    """Run advanced benchmarking example."""
    setup_logging()
    logger = logging.getLogger("examples.advanced_usage")

    logger.info("Starting Advanced OpenEvals Example")

    logger.info("Checking for HuggingFace authentication token...")
    if not AuthManager().get_token():
        logger.error(
            "Authentication failed. Please set the HF_TOKEN environment "
            "variable or run `huggingface-cli login`."
        )
        return
    logger.info("HuggingFace token found.")

    config_path = "examples/advanced_config.yaml"
    logger.info("Creating advanced configuration...")
    create_advanced_config(config_path)

    try:
        logger.info("Running multiple evaluation rounds...")
        all_results = run_multiple_evaluations(config_path, num_runs=2)

        logger.info("Performing statistical analysis...")
        analysis = analyze_results(all_results)

        logger.info("Comparing models...")
        comparison = compare_models(analysis)

        generate_comprehensive_visualizations(all_results, analysis)

        save_comprehensive_report(all_results, analysis, comparison)

        display_key_findings(analysis, comparison)

        logger.info("Advanced benchmark analysis completed successfully!")
        logger.info(
            "Check 'examples/advanced_results/' for detailed reports and visualizations"
        )

    except Exception as e:
        logger.error(f"Advanced benchmark failed: {e}")
        raise


def display_key_findings(analysis: Dict[str, Any], comparison: Dict[str, Any]):
    """Display key findings from the analysis."""
    logger = logging.getLogger("examples.advanced_usage")

    logger.info("\n" + "=" * 60)
    logger.info("KEY FINDINGS")
    logger.info("=" * 60)

    if "model_rankings" in comparison:
        logger.info("\nMODEL RANKINGS BY TASK:")
        for task_name, rankings in comparison["model_rankings"].items():
            logger.info(f"\n  {task_name.upper()}:")
            for i, ranking in enumerate(rankings):
                logger.info(f"    {i+1}. {ranking['model']}: {ranking['accuracy']:.4f}")

    if "task_difficulty" in comparison:
        logger.info("\nTASK DIFFICULTY RANKING:")
        difficulty_items = list(comparison["task_difficulty"].items())
        difficulty_items.sort(key=lambda x: x[1]["difficulty_rank"])

        for task_name, task_info in difficulty_items:
            logger.info(
                f"    {task_name}: {task_info['average_accuracy']:.4f} avg accuracy"
            )

    if "summary" in analysis:
        logger.info("\nSTATISTICAL RELIABILITY:")
        for combo_key, stats in analysis["summary"].items():
            model_name, task_name = combo_key.split("_", 1)
            reliability = (
                "High"
                if stats["std_accuracy"] < 0.02
                else "Medium" if stats["std_accuracy"] < 0.05 else "Low"
            )
            logger.info(
                f"    {model_name} on {task_name}: std_dev={stats['std_accuracy']:.4f} ({reliability} reliability)"
            )


if __name__ == "__main__":
    main()
