#!/usr/bin/env python3
"""
Advanced usage example for the Gemma Benchmarking Suite.

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

from gemma_benchmark.auth import setup_huggingface_auth
from gemma_benchmark.core.benchmark import GemmaBenchmark
from gemma_benchmark.visualization.charts import ChartGenerator
from gemma_benchmark.utils.metrics import aggregate_results, calculate_confidence_interval


def setup_logging():
    """Set up detailed logging for the example."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
                "device_map": "auto"
            },
            "gemma-9b": {
                "type": "gemma", 
                "size": "9b",
                "variant": "it",
                "cache_dir": "cache/models",
                "quantization": True,
                "device_map": "auto"
            }
        },
        "tasks": {
            "mmlu": {
                "type": "mmlu",
                "subset": "mathematics",  # Focus on math for faster evaluation
                "shot_count": 5
            },
            "gsm8k": {
                "type": "gsm8k",
                "shot_count": 5,
                "use_chain_of_thought": True
            },
            "efficiency": {
                "type": "efficiency",
                "sample_prompts": [
                    "Explain the theory of relativity in simple terms",
                    "Write a Python function to calculate fibonacci numbers",
                    "Describe the process of machine learning model training",
                    "Compare renewable vs fossil fuel energy sources"
                ],
                "output_lengths": [128, 256, 512]
            }
        },
        "evaluation": {
            "runs": 2,  # Multiple runs for statistical analysis
            "batch_size": 4,
            "statistical_tests": True,
            "confidence_level": 0.95
        },
        "output": {
            "path": "examples/advanced_results",
            "save_predictions": True,
            "visualize": True,
            "export_formats": ["json", "yaml"]
        },
        "hardware": {
            "device": "auto",
            "precision": "bfloat16",
            "mixed_precision": True,
            "gradient_checkpointing": True
        }
    }
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def run_multiple_evaluations(config_path: str, num_runs: int = 2) -> Dict[str, Any]:
    """Run multiple evaluation rounds for statistical analysis."""
    logger = logging.getLogger("examples.advanced_usage")
    
    all_results = []
    
    for run_idx in range(num_runs):
        logger.info(f"Starting evaluation run {run_idx + 1}/{num_runs}")
        
        # Initialize fresh benchmark for each run
        benchmark = GemmaBenchmark(config_path)
        
        # Load models and tasks
        benchmark.load_models()
        benchmark.load_tasks()
        
        # Run benchmarks
        results = benchmark.run_benchmarks()
        
        # Save individual run results
        run_results_path = f"examples/advanced_results/run_{run_idx + 1}_results.yaml"
        benchmark.save_results(run_results_path)
        
        all_results.append(results)
        logger.info(f"Completed run {run_idx + 1}")
    
    return all_results


def analyze_results(all_results: list) -> Dict[str, Any]:
    """Perform statistical analysis on multiple benchmark runs."""
    logger = logging.getLogger("examples.advanced_usage")
    
    logger.info("Performing statistical analysis...")
    
    # Extract accuracy values for each model-task combination
    analysis = {
        "summary": {},
        "statistical_significance": {},
        "confidence_intervals": {}
    }
    
    # Get all model-task combinations
    model_task_combinations = []
    if all_results:
        for model_name in all_results[0].keys():
            for task_name in all_results[0][model_name].keys():
                if task_name != "efficiency":  # Skip efficiency for accuracy analysis
                    model_task_combinations.append((model_name, task_name))
    
    # Analyze each combination
    for model_name, task_name in model_task_combinations:
        combination_key = f"{model_name}_{task_name}"
        
        # Extract accuracies across runs
        accuracies = []
        for run_result in all_results:
            if (model_name in run_result and 
                task_name in run_result[model_name] and
                "overall" in run_result[model_name][task_name]):
                accuracy = run_result[model_name][task_name]["overall"]["accuracy"]
                accuracies.append(accuracy)
        
        if accuracies:
            # Calculate statistics
            mean_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)
            
            # Calculate confidence interval
            n_samples = run_result[model_name][task_name]["overall"].get("total", 100)
            ci_lower, ci_upper = calculate_confidence_interval(mean_accuracy, n_samples)
            
            analysis["summary"][combination_key] = {
                "mean_accuracy": float(mean_accuracy),
                "std_accuracy": float(std_accuracy),
                "min_accuracy": float(min(accuracies)),
                "max_accuracy": float(max(accuracies)),
                "num_runs": len(accuracies)
            }
            
            analysis["confidence_intervals"][combination_key] = {
                "lower": float(ci_lower),
                "upper": float(ci_upper),
                "confidence_level": 0.95
            }
    
    return analysis


def compare_models(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Compare models across different tasks."""
    logger = logging.getLogger("examples.advanced_usage")
    
    logger.info("Comparing model performance...")
    
    comparison = {
        "model_rankings": {},
        "task_difficulty": {},
        "model_strengths": {}
    }
    
    # Group results by task
    task_results = {}
    for combo_key, stats in analysis["summary"].items():
        model_name, task_name = combo_key.split("_", 1)
        
        if task_name not in task_results:
            task_results[task_name] = {}
        
        task_results[task_name][model_name] = stats["mean_accuracy"]
    
    # Rank models for each task
    for task_name, model_accuracies in task_results.items():
        sorted_models = sorted(model_accuracies.items(), 
                             key=lambda x: x[1], reverse=True)
        comparison["model_rankings"][task_name] = [
            {"model": model, "accuracy": acc} for model, acc in sorted_models
        ]
    
    # Calculate task difficulty (lower average accuracy = harder)
    for task_name, model_accuracies in task_results.items():
        avg_accuracy = np.mean(list(model_accuracies.values()))
        comparison["task_difficulty"][task_name] = {
            "average_accuracy": float(avg_accuracy),
            "difficulty_rank": None  # Will be filled after sorting
        }
    
    # Rank tasks by difficulty
    sorted_tasks = sorted(comparison["task_difficulty"].items(),
                         key=lambda x: x[1]["average_accuracy"])
    
    for rank, (task_name, task_info) in enumerate(sorted_tasks):
        comparison["task_difficulty"][task_name]["difficulty_rank"] = rank + 1
    
    return comparison


def generate_comprehensive_visualizations(all_results: list, analysis: Dict[str, Any]):
    """Generate comprehensive visualizations for the advanced analysis."""
    logger = logging.getLogger("examples.advanced_usage")
    
    logger.info("Generating comprehensive visualizations...")
    
    # Use the last run's results for visualization structure
    latest_results = all_results[-1]
    
    output_dir = "examples/advanced_results/visualizations"
    chart_generator = ChartGenerator(output_dir)
    
    # 1. Performance heatmap
    heatmap_path = chart_generator.create_performance_heatmap(latest_results)
    logger.info(f"Generated performance heatmap: {heatmap_path}")
    
    # 2. Task-specific model comparisons
    for task_name in ["mmlu", "gsm8k"]:
        if any(task_name in model_results for model_results in latest_results.values()):
            comparison_path = chart_generator.create_model_comparison_chart(
                latest_results, task_name
            )
            logger.info(f"Generated {task_name} comparison: {comparison_path}")
    
    # 3. Efficiency comparisons
    efficiency_charts = chart_generator.create_efficiency_comparison_chart(latest_results)
    for chart_type, path in efficiency_charts.items():
        logger.info(f"Generated efficiency chart ({chart_type}): {path}")
    
    # 4. Subject breakdown for MMLU (if available)
    for model_name in latest_results.keys():
        if "mmlu" in latest_results[model_name]:
            subject_path = chart_generator.create_subject_breakdown_chart(
                latest_results, model_name, "mmlu"
            )
            if subject_path:
                logger.info(f"Generated subject breakdown for {model_name}: {subject_path}")


def save_comprehensive_report(all_results: list, analysis: Dict[str, Any], 
                            comparison: Dict[str, Any]):
    """Save a comprehensive analysis report."""
    logger = logging.getLogger("examples.advanced_usage")
    
    logger.info("Saving comprehensive report...")
    
    import datetime
    
    report = {
        "metadata": {
            "num_runs": len(all_results),
            "timestamp": str(datetime.datetime.now()),
            "models_evaluated": list(all_results[0].keys()) if all_results else [],
            "tasks_evaluated": list(set(
                task for result in all_results 
                for model_results in result.values() 
                for task in model_results.keys()
            )) if all_results else []
        },
        "statistical_analysis": analysis,
        "model_comparison": comparison,
        "raw_results": all_results
    }
    
    # Save as JSON for programmatic access
    report_path = "examples/advanced_results/comprehensive_report.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Comprehensive report saved: {report_path}")
    
    # Generate human-readable summary
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
    
    # Add rankings for each task
    if "model_comparison" in report and "model_rankings" in report["model_comparison"]:
        for task_name, rankings in report["model_comparison"]["model_rankings"].items():
            summary_content += f"**{task_name.upper()}:**\n"
            for i, ranking in enumerate(rankings):
                summary_content += f"{i+1}. {ranking['model']}: {ranking['accuracy']:.4f}\n"
            summary_content += "\n"
    
    # Add task difficulty analysis
    if "model_comparison" in report and "task_difficulty" in report["model_comparison"]:
        summary_content += "### Task Difficulty Analysis\n\n"
        
        difficulty_items = list(report["model_comparison"]["task_difficulty"].items())
        difficulty_items.sort(key=lambda x: x[1]["difficulty_rank"])
        
        for task_name, task_info in difficulty_items:
            difficulty = "Easy" if task_info["average_accuracy"] > 0.7 else "Medium" if task_info["average_accuracy"] > 0.5 else "Hard"
            summary_content += f"- **{task_name}**: {task_info['average_accuracy']:.4f} avg accuracy ({difficulty})\n"
    
    # Add statistical significance notes
    summary_content += "\n### Statistical Notes\n\n"
    summary_content += f"- All results based on {report['metadata']['num_runs']} independent runs\n"
    summary_content += "- Confidence intervals calculated at 95% confidence level\n"
    summary_content += "- Results show variability across runs indicating importance of multiple evaluations\n"
    
    # Save the summary
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(summary_content)


def main():
    """Run advanced benchmarking example."""
    setup_logging()
    logger = logging.getLogger("examples.advanced_usage")
    
    logger.info("🚀 Starting Advanced Gemma Benchmark Example")
    
    # 1. Authentication
    logger.info("Setting up authentication...")
    if not setup_huggingface_auth():
        logger.error("Authentication failed. Please set up your HF_TOKEN.")
        return
    
    # 2. Create advanced configuration
    config_path = "examples/advanced_config.yaml"
    logger.info("Creating advanced configuration...")
    create_advanced_config(config_path)
    
    try:
        # 3. Run multiple evaluation rounds
        logger.info("Running multiple evaluation rounds...")
        all_results = run_multiple_evaluations(config_path, num_runs=2)
        
        # 4. Statistical analysis
        logger.info("Performing statistical analysis...")
        analysis = analyze_results(all_results)
        
        # 5. Model comparison
        logger.info("Comparing models...")
        comparison = compare_models(analysis)
        
        # 6. Generate visualizations
        generate_comprehensive_visualizations(all_results, analysis)
        
        # 7. Save comprehensive report
        save_comprehensive_report(all_results, analysis, comparison)
        
        # 8. Display key findings
        display_key_findings(analysis, comparison)
        
        logger.info("✅ Advanced benchmark analysis completed successfully!")
        logger.info("📊 Check 'examples/advanced_results/' for detailed reports and visualizations")
        
    except Exception as e:
        logger.error(f"❌ Advanced benchmark failed: {e}")
        raise


def display_key_findings(analysis: Dict[str, Any], comparison: Dict[str, Any]):
    """Display key findings from the analysis."""
    logger = logging.getLogger("examples.advanced_usage")
    
    logger.info("\n" + "="*60)
    logger.info("KEY FINDINGS")
    logger.info("="*60)
    
    # Model rankings
    if "model_rankings" in comparison:
        logger.info("\n🏆 MODEL RANKINGS BY TASK:")
        for task_name, rankings in comparison["model_rankings"].items():
            logger.info(f"\n  📋 {task_name.upper()}:")
            for i, ranking in enumerate(rankings):
                medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                logger.info(f"    {medal} {ranking['model']}: {ranking['accuracy']:.4f}")
    
    # Task difficulty
    if "task_difficulty" in comparison:
        logger.info("\n📊 TASK DIFFICULTY RANKING:")
        difficulty_items = list(comparison["task_difficulty"].items())
        difficulty_items.sort(key=lambda x: x[1]["difficulty_rank"])
        
        for task_name, task_info in difficulty_items:
            difficulty_icon = "🟢" if task_info["average_accuracy"] > 0.7 else "🟡" if task_info["average_accuracy"] > 0.5 else "🔴"
            logger.info(f"    {difficulty_icon} {task_name}: {task_info['average_accuracy']:.4f} avg accuracy")
    
    # Statistical reliability
    if "summary" in analysis:
        logger.info("\n📈 STATISTICAL RELIABILITY:")
        for combo_key, stats in analysis["summary"].items():
            model_name, task_name = combo_key.split("_", 1)
            reliability = "High" if stats["std_accuracy"] < 0.02 else "Medium" if stats["std_accuracy"] < 0.05 else "Low"
            logger.info(f"    {model_name} on {task_name}: σ={stats['std_accuracy']:.4f} ({reliability} reliability)")


if __name__ == "__main__":
    main()
