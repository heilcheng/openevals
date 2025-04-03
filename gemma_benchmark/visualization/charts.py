"""
Chart generation for benchmark results visualization.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Any, Optional

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

class ChartGenerator:
    """Generator for benchmark visualization charts."""
    
    def __init__(self, output_dir: str):
        """
        Initialize the chart generator.
        
        Args:
            output_dir: Directory to save generated charts
        """
        self.logger = logging.getLogger("gemma_benchmark.visualization.charts")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_performance_heatmap(self, results: Dict[str, Dict[str, Any]]) -> str:
        """
        Generate heatmap visualization of model performance across benchmarks.
        
        Args:
            results: Benchmark results dictionary
            
        Returns:
            Path to the saved heatmap image
        """
        self.logger.info("Creating performance heatmap...")
        
        # Extract models and tasks
        models = list(results.keys())
        tasks = set()
        for model_result in results.values():
            for task in model_result.keys():
                if task != "efficiency":  # Skip efficiency metrics for heatmap
                    tasks.add(task)
        tasks = sorted(list(tasks))
        
        # Create the data matrix
        data = np.zeros((len(models), len(tasks)))
        for i, model in enumerate(models):
            for j, task in enumerate(tasks):
                if task in results[model] and "overall" in results[model][task]:
                    accuracy = results[model][task]["overall"]["accuracy"]
                    data[i, j] = accuracy
        
        # Create the heatmap
        plt.figure(figsize=(12, len(models) * 0.5 + 2))
        im = plt.imshow(data, cmap="YlGnBu")
        
        # Add labels
        plt.xticks(np.arange(len(tasks)), tasks, rotation=45, ha="right")
        plt.yticks(np.arange(len(models)), models)
        
        # Add colorbar and annotations
        plt.colorbar(im, label="Accuracy")
        
        # Add annotations
        for i in range(len(models)):
            for j in range(len(tasks)):
                text = plt.text(j, i, f"{data[i, j]:.2f}",
                             ha="center", va="center", color="black" if data[i, j] > 0.7 else "white")
        
        # Add title and save
        plt.title("Model Performance Across Benchmarks")
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, "performance_heatmap.png")
        plt.savefig(output_path)
        plt.close()
        
        self.logger.info(f"Saved performance heatmap to {output_path}")
        return output_path
    
    def create_model_comparison_chart(self, results: Dict[str, Dict[str, Any]], task_name: str) -> str:
        """
        Generate bar chart comparing model performance on a specific task.
        
        Args:
            results: Benchmark results dictionary
            task_name: Name of the task to visualize
            
        Returns:
            Path to the saved chart image
        """
        self.logger.info(f"Creating model comparison chart for {task_name}...")
        
        models = []
        scores = []
        ci_errors = []
        
        for model_name, model_results in results.items():
            if task_name in model_results and "overall" in model_results[task_name]:
                models.append(model_name)
                
                # Get the accuracy
                accuracy = model_results[task_name]["overall"]["accuracy"]
                scores.append(accuracy)
                
                # Get confidence interval if available
                ci_lower = model_results[task_name]["overall"].get("accuracy", {}).get("ci_lower", None)
                ci_upper = model_results[task_name]["overall"].get("accuracy", {}).get("ci_upper", None)
                
                if ci_lower is not None and ci_upper is not None:
                    ci_errors.append([accuracy - ci_lower, ci_upper - accuracy])
                else:
                    ci_errors.append(None)
        
        if not models:
            self.logger.warning(f"No data found for task {task_name}")
            return ""
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, scores)
        
        # Add error bars for confidence intervals if available
        if any(ci is not None for ci in ci_errors):
            error_bars = np.array([ci for ci in ci_errors if ci is not None]).T
            plt.errorbar(
                [m for m, ci in zip(models, ci_errors) if ci is not None],
                [s for s, ci in zip(scores, ci_errors) if ci is not None],
                yerr=error_bars,
                fmt='none',
                color='black',
                capsize=5
            )
        
        # Customize the chart
        plt.ylim(0, 1.0)
        plt.title(f"Model Performance on {task_name}")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=45, ha="right")
        
        # Add note about statistical significance if performed
        if any(ci is not None for ci in ci_errors):
            plt.figtext(0.5, 0.01, "Error bars represent 95% confidence intervals", 
                     ha="center", fontsize=8, style="italic")
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f"{task_name}_comparison.png")
        plt.savefig(output_path)
        plt.close()
        
        self.logger.info(f"Saved model comparison chart to {output_path}")
        return output_path
    
    def create_efficiency_comparison_chart(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """
        Generate efficiency comparison charts.
        
        Args:
            results: Benchmark results dictionary
            
        Returns:
            Dictionary of chart types to saved chart paths
        """
        self.logger.info("Creating efficiency comparison charts...")
        
        models = list(results.keys())
        output_paths = {}
        
        # Check if efficiency data is available
        models_with_efficiency = [m for m in models if "efficiency" in results[m]]
        
        if not models_with_efficiency:
            self.logger.warning("No efficiency data found")
            return output_paths
        
        # Generate tokens per second chart
        plt.figure(figsize=(12, 6))
        
        x_positions = np.arange(len(models_with_efficiency))
        width = 0.2
        token_lengths = ["tokens_128", "tokens_256", "tokens_512", "tokens_1024"]
        
        # Colors for different token lengths
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, length in enumerate(token_lengths):
            values = []
            for model in models_with_efficiency:
                if ("efficiency" in results[model] and 
                    "tokens_per_second" in results[model]["efficiency"] and
                    length in results[model]["efficiency"]["tokens_per_second"]):
                    values.append(results[model]["efficiency"]["tokens_per_second"][length])
                else:
                    values.append(0)
            
            plt.bar(x_positions + (i - 1.5) * width, values, width, label=f"{length.split('_')[1]} tokens", color=colors[i])
        
        plt.title("Model Inference Speed by Output Length")
        plt.xlabel("Model")
        plt.ylabel("Tokens per Second")
        plt.xticks(x_positions, models_with_efficiency, rotation=45, ha="right")
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        tokens_path = os.path.join(self.output_dir, "tokens_per_second.png")
        plt.savefig(tokens_path)
        plt.close()
        output_paths["tokens_per_second"] = tokens_path
        
        # Generate latency chart
        plt.figure(figsize=(12, 6))
        
        for i, length in enumerate(token_lengths):
            values = []
            for model in models_with_efficiency:
                if ("efficiency" in results[model] and 
                    "latency" in results[model]["efficiency"] and
                    length in results[model]["efficiency"]["latency"]):
                    values.append(results[model]["efficiency"]["latency"][length])
                else:
                    values.append(0)
            
            plt.bar(x_positions + (i - 1.5) * width, values, width, label=f"{length.split('_')[1]} tokens", color=colors[i])
        
        plt.title("Model Latency by Output Length")
        plt.xlabel("Model")
        plt.ylabel("Latency (seconds)")
        plt.xticks(x_positions, models_with_efficiency, rotation=45, ha="right")
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        latency_path = os.path.join(self.output_dir, "latency.png")
        plt.savefig(latency_path)
        plt.close()
        output_paths["latency"] = latency_path
        
        # Generate memory usage chart
        plt.figure(figsize=(12, 6))
        
        for i, length in enumerate(token_lengths):
            values = []
            for model in models_with_efficiency:
                if ("efficiency" in results[model] and 
                    "memory_usage" in results[model]["efficiency"] and
                    length in results[model]["efficiency"]["memory_usage"]):
                    values.append(results[model]["efficiency"]["memory_usage"][length])
                else:
                    values.append(0)
            
            plt.bar(x_positions + (i - 1.5) * width, values, width, label=f"{length.split('_')[1]} tokens", color=colors[i])
        
        plt.title("Model Memory Usage by Output Length")
        plt.xlabel("Model")
        plt.ylabel("Memory Usage (GB)")
        plt.xticks(x_positions, models_with_efficiency, rotation=45, ha="right")
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        memory_path = os.path.join(self.output_dir, "memory_usage.png")
        plt.savefig(memory_path)
        plt.close()
        output_paths["memory_usage"] = memory_path
        
        self.logger.info(f"Saved efficiency comparison charts to {self.output_dir}")
        return output_paths
    
    def create_subject_breakdown_chart(self, results: Dict[str, Dict[str, Any]], model_name: str, task_name: str) -> str:
        """
        Generate chart showing performance breakdown by subject for a specific model.
        
        Args:
            results: Benchmark results dictionary
            model_name: Name of the model to visualize
            task_name: Name of the task to visualize
            
        Returns:
            Path to the saved chart image
        """
        if (model_name not in results or 
            task_name not in results[model_name] or 
            "subjects" not in results[model_name][task_name]):
            self.logger.warning(f"No subject data found for {model_name} on {task_name}")
            return ""
        
        self.logger.info(f"Creating subject breakdown chart for {model_name} on {task_name}...")
        
        # Get subject results
        subject_results = results[model_name][task_name]["subjects"]
        subjects = list(subject_results.keys())
        
        # Sort subjects by accuracy
        subject_accuracies = [subject_results[s]["accuracy"] for s in subjects]
        sorted_indices = np.argsort(subject_accuracies)
        sorted_subjects = [subjects[i] for i in sorted_indices]
        sorted_accuracies = [subject_accuracies[i] for i in sorted_indices]
        
        # Create chart
        plt.figure(figsize=(12, max(6, len(subjects) * 0.3)))
        bars = plt.barh(sorted_subjects, sorted_accuracies)
        
        # Color bars based on accuracy
        for i, acc in enumerate(sorted_accuracies):
            bars[i].set_color(plt.cm.YlGnBu(acc))
        
        # Customize chart
        plt.xlim(0, 1.0)
        plt.title(f"{model_name} Performance by Subject on {task_name}")
        plt.xlabel("Accuracy")
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f"{model_name}_{task_name}_subjects.png")
        plt.savefig(output_path)
        plt.close()
        
        self.logger.info(f"Saved subject breakdown chart to {output_path}")
        return output_path
    
    def create_model_size_vs_performance_chart(self, results: Dict[str, Dict[str, Any]], task_name: str) -> str:
        """
        Generate chart comparing model size vs performance.
        
        Args:
            results: Benchmark results dictionary
            task_name: Name of the task to visualize
            
        Returns:
            Path to the saved chart image
        """
        self.logger.info(f"Creating model size vs performance chart for {task_name}...")
        
        # Extract model sizes from names (assuming format like "gemma-2b", "gemma-7b", etc.)
        model_sizes = {}
        for model_name in results.keys():
            try:
                size_str = model_name.split('-')[1].rstrip('b')
                model_sizes[model_name] = int(size_str)
            except (IndexError, ValueError):
                self.logger.warning(f"Could not extract size from model name: {model_name}")
                continue
        
        if not model_sizes:
            self.logger.warning("No model sizes could be determined")
            return ""
        
        # Get accuracies for the task
        models = []
        sizes = []
        accuracies = []
        
        for model_name, size in model_sizes.items():
            if (task_name in results[model_name] and 
                "overall" in results[model_name][task_name]):
                models.append(model_name)
                sizes.append(size)
                accuracies.append(results[model_name][task_name]["overall"]["accuracy"])
        
        if not models:
            self.logger.warning(f"No data found for task {task_name}")
            return ""
        
        # Create chart
        plt.figure(figsize=(10, 6))
        
        # Scatter plot with model names as annotations
        plt.scatter(sizes, accuracies, s=100)
        
        for i, model in enumerate(models):
            plt.annotate(model, (sizes[i], accuracies[i]), 
                      xytext=(5, 5), textcoords='offset points')
        
        # Add best fit line
        if len(sizes) > 1:
            z = np.polyfit(sizes, accuracies, 1)
            p = np.poly1d(z)
            plt.plot(sorted(sizes), p(sorted(sizes)), "r--", alpha=0.8)
        
        # Customize chart
        plt.title(f"Model Size vs Performance on {task_name}")
        plt.xlabel("Model Size (Billions of Parameters)")
        plt.ylabel("Accuracy")
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f"{task_name}_size_vs_performance.png")
        plt.savefig(output_path)
        plt.close()
        
        self.logger.info(f"Saved model size vs performance chart to {output_path}")
        return output_path