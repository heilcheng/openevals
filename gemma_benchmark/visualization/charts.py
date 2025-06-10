"""
Advanced visualization system for benchmark results with charts.

This module provides comprehensive visualization capabilities for language model
benchmarking results, including performance comparisons, efficiency analysis,
statistical significance testing, and error analysis.

Dependencies:
    matplotlib>=3.4.0
    seaborn>=0.11.0  
    pandas>=1.3.0
    scipy>=1.7.0
    numpy>=1.20.0
"""

import os
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import json
from datetime import datetime

# Core plotting libraries
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

# Optional dependencies
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    logging.getLogger(__name__).warning("seaborn not available - using matplotlib defaults")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    logging.getLogger(__name__).warning("pandas not available - some export features disabled")

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logging.getLogger(__name__).warning("scipy not available - statistical tests disabled")

# Set style with proper fallbacks
def set_plotting_style():
    """Set matplotlib style with proper fallbacks."""
    available_styles = plt.style.available
    
    # Try different style options in order of preference
    style_options = [
        'seaborn-v0_8-whitegrid',
        'seaborn-whitegrid',
        'seaborn',
        'ggplot',
        'default'
    ]
    
    style_set = False
    for style in style_options:
        if style in available_styles:
            try:
                plt.style.use(style)
                style_set = True
                break
            except:
                continue
    
    if not style_set:
        plt.style.use('default')
    
    # Apply seaborn if available
    if HAS_SEABORN:
        try:
            sns.set_palette("husl")
        except:
            pass

# Apply the style
set_plotting_style()

# Set default parameters
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


class BenchmarkVisualizer:
    """
    Comprehensive visualization system for benchmark results.
    
    Provides multiple chart types for analyzing model performance,
    efficiency, and statistical significance.
    """
    
    def __init__(self, output_dir: str, style: str = "publication"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save generated charts
            style: Visualization style ('publication', 'presentation', 'web')
        """
        self.logger = logging.getLogger("gemma_benchmark.visualization")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.style = style
        self._setup_style()
        
        # Color schemes for different categories
        self.model_colors = {
            'gemma': '#4285F4',      # Google Blue
            'mistral': '#FF6B35',    # Orange
            'llama': '#34A853',      # Green
            'claude': '#9C27B0',     # Purple
            'gpt': '#FF5722',        # Deep Orange
        }
        
        self.task_colors = {
            'mmlu': '#1f77b4',
            'gsm8k': '#ff7f0e', 
            'humaneval': '#2ca02c',
            'arc': '#d62728',
            'truthfulqa': '#9467bd',
            'efficiency': '#8c564b'
        }
        
        self.logger.info(f"Initialized visualizer with {style} style, output: {output_dir}")
    
    def _setup_style(self):
        """Setup matplotlib style based on selected theme."""
        if self.style == "publication":
            plt.rcParams.update({
                'font.family': 'serif',
                'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times'],
                'figure.dpi': 300,
                'savefig.dpi': 300,
                'savefig.format': 'pdf',
                'text.usetex': False,  # Keep False for compatibility
            })
        elif self.style == "presentation":
            plt.rcParams.update({
                'font.family': 'sans-serif',
                'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
                'figure.figsize': (16, 10),
                'font.size': 14,
                'axes.titlesize': 18,
                'figure.titlesize': 22,
            })
        elif self.style == "web":
            plt.rcParams.update({
                'figure.figsize': (12, 8),
                'savefig.format': 'png',
                'savefig.bbox': 'tight',
                'savefig.pad_inches': 0.1,
            })
    
    def _get_model_color(self, model_name: str) -> str:
        """Get color for a model based on its family."""
        model_lower = model_name.lower()
        for family, color in self.model_colors.items():
            if family in model_lower:
                return color
        # Default color for unknown models
        return '#7f7f7f'
    
    def _extract_model_size(self, model_name: str) -> Optional[float]:
        """Extract model size in billions of parameters."""
        try:
            # Look for patterns like "2b", "7b", "13b" etc.
            import re
            match = re.search(r'(\d+(?:\.\d+)?)b', model_name.lower())
            if match:
                return float(match.group(1))
        except:
            pass
        return None
    
    def _save_figure(self, fig: plt.Figure, filename: str, formats: List[str] = None) -> List[str]:
        """Save figure in multiple formats."""
        if formats is None:
            formats = ['png', 'pdf'] if self.style == 'publication' else ['png']
        
        saved_paths = []
        for fmt in formats:
            filepath = self.output_dir / f"{filename}.{fmt}"
            fig.savefig(filepath, format=fmt, bbox_inches='tight', 
                       dpi=300 if fmt in ['png', 'jpg'] else None)
            saved_paths.append(str(filepath))
            
        self.logger.info(f"Saved {filename} in formats: {formats}")
        return saved_paths
    
    def create_performance_overview(self, results: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Create comprehensive performance overview dashboard.
        
        Args:
            results: Benchmark results dictionary
            
        Returns:
            List of saved file paths
        """
        self.logger.info("Creating performance overview dashboard...")
        
        # Extract data
        models = list(results.keys())
        tasks = set()
        for model_results in results.values():
            tasks.update([task for task in model_results.keys() if task != 'efficiency'])
        tasks = sorted(list(tasks))
        
        if not models or not tasks:
            self.logger.warning("No data available for performance overview")
            return []
        
        # Create subplot layout
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Main heatmap (top-left, large)
        ax_heatmap = fig.add_subplot(gs[0:2, 0:2])
        self._create_performance_heatmap(ax_heatmap, results, models, tasks)
        
        # 2. Model ranking (top-right)
        ax_ranking = fig.add_subplot(gs[0, 2])
        self._create_model_ranking(ax_ranking, results, models, tasks)
        
        # 3. Task difficulty (middle-right)
        ax_difficulty = fig.add_subplot(gs[1, 2])
        self._create_task_difficulty(ax_difficulty, results, tasks)
        
        # 4. Performance distribution (bottom row)
        ax_dist = fig.add_subplot(gs[2, :])
        self._create_performance_distribution(ax_dist, results, models, tasks)
        
        plt.suptitle('Language Model Benchmark Performance Overview', 
                    fontsize=20, fontweight='bold')
        
        return self._save_figure(fig, 'performance_overview')
    
    def _create_performance_heatmap(self, ax: plt.Axes, results: Dict, models: List[str], tasks: List[str]):
        """Create performance heatmap."""
        # Prepare data matrix
        data = np.full((len(models), len(tasks)), np.nan)
        
        for i, model in enumerate(models):
            for j, task in enumerate(tasks):
                if (task in results[model] and 
                    'overall' in results[model][task]):
                    # Handle different metric names
                    overall = results[model][task]['overall']
                    if 'accuracy' in overall:
                        data[i, j] = overall['accuracy']
                    elif 'pass_at_1' in overall:
                        data[i, j] = overall['pass_at_1']
                    elif 'truthfulness_rate' in overall:
                        data[i, j] = overall['truthfulness_rate']
        
        # Create heatmap
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(tasks)):
                if not np.isnan(data[i, j]):
                    text = ax.text(j, i, f'{data[i, j]:.2f}',
                                 ha="center", va="center",
                                 color="white" if data[i, j] < 0.5 else "black",
                                 fontweight='bold')
        
        # Customize
        ax.set_xticks(np.arange(len(tasks)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(tasks, rotation=45, ha='right')
        ax.set_yticklabels(models)
        ax.set_title('Performance Heatmap (Accuracy)', fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Accuracy', rotation=270, labelpad=15)
    
    def _create_model_ranking(self, ax: plt.Axes, results: Dict, models: List[str], tasks: List[str]):
        """Create overall model ranking."""
        # Calculate average performance
        avg_scores = {}
        for model in models:
            scores = []
            for task in tasks:
                if task in results[model] and 'overall' in results[model][task]:
                    overall = results[model][task]['overall']
                    # Handle different metric names
                    if 'accuracy' in overall:
                        scores.append(overall['accuracy'])
                    elif 'pass_at_1' in overall:
                        scores.append(overall['pass_at_1'])
                    elif 'truthfulness_rate' in overall:
                        scores.append(overall['truthfulness_rate'])
            avg_scores[model] = np.mean(scores) if scores else 0
        
        # Sort by performance
        sorted_models = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        models_sorted, scores_sorted = zip(*sorted_models)
        
        # Create horizontal bar chart
        colors = [self._get_model_color(model) for model in models_sorted]
        bars = ax.barh(range(len(models_sorted)), scores_sorted, color=colors, alpha=0.8)
        
        # Customize
        ax.set_yticks(range(len(models_sorted)))
        ax.set_yticklabels(models_sorted)
        ax.set_xlabel('Average Accuracy')
        ax.set_title('Model Ranking', fontweight='bold')
        ax.set_xlim(0, 1)
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, scores_sorted)):
            ax.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{score:.3f}', va='center', fontsize=10)

    def _create_task_difficulty(self, ax: plt.Axes, results: Dict, tasks: List[str]):
        """Create task difficulty analysis."""
        # Calculate average performance per task
        task_scores = {}
        for task in tasks:
            scores = []
            for model_results in results.values():
                if (task in model_results and 
                    'overall' in model_results[task] and
                    'accuracy' in model_results[task]['overall']):
                    scores.append(model_results[task]['overall']['accuracy'])
            task_scores[task] = np.mean(scores) if scores else 0
        
        # Sort by difficulty (lower score = harder)
        sorted_tasks = sorted(task_scores.items(), key=lambda x: x[1])
        tasks_sorted, scores_sorted = zip(*sorted_tasks)
        
        # Create bar chart
        colors = [self.task_colors.get(task, '#7f7f7f') for task in tasks_sorted]
        bars = ax.bar(range(len(tasks_sorted)), scores_sorted, color=colors, alpha=0.8)
        
        # Customize
        ax.set_xticks(range(len(tasks_sorted)))
        ax.set_xticklabels(tasks_sorted, rotation=45, ha='right')
        ax.set_ylabel('Average Accuracy')
        ax.set_title('Task Difficulty', fontweight='bold')
        ax.set_ylim(0, 1)
        
        # Add difficulty labels
        for i, (bar, score) in enumerate(zip(bars, scores_sorted)):
            difficulty = "Hard" if score < 0.4 else "Medium" if score < 0.7 else "Easy"
            ax.text(bar.get_x() + bar.get_width()/2, score + 0.02, 
                   difficulty, ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    def _create_performance_distribution(self, ax: plt.Axes, results: Dict, models: List[str], tasks: List[str]):
        """Create performance distribution across tasks."""
        # Prepare data for violin plot
        data_for_plot = []
        labels = []
        
        for model in models:
            model_scores = []
            for task in tasks:
                if (task in results[model] and 
                    'overall' in results[model][task] and
                    'accuracy' in results[model][task]['overall']):
                    model_scores.append(results[model][task]['overall']['accuracy'])
            
            if model_scores:
                data_for_plot.append(model_scores)
                labels.append(model)
        
        if data_for_plot:
            # Create violin plot
            parts = ax.violinplot(data_for_plot, positions=range(len(labels)), 
                                showmeans=True, showmedians=True)
            
            # Customize colors
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(self._get_model_color(labels[i]))
                pc.set_alpha(0.7)
            
            # Customize
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel('Accuracy Distribution')
            ax.set_title('Performance Distribution Across Tasks', fontweight='bold')
            ax.set_ylim(0, 1)
    
    def create_efficiency_analysis(self, results: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Create comprehensive efficiency analysis charts.
        
        Args:
            results: Benchmark results dictionary
            
        Returns:
            List of saved file paths
        """
        self.logger.info("Creating efficiency analysis...")
        
        # Filter models with efficiency data
        models_with_efficiency = {
            model: data for model, data in results.items() 
            if 'efficiency' in data and data['efficiency']
        }
        
        if not models_with_efficiency:
            self.logger.warning("No efficiency data found")
            return []
        
        # Create comprehensive efficiency dashboard
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Latency vs Throughput scatter plot
        self._create_latency_throughput_plot(ax1, models_with_efficiency)
        
        # 2. Memory usage scaling
        self._create_memory_scaling_plot(ax2, models_with_efficiency)
        
        # 3. Efficiency frontier
        self._create_efficiency_frontier(ax3, models_with_efficiency)
        
        # 4. Performance vs Efficiency trade-off
        self._create_performance_efficiency_tradeoff(ax4, results, models_with_efficiency)
        
        plt.suptitle('Model Efficiency Analysis', fontsize=20, fontweight='bold')
        plt.tight_layout()
        
        return self._save_figure(fig, 'efficiency_analysis')
    
    def _create_latency_throughput_plot(self, ax: plt.Axes, efficiency_data: Dict):
        """Create latency vs throughput scatter plot."""
        for model_name, model_data in efficiency_data.items():
            eff_data = model_data['efficiency']
            
            if 'latency' in eff_data and 'tokens_per_second' in eff_data:
                # Use medium length tokens as representative
                latencies = []
                throughputs = []
                
                for key in eff_data['latency'].keys():
                    if key in eff_data['tokens_per_second']:
                        latencies.append(eff_data['latency'][key])
                        throughputs.append(eff_data['tokens_per_second'][key])
                
                if latencies and throughputs:
                    color = self._get_model_color(model_name)
                    ax.scatter(latencies, throughputs, 
                             label=model_name, color=color, s=100, alpha=0.7)
                    
                    # Add model size annotation if available
                    model_size = self._extract_model_size(model_name)
                    if model_size:
                        ax.annotate(f'{model_size}B', 
                                  (np.mean(latencies), np.mean(throughputs)),
                                  xytext=(5, 5), textcoords='offset points',
                                  fontsize=9, alpha=0.8)
        
        ax.set_xlabel('Average Latency (seconds)')
        ax.set_ylabel('Throughput (tokens/sec)')
        ax.set_title('Latency vs Throughput Trade-off', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_memory_scaling_plot(self, ax: plt.Axes, efficiency_data: Dict):
        """Create memory usage scaling plot."""
        for model_name, model_data in efficiency_data.items():
            eff_data = model_data['efficiency']
            
            if 'memory_usage' in eff_data:
                token_lengths = []
                memory_usage = []
                
                for key, mem in eff_data['memory_usage'].items():
                    if key.startswith('tokens_'):
                        length = int(key.split('_')[1])
                        token_lengths.append(length)
                        memory_usage.append(mem)
                
                if token_lengths and memory_usage:
                    # Sort by token length
                    sorted_data = sorted(zip(token_lengths, memory_usage))
                    token_lengths, memory_usage = zip(*sorted_data)
                    
                    color = self._get_model_color(model_name)
                    ax.plot(token_lengths, memory_usage, 
                           marker='o', label=model_name, color=color, linewidth=2)
        
        ax.set_xlabel('Output Length (tokens)')
        ax.set_ylabel('Memory Usage (GB)')
        ax.set_title('Memory Scaling by Output Length', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_efficiency_frontier(self, ax: plt.Axes, efficiency_data: Dict):
        """Create efficiency frontier analysis."""
        # Calculate efficiency score (throughput/memory_usage)
        model_efficiency = {}
        
        for model_name, model_data in efficiency_data.items():
            eff_data = model_data['efficiency']
            
            if 'tokens_per_second' in eff_data and 'memory_usage' in eff_data:
                # Use average metrics
                avg_throughput = np.mean(list(eff_data['tokens_per_second'].values()))
                avg_memory = np.mean(list(eff_data['memory_usage'].values()))
                
                if avg_memory > 0:
                    efficiency_score = avg_throughput / avg_memory
                    model_efficiency[model_name] = {
                        'throughput': avg_throughput,
                        'memory': avg_memory,
                        'efficiency': efficiency_score
                    }
        
        if model_efficiency:
            models = list(model_efficiency.keys())
            throughputs = [model_efficiency[m]['throughput'] for m in models]
            memories = [model_efficiency[m]['memory'] for m in models]
            efficiencies = [model_efficiency[m]['efficiency'] for m in models]
            
            # Create scatter plot with efficiency as color
            scatter = ax.scatter(memories, throughputs, c=efficiencies, 
                               s=100, cmap='viridis', alpha=0.8)
            
            # Add model labels
            for i, model in enumerate(models):
                ax.annotate(model, (memories[i], throughputs[i]),
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=9)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Efficiency Score\n(tokens/sec per GB)', rotation=270, labelpad=20)
            
            ax.set_xlabel('Memory Usage (GB)')
            ax.set_ylabel('Throughput (tokens/sec)')
            ax.set_title('Efficiency Frontier', fontweight='bold')
    
    def _create_performance_efficiency_tradeoff(self, ax: plt.Axes, results: Dict, efficiency_data: Dict):
        """Create performance vs efficiency trade-off plot."""
        # Calculate average performance for models with efficiency data
        model_metrics = {}
        
        for model_name in efficiency_data.keys():
            if model_name in results:
                # Calculate average accuracy
                accuracies = []
                for task, task_results in results[model_name].items():
                    if (task != 'efficiency' and 
                        'overall' in task_results and 
                        'accuracy' in task_results['overall']):
                        accuracies.append(task_results['overall']['accuracy'])
                
                if accuracies:
                    avg_accuracy = np.mean(accuracies)
                    
                    # Get efficiency metrics
                    eff_data = efficiency_data[model_name]['efficiency']
                    if 'tokens_per_second' in eff_data:
                        avg_throughput = np.mean(list(eff_data['tokens_per_second'].values()))
                        
                        model_metrics[model_name] = {
                            'accuracy': avg_accuracy,
                            'throughput': avg_throughput
                        }
        
        if model_metrics:
            models = list(model_metrics.keys())
            accuracies = [model_metrics[m]['accuracy'] for m in models]
            throughputs = [model_metrics[m]['throughput'] for m in models]
            
            # Create scatter plot
            colors = [self._get_model_color(model) for model in models]
            ax.scatter(throughputs, accuracies, c=colors, s=100, alpha=0.8)
            
            # Add model labels
            for i, model in enumerate(models):
                ax.annotate(model, (throughputs[i], accuracies[i]),
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=9)
            
            ax.set_xlabel('Throughput (tokens/sec)')
            ax.set_ylabel('Average Accuracy')
            ax.set_title('Performance vs Efficiency Trade-off', fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    def create_statistical_analysis(self, results: Dict[str, Dict[str, Any]], 
                                  multi_run_data: Optional[List[Dict]] = None) -> List[str]:
        """
        Create statistical analysis charts with confidence intervals.
        
        Args:
            results: Single run results
            multi_run_data: Optional multi-run data for statistical analysis
            
        Returns:
            List of saved file paths
        """
        self.logger.info("Creating statistical analysis...")
        
        if multi_run_data:
            return self._create_multi_run_analysis(multi_run_data)
        else:
            return self._create_single_run_analysis(results)
    
    def _create_multi_run_analysis(self, multi_run_data: List[Dict]) -> List[str]:
        """Create analysis for multiple runs with error bars."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Extract all models and tasks
        all_models = set()
        all_tasks = set()
        for run_data in multi_run_data:
            all_models.update(run_data.keys())
            for model_results in run_data.values():
                all_tasks.update([task for task in model_results.keys() if task != 'efficiency'])
        
        all_models = sorted(list(all_models))
        all_tasks = sorted(list(all_tasks))
        
        # 1. Performance with confidence intervals
        self._create_performance_with_ci(ax1, multi_run_data, all_models, all_tasks)
        
        # 2. Variance analysis
        self._create_variance_analysis(ax2, multi_run_data, all_models, all_tasks)
        
        # 3. Statistical significance heatmap
        self._create_significance_heatmap(ax3, multi_run_data, all_models, all_tasks)
        
        # 4. Run-to-run consistency
        self._create_consistency_analysis(ax4, multi_run_data, all_models, all_tasks)
        
        plt.suptitle('Statistical Analysis (Multiple Runs)', fontsize=20, fontweight='bold')
        plt.tight_layout()
        
        return self._save_figure(fig, 'statistical_analysis')
    
    def _create_performance_with_ci(self, ax: plt.Axes, multi_run_data: List[Dict], 
                                   models: List[str], tasks: List[str]):
        """Create performance chart with confidence intervals."""
        # Calculate mean and std for each model
        model_stats = {}
        
        for model in models:
            all_scores = []
            for run_data in multi_run_data:
                if model in run_data:
                    model_scores = []
                    for task in tasks:
                        if (task in run_data[model] and 
                            'overall' in run_data[model][task] and
                            'accuracy' in run_data[model][task]['overall']):
                            model_scores.append(run_data[model][task]['overall']['accuracy'])
                    if model_scores:
                        all_scores.append(np.mean(model_scores))
            
            if all_scores:
                model_stats[model] = {
                    'mean': np.mean(all_scores),
                    'std': np.std(all_scores),
                    'scores': all_scores
                }
        
        if model_stats:
            models_with_data = list(model_stats.keys())
            means = [model_stats[m]['mean'] for m in models_with_data]
            stds = [model_stats[m]['std'] for m in models_with_data]
            colors = [self._get_model_color(model) for model in models_with_data]
            
            # Create bar chart with error bars
            bars = ax.bar(range(len(models_with_data)), means, yerr=stds, 
                         color=colors, alpha=0.7, capsize=5)
            
            ax.set_xticks(range(len(models_with_data)))
            ax.set_xticklabels(models_with_data, rotation=45, ha='right')
            ax.set_ylabel('Average Accuracy')
            ax.set_title('Performance with Confidence Intervals', fontweight='bold')
            ax.set_ylim(0, 1)
    
    def _create_variance_analysis(self, ax: plt.Axes, multi_run_data: List[Dict], 
                                 models: List[str], tasks: List[str]):
        """Create variance analysis across runs."""
        # Calculate coefficient of variation for each model
        model_cv = {}
        
        for model in models:
            all_scores = []
            for run_data in multi_run_data:
                if model in run_data:
                    model_scores = []
                    for task in tasks:
                        if (task in run_data[model] and 
                            'overall' in run_data[model][task] and
                            'accuracy' in run_data[model][task]['overall']):
                            model_scores.append(run_data[model][task]['overall']['accuracy'])
                    if model_scores:
                        all_scores.append(np.mean(model_scores))
            
            if all_scores and np.mean(all_scores) > 0:
                cv = np.std(all_scores) / np.mean(all_scores)
                model_cv[model] = cv
        
        if model_cv:
            models_sorted = sorted(model_cv.items(), key=lambda x: x[1])
            models_list, cv_values = zip(*models_sorted)
            
            colors = [self._get_model_color(model) for model in models_list]
            bars = ax.bar(range(len(models_list)), cv_values, color=colors, alpha=0.7)
            
            ax.set_xticks(range(len(models_list)))
            ax.set_xticklabels(models_list, rotation=45, ha='right')
            ax.set_ylabel('Coefficient of Variation')
            ax.set_title('Model Consistency (Lower is Better)', fontweight='bold')
            
            # Add consistency labels
            for i, (bar, cv) in enumerate(zip(bars, cv_values)):
                consistency = "High" if cv < 0.05 else "Medium" if cv < 0.1 else "Low"
                ax.text(bar.get_x() + bar.get_width()/2, cv + 0.001, 
                       consistency, ha='center', va='bottom', fontsize=9)
    
    def _create_significance_heatmap(self, ax: plt.Axes, multi_run_data: List[Dict], 
                                   models: List[str], tasks: List[str]):
        """Create statistical significance heatmap."""
        if not HAS_SCIPY:
            ax.text(0.5, 0.5, 'Statistical tests require scipy\nInstall with: pip install scipy',
                   ha='center', va='center', transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            ax.set_title('Statistical Significance (scipy required)', fontweight='bold')
            return
        
        # Simplified pairwise t-test (placeholder)
        # In practice, you'd want more sophisticated statistical tests
        n_models = len(models)
        p_values = np.ones((n_models, n_models))
        
        # Get model performance data
        model_data = {}
        for model in models:
            all_scores = []
            for run_data in multi_run_data:
                if model in run_data:
                    model_scores = []
                    for task in tasks:
                        if (task in run_data[model] and 
                            'overall' in run_data[model][task] and
                            'accuracy' in run_data[model][task]['overall']):
                            model_scores.append(run_data[model][task]['overall']['accuracy'])
                    if model_scores:
                        all_scores.append(np.mean(model_scores))
            model_data[model] = all_scores
        
        # Pairwise comparisons
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i != j and model1 in model_data and model2 in model_data:
                    data1 = model_data[model1]
                    data2 = model_data[model2]
                    if len(data1) > 1 and len(data2) > 1:
                        _, p_val = stats.ttest_ind(data1, data2)
                        p_values[i, j] = p_val
        
        # Create heatmap
        im = ax.imshow(p_values, cmap='RdYlGn_r', vmin=0, vmax=0.1)
        
        # Add significance markers
        for i in range(n_models):
            for j in range(n_models):
                if i != j:
                    significance = "***" if p_values[i, j] < 0.001 else \
                                 "**" if p_values[i, j] < 0.01 else \
                                 "*" if p_values[i, j] < 0.05 else ""
                    ax.text(j, i, significance, ha="center", va="center",
                           fontweight='bold', fontsize=12)
        
        ax.set_xticks(range(len(models)))
        ax.set_yticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_yticklabels(models)
        ax.set_title('Statistical Significance (p-values)', fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('p-value', rotation=270, labelpad=15)
    
    def _create_consistency_analysis(self, ax: plt.Axes, multi_run_data: List[Dict], 
                                   models: List[str], tasks: List[str]):
        """Create run-to-run consistency analysis."""
        # Calculate range of performance across runs for each model-task pair
        consistency_data = []
        
        for model in models:
            for task in tasks:
                task_scores = []
                for run_data in multi_run_data:
                    if (model in run_data and task in run_data[model] and
                        'overall' in run_data[model][task] and
                        'accuracy' in run_data[model][task]['overall']):
                        task_scores.append(run_data[model][task]['overall']['accuracy'])
                
                if len(task_scores) > 1:
                    score_range = max(task_scores) - min(task_scores)
                    consistency_data.append({
                        'model': model,
                        'task': task,
                        'range': score_range,
                        'mean': np.mean(task_scores)
                    })
        
        if consistency_data:
            # Create scatter plot
            models_list = [d['model'] for d in consistency_data]
            ranges = [d['range'] for d in consistency_data]
            means = [d['mean'] for d in consistency_data]
            
            colors = [self._get_model_color(model) for model in models_list]
            ax.scatter(means, ranges, c=colors, alpha=0.6, s=50)
            
            ax.set_xlabel('Mean Accuracy')
            ax.set_ylabel('Performance Range (Max - Min)')
            ax.set_title('Performance Consistency Analysis', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            if len(means) > 1:
                z = np.polyfit(means, ranges, 1)
                p = np.poly1d(z)
                ax.plot(sorted(means), p(sorted(means)), "r--", alpha=0.8, linewidth=2)
    
    def _create_single_run_analysis(self, results: Dict[str, Dict[str, Any]]) -> List[str]:
        """Create analysis for single run results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Simple analysis without statistical tests
        models = list(results.keys())
        tasks = set()
        for model_results in results.values():
            tasks.update([task for task in model_results.keys() if task != 'efficiency'])
        tasks = sorted(list(tasks))
        
        # 1. Model comparison by task
        self._create_model_comparison_by_task(ax1, results, models, tasks)
        
        # 2. Task difficulty ranking
        self._create_task_difficulty_ranking(ax2, results, tasks)
        
        # 3. Performance correlation matrix
        self._create_performance_correlation(ax3, results, models, tasks)
        
        # 4. Score distribution
        self._create_score_distribution(ax4, results, models, tasks)
        
        plt.suptitle('Single Run Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return self._save_figure(fig, 'single_run_analysis')
    
    def _create_model_comparison_by_task(self, ax: plt.Axes, results: Dict, models: List[str], tasks: List[str]):
        """Create detailed model comparison by task."""
        x = np.arange(len(tasks))
        width = 0.8 / len(models)
        
        for i, model in enumerate(models):
            scores = []
            for task in tasks:
                if (task in results[model] and 
                    'overall' in results[model][task] and
                    'accuracy' in results[model][task]['overall']):
                    scores.append(results[model][task]['overall']['accuracy'])
                else:
                    scores.append(0)
            
            color = self._get_model_color(model)
            ax.bar(x + i * width, scores, width, label=model, color=color, alpha=0.8)
        
        ax.set_xlabel('Tasks')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Performance by Task', fontweight='bold')
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(tasks, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
    
    def _create_task_difficulty_ranking(self, ax: plt.Axes, results: Dict, tasks: List[str]):
        """Create task difficulty ranking."""
        task_scores = {}
        for task in tasks:
            scores = []
            for model_results in results.values():
                if (task in model_results and 
                    'overall' in model_results[task] and
                    'accuracy' in model_results[task]['overall']):
                    scores.append(model_results[task]['overall']['accuracy'])
            task_scores[task] = np.mean(scores) if scores else 0
        
        sorted_tasks = sorted(task_scores.items(), key=lambda x: x[1])
        task_names, scores = zip(*sorted_tasks)
        
        colors = [self.task_colors.get(task, '#7f7f7f') for task in task_names]
        bars = ax.barh(range(len(task_names)), scores, color=colors, alpha=0.8)
        
        ax.set_yticks(range(len(task_names)))
        ax.set_yticklabels(task_names)
        ax.set_xlabel('Average Accuracy')
        ax.set_title('Task Difficulty Ranking', fontweight='bold')
        ax.set_xlim(0, 1)
        
        # Add difficulty categories
        for i, (bar, score) in enumerate(zip(bars, scores)):
            category = "Easy" if score > 0.7 else "Medium" if score > 0.4 else "Hard"
            color = '#2E8B57' if score > 0.7 else '#DAA520' if score > 0.4 else '#DC143C'
            ax.text(score + 0.02, bar.get_y() + bar.get_height()/2, 
                   category, va='center', color=color, fontweight='bold')
    
    def _create_performance_correlation(self, ax: plt.Axes, results: Dict, models: List[str], tasks: List[str]):
        """Create task performance correlation matrix."""
        # Create performance matrix
        perf_matrix = np.full((len(models), len(tasks)), np.nan)
        
        for i, model in enumerate(models):
            for j, task in enumerate(tasks):
                if (task in results[model] and 
                    'overall' in results[model][task] and
                    'accuracy' in results[model][task]['overall']):
                    perf_matrix[i, j] = results[model][task]['overall']['accuracy']
        
        # Calculate correlation between tasks
        valid_tasks = []
        task_data = []
        
        for j, task in enumerate(tasks):
            task_scores = perf_matrix[:, j]
            if not np.all(np.isnan(task_scores)):
                valid_tasks.append(task)
                task_data.append(task_scores[~np.isnan(task_scores)])
        
        if len(valid_tasks) > 1:
            # Calculate correlation matrix
            corr_matrix = np.corrcoef([data for data in task_data if len(data) > 1])
            
            if corr_matrix.size > 1:
                im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
                
                ax.set_xticks(range(len(valid_tasks)))
                ax.set_yticks(range(len(valid_tasks)))
                ax.set_xticklabels(valid_tasks, rotation=45, ha='right')
                ax.set_yticklabels(valid_tasks)
                ax.set_title('Task Performance Correlation', fontweight='bold')
                
                # Add correlation values
                for i in range(len(valid_tasks)):
                    for j in range(len(valid_tasks)):
                        text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                                     ha="center", va="center",
                                     color="white" if abs(corr_matrix[i, j]) > 0.5 else "black")
                
                plt.colorbar(im, ax=ax, shrink=0.8)
    
    def _create_score_distribution(self, ax: plt.Axes, results: Dict, models: List[str], tasks: List[str]):
        """Create overall score distribution."""
        all_scores = []
        
        for model in models:
            for task in tasks:
                if (task in results[model] and 
                    'overall' in results[model][task] and
                    'accuracy' in results[model][task]['overall']):
                    all_scores.append(results[model][task]['overall']['accuracy'])
        
        if all_scores:
            ax.hist(all_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(np.mean(all_scores), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(all_scores):.3f}')
            ax.axvline(np.median(all_scores), color='green', linestyle='--', 
                      label=f'Median: {np.median(all_scores):.3f}')
            
            ax.set_xlabel('Accuracy')
            ax.set_ylabel('Frequency')
            ax.set_title('Score Distribution', fontweight='bold')
            ax.legend()
    
    def export_results_summary(self, results: Dict[str, Dict[str, Any]], 
                             output_format: str = 'json') -> str:
        """
        Export a structured summary of results.
        
        Args:
            results: Benchmark results
            output_format: 'json', 'csv', or 'excel'
            
        Returns:
            Path to exported file
        """
        summary = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'models_evaluated': list(results.keys()),
                'tasks_evaluated': list(set(
                    task for model_results in results.values() 
                    for task in model_results.keys()
                ))
            },
            'performance_summary': {},
            'efficiency_summary': {}
        }
        
        # Performance summary
        for model_name, model_results in results.items():
            model_summary = {}
            for task_name, task_results in model_results.items():
                if task_name != 'efficiency' and 'overall' in task_results:
                    model_summary[task_name] = task_results['overall']
                elif task_name == 'efficiency':
                    summary['efficiency_summary'][model_name] = task_results
            summary['performance_summary'][model_name] = model_summary
        
        # Export
        if output_format == 'json':
            output_path = self.output_dir / 'results_summary.json'
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
        
        elif output_format == 'csv':
            if not HAS_PANDAS:
                self.logger.warning("Pandas not available, using basic CSV export")
                # Basic CSV export without pandas
                output_path = self.output_dir / 'results_summary.csv'
                
                with open(output_path, 'w') as f:
                    # Write header
                    f.write("model,task,accuracy,total\n")
                    
                    # Write data
                    for model_name, model_results in summary['performance_summary'].items():
                        for task_name, task_results in model_results.items():
                            accuracy = task_results.get('accuracy', 'N/A')
                            total = task_results.get('total', 'N/A')
                            f.write(f"{model_name},{task_name},{accuracy},{total}\n")
            else:
                # Full pandas export
                rows = []
                for model_name, model_results in summary['performance_summary'].items():
                    for task_name, task_results in model_results.items():
                        row = {
                            'model': model_name,
                            'task': task_name,
                            **task_results
                        }
                        rows.append(row)
                
                df = pd.DataFrame(rows)
                output_path = self.output_dir / 'results_summary.csv'
                df.to_csv(output_path, index=False)
        
        self.logger.info(f"Exported results summary to {output_path}")
        return str(output_path)


# Backward Compatibility Layer
class ChartGenerator:
    """
    Backward-compatible chart generator that wraps BenchmarkVisualizer.
    
    This class maintains compatibility with the existing benchmark runner
    while providing access to the enhanced visualization capabilities.
    """
    
    def __init__(self, output_dir: str):
        """Initialize chart generator."""
        self.visualizer = BenchmarkVisualizer(output_dir, style="web")
        self.output_dir = output_dir
        self.logger = logging.getLogger("gemma_benchmark.visualization.charts")
    
    def create_performance_heatmap(self, results: Dict[str, Dict[str, Any]]) -> str:
        """Create performance heatmap (backward compatible)."""
        try:
            files = self.visualizer.create_performance_overview(results)
            return files[0] if files else ""
        except Exception as e:
            self.logger.error(f"Error creating performance heatmap: {e}")
            return ""
    
    def create_model_comparison_chart(self, results: Dict[str, Dict[str, Any]], task_name: str) -> str:
        """Create model comparison chart for specific task."""
        try:
            # Extract data for specific task
            task_results = {}
            for model_name, model_data in results.items():
                if task_name in model_data:
                    task_results[model_name] = {task_name: model_data[task_name]}
            
            if not task_results:
                self.logger.warning(f"No data found for task {task_name}")
                return ""
            
            # Create simple comparison chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            models = list(task_results.keys())
            scores = []
            
            for model in models:
                if ('overall' in task_results[model][task_name] and 
                    'accuracy' in task_results[model][task_name]['overall']):
                    scores.append(task_results[model][task_name]['overall']['accuracy'])
                else:
                    scores.append(0)
            
            colors = [self.visualizer._get_model_color(model) for model in models]
            bars = ax.bar(models, scores, color=colors, alpha=0.8)
            
            # Add value labels
            for bar, score in zip(bars, scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_ylabel('Accuracy')
            ax.set_title(f'Model Performance on {task_name.upper()}', fontweight='bold')
            ax.set_ylim(0, 1.1)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            filename = f"{task_name}_model_comparison"
            saved_files = self.visualizer._save_figure(fig, filename)
            plt.close()
            
            return saved_files[0] if saved_files else ""
            
        except Exception as e:
            self.logger.error(f"Error creating model comparison chart: {e}")
            return ""
    
    def create_efficiency_comparison_chart(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """Create efficiency comparison charts."""
        try:
            files = self.visualizer.create_efficiency_analysis(results)
            # Return dict for backward compatibility
            return {"efficiency_analysis": files[0]} if files else {}
        except Exception as e:
            self.logger.error(f"Error creating efficiency charts: {e}")
            return {}
    
    def create_subject_breakdown_chart(self, results: Dict[str, Dict[str, Any]], 
                                     model_name: str, task_name: str) -> str:
        """Create subject breakdown chart for MMLU-like tasks."""
        try:
            if (model_name not in results or 
                task_name not in results[model_name] or 
                'subjects' not in results[model_name][task_name]):
                self.logger.warning(f"No subject data for {model_name} on {task_name}")
                return ""
            
            subject_data = results[model_name][task_name]['subjects']
            subjects = list(subject_data.keys())
            accuracies = [subject_data[s]['accuracy'] for s in subjects]
            
            # Sort by accuracy
            sorted_data = sorted(zip(subjects, accuracies), key=lambda x: x[1])
            subjects_sorted, accuracies_sorted = zip(*sorted_data)
            
            # Create horizontal bar chart
            fig, ax = plt.subplots(figsize=(12, max(6, len(subjects) * 0.4)))
            
            colors = plt.cm.RdYlGn([acc for acc in accuracies_sorted])
            bars = ax.barh(subjects_sorted, accuracies_sorted, color=colors, alpha=0.8)
            
            # Add value labels
            for bar, acc in zip(bars, accuracies_sorted):
                ax.text(acc + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{acc:.3f}', va='center', fontweight='bold')
            
            ax.set_xlabel('Accuracy')
            ax.set_title(f'{model_name} - {task_name.upper()} Subject Breakdown', 
                        fontweight='bold')
            ax.set_xlim(0, 1.1)
            plt.tight_layout()
            
            filename = f"{model_name}_{task_name}_subjects"
            saved_files = self.visualizer._save_figure(fig, filename)
            plt.close()
            
            return saved_files[0] if saved_files else ""
            
        except Exception as e:
            self.logger.error(f"Error creating subject breakdown: {e}")
            return ""
    
    def create_model_size_vs_performance_chart(self, results: Dict[str, Dict[str, Any]], 
                                             task_name: str) -> str:
        """Create model size vs performance chart."""
        try:
            models = []
            sizes = []
            performances = []
            
            for model_name, model_data in results.items():
                if task_name in model_data and 'overall' in model_data[task_name]:
                    size = self.visualizer._extract_model_size(model_name)
                    if size is not None:
                        models.append(model_name)
                        sizes.append(size)
                        performances.append(model_data[task_name]['overall']['accuracy'])
            
            if not models:
                self.logger.warning(f"No size data available for {task_name}")
                return ""
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            colors = [self.visualizer._get_model_color(model) for model in models]
            scatter = ax.scatter(sizes, performances, c=colors, s=100, alpha=0.8)
            
            # Add model labels
            for model, size, perf in zip(models, sizes, performances):
                ax.annotate(model, (size, perf), xytext=(5, 5), 
                          textcoords='offset points', fontsize=9)
            
            # Add trend line if we have enough points
            if len(sizes) > 2:
                z = np.polyfit(sizes, performances, 1)
                p = np.poly1d(z)
                ax.plot(sorted(sizes), p(sorted(sizes)), "r--", alpha=0.8, linewidth=2)
                
                # Add R² value
                correlation = np.corrcoef(sizes, performances)[0, 1]
                ax.text(0.05, 0.95, f'R² = {correlation**2:.3f}', 
                       transform=ax.transAxes, fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            ax.set_xlabel('Model Size (Billions of Parameters)')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'Model Size vs Performance on {task_name.upper()}', fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            filename = f"{task_name}_size_vs_performance"
            saved_files = self.visualizer._save_figure(fig, filename)
            plt.close()
            
            return saved_files[0] if saved_files else ""
            
        except Exception as e:
            self.logger.error(f"Error creating size vs performance chart: {e}")
            return ""


def create_comprehensive_report(results: Dict[str, Dict[str, Any]], 
                              output_dir: str,
                              multi_run_data: Optional[List[Dict]] = None,
                              style: str = "publication") -> Dict[str, List[str]]:
    """
    Create a comprehensive benchmark report with all visualizations.
    
    Args:
        results: Benchmark results dictionary
        output_dir: Output directory for all charts
        multi_run_data: Optional multi-run data for statistical analysis
        style: Visualization style
        
    Returns:
        Dictionary mapping chart types to saved file paths
    """
    visualizer = BenchmarkVisualizer(output_dir, style)
    
    generated_files = {}
    
    # Create all visualizations
    try:
        generated_files['overview'] = visualizer.create_performance_overview(results)
        generated_files['efficiency'] = visualizer.create_efficiency_analysis(results)
        generated_files['statistical'] = visualizer.create_statistical_analysis(results, multi_run_data)
        generated_files['summary'] = [visualizer.export_results_summary(results)]
        
        logging.getLogger("gemma_benchmark.visualization").info(
            f"Generated comprehensive report in {output_dir}"
        )
        
    except Exception as e:
        logging.getLogger("gemma_benchmark.visualization").error(
            f"Error generating report: {e}"
        )
        raise
    
    return generated_files


# Example Usage
if __name__ == "__main__":
    # Example of how to use the enhanced visualization system
    
    # Mock benchmark results for demonstration
    example_results = {
        "gemma-2b": {
            "mmlu": {
                "overall": {"accuracy": 0.65, "total": 1000},
                "subjects": {
                    "mathematics": {"accuracy": 0.58, "total": 100},
                    "physics": {"accuracy": 0.72, "total": 100},
                    "computer_science": {"accuracy": 0.68, "total": 100}
                }
            },
            "gsm8k": {
                "overall": {"accuracy": 0.45, "total": 500}
            },
            "efficiency": {
                "latency": {"tokens_128": 1.2, "tokens_256": 2.4, "tokens_512": 4.8},
                "tokens_per_second": {"tokens_128": 106, "tokens_256": 107, "tokens_512": 107},
                "memory_usage": {"tokens_128": 2.1, "tokens_256": 2.3, "tokens_512": 2.7}
            }
        },
        "gemma-9b": {
            "mmlu": {
                "overall": {"accuracy": 0.72, "total": 1000},
                "subjects": {
                    "mathematics": {"accuracy": 0.68, "total": 100},
                    "physics": {"accuracy": 0.78, "total": 100},
                    "computer_science": {"accuracy": 0.75, "total": 100}
                }
            },
            "gsm8k": {
                "overall": {"accuracy": 0.58, "total": 500}
            },
            "efficiency": {
                "latency": {"tokens_128": 2.1, "tokens_256": 4.2, "tokens_512": 8.4},
                "tokens_per_second": {"tokens_128": 61, "tokens_256": 61, "tokens_512": 61},
                "memory_usage": {"tokens_128": 8.2, "tokens_256": 8.5, "tokens_512": 9.1}
            }
        },
        "mistral-7b": {
            "mmlu": {
                "overall": {"accuracy": 0.68, "total": 1000},
                "subjects": {
                    "mathematics": {"accuracy": 0.62, "total": 100},
                    "physics": {"accuracy": 0.74, "total": 100},
                    "computer_science": {"accuracy": 0.71, "total": 100}
                }
            },
            "gsm8k": {
                "overall": {"accuracy": 0.52, "total": 500}
            },
            "efficiency": {
                "latency": {"tokens_128": 1.8, "tokens_256": 3.6, "tokens_512": 7.2},
                "tokens_per_second": {"tokens_128": 71, "tokens_256": 71, "tokens_512": 71},
                "memory_usage": {"tokens_128": 6.8, "tokens_256": 7.1, "tokens_512": 7.7}
            }
        }
    }
    
    # Create visualizations
    output_dir = "example_visualizations"
    
    # Method 1: Use the comprehensive report function
    print("Creating comprehensive report...")
    report_files = create_comprehensive_report(
        example_results, 
        output_dir, 
        style="presentation"
    )
    
    print(f"Generated files: {report_files}")
    
    # Method 2: Use individual visualization components
    print("\nCreating individual visualizations...")
    visualizer = BenchmarkVisualizer(output_dir + "_individual", style="web")
    
    # Performance overview
    overview_files = visualizer.create_performance_overview(example_results)
    print(f"Overview: {overview_files}")
    
    # Efficiency analysis
    efficiency_files = visualizer.create_efficiency_analysis(example_results)
    print(f"Efficiency: {efficiency_files}")
    
    # Export summary
    summary_file = visualizer.export_results_summary(example_results, 'json')
    print(f"Summary: {summary_file}")
    
    # Method 3: Use backward-compatible ChartGenerator
    print("\nUsing backward-compatible interface...")
    chart_gen = ChartGenerator(output_dir + "_compatible")
    
    heatmap = chart_gen.create_performance_heatmap(example_results)
    comparison = chart_gen.create_model_comparison_chart(example_results, "mmlu")
    efficiency = chart_gen.create_efficiency_comparison_chart(example_results)
    
    print(f"Heatmap: {heatmap}")
    print(f"Comparison: {comparison}")
    print(f"Efficiency: {efficiency}")
    
    print("\nVisualization examples completed!")