"""
Leaderboard generation for the Gemma Benchmarking Suite.

This module provides tools to generate summary leaderboards from benchmark
results, suitable for formal reports and academic use.
"""

import logging
import datetime
from typing import Dict, Any, List, Optional


class LeaderboardGenerator:
    """
    Generates structured leaderboards from benchmark results.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the LeaderboardGenerator.

        Args:
            config: An optional configuration dictionary. This can be used
                    to customize leaderboard generation, such as defining
                    which tasks and metrics to include.
        """
        self.logger = logging.getLogger("gemma_benchmark.leaderboard")
        self.config = config or {}

        # Defines the primary metric for each task to be included in the leaderboard.
        # This dictionary can be extended to support new benchmark tasks.
        self.task_metrics = {
            "mmlu": "accuracy",
            "gsm8k": "accuracy",
            "humaneval": "pass_at_1",
            "arc": "accuracy",
            "truthfulqa": "mc2_accuracy",
        }

    def _get_metric(
        self, task_result: Dict[str, Any], metric_name: str
    ) -> Optional[float]:
        """
        Safely extracts a numeric metric from a task's result dictionary.

        It looks inside the 'overall' key for the specified metric name and handles
        the normalization of percentage-based scores.
        """
        if (
            task_result
            and "overall" in task_result
            and metric_name in task_result["overall"]
        ):
            value = task_result["overall"][metric_name]
            if isinstance(value, (int, float)):
                # Normalize scores reported as percentages (e.g., 85.0) to a 0-1 scale.
                if "accuracy" in metric_name and value > 1.0:
                    return value / 100.0
                return float(value)
        return None

    def _calculate_average_score(self, tasks: Dict[str, Any]) -> float:
        """
        Calculates the average score for a model across all relevant tasks.

        The average is computed only over the tasks defined in `self.task_metrics`.
        """
        scores = [
            score
            for task_name, metric in self.task_metrics.items()
            if (score := self._get_metric(tasks.get(task_name, {}), metric)) is not None
        ]
        return sum(scores) / len(scores) if scores else 0.0

    def generate_markdown_leaderboard(self, results: Dict[str, Dict[str, Any]]) -> str:
        """
        Generates a formal, formatted leaderboard in Markdown from benchmark results.

        Args:
            results: A dictionary containing the benchmark results, structured as
                     {model_name: {task_name: task_results}}.

        Returns:
            A string containing the leaderboard in Markdown format.
        """
        self.logger.info("Generating Markdown leaderboard...")
        leaderboard_data = []

        # 1. Process and aggregate results for each model.
        for model_name, tasks in results.items():
            if not isinstance(tasks, dict):
                self.logger.warning(
                    f"Skipping invalid result entry for model: {model_name}"
                )
                continue

            avg_score = self._calculate_average_score(tasks)
            leaderboard_data.append(
                {"model": model_name, "avg_score": avg_score, "tasks": tasks}
            )

        # 2. Sort models by their average score in descending order for ranking.
        leaderboard_data.sort(key=lambda x: x["avg_score"], reverse=True)

        # 3. Dynamically create the table header based on the configured tasks.
        header_tasks = sorted(self.task_metrics.keys())
        header = (
            "| Rank | Model | Average Score |"
            + " | ".join([t.upper() for t in header_tasks])
            + " |"
        )
        separator = (
            "|:----:|:------|:--------------:|:"
            + ":|:".join(["---:"] * len(header_tasks))
            + ":|"
        )

        # 4. Build the Markdown table.
        md_rows = [
            f"# Benchmark Results Leaderboard",
            f"Report Generated On: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            header,
            separator,
        ]

        for i, entry in enumerate(leaderboard_data):
            rank = i + 1
            model_name = entry["model"]
            avg_score_str = f"**{entry['avg_score']:.4f}**"

            row = [str(rank), model_name, avg_score_str]

            # Append the formatted score for each task defined in the header.
            for task_name in header_tasks:
                metric = self.task_metrics[task_name]
                score = self._get_metric(entry["tasks"].get(task_name, {}), metric)
                score_str = f"{score:.4f}" if score is not None else "N/A"
                row.append(score_str)

            md_rows.append("| " + " | ".join(row) + " |")

        self.logger.info("Markdown leaderboard generated successfully.")
        return "\n".join(md_rows)
