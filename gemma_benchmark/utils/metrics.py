"""
Metrics calculation utilities for benchmarking.
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union

def calculate_accuracy(correct: int, total: int) -> float:
    """
    Calculate simple accuracy metric.
    
    Args:
        correct: Number of correct predictions
        total: Total number of predictions
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    return correct / total if total > 0 else 0.0

def calculate_f1_score(precision: float, recall: float) -> float:
    """
    Calculate F1 score from precision and recall.
    
    Args:
        precision: Precision value
        recall: Recall value
        
    Returns:
        F1 score as a float between 0 and 1
    """
    return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

def calculate_pass_at_k(n_samples: int, n_correct: int, k: int) -> float:
    """
    Calculate pass@k for code evaluation tasks.
    
    Args:
        n_samples: Number of samples
        n_correct: Number of correct samples
        k: K value
        
    Returns:
        pass@k metric as a float between 0 and 1
    """
    if n_samples == 0:
        return 0.0
    return 1.0 - math.comb(n_samples - n_correct, k) / math.comb(n_samples, k)

def extract_answer(text: str) -> float:
    """
    Extract numerical answer from text for math reasoning tasks.
    
    Args:
        text: Text containing a numerical answer
        
    Returns:
        Extracted number or NaN if no number found
    """
    # Simple implementation - in a real benchmark this would be more sophisticated
    try:
        # Find the last number in the text
        words = text.replace(',', '').split()
        for word in reversed(words):
            try:
                return float(word)
            except ValueError:
                continue
        return float('nan')
    except Exception:
        return float('nan')

def calculate_execution_accuracy(predictions: List[str], references: List[str]) -> float:
    """
    Calculate execution accuracy for math reasoning tasks.
    
    Args:
        predictions: List of model predictions
        references: List of reference answers
        
    Returns:
        Execution accuracy as a float between 0 and 1
    """
    if not predictions:
        return 0.0
    
    correct = 0
    for pred, ref in zip(predictions, references):
        # Extract numerical answers
        pred_answer = extract_answer(pred)
        ref_answer = extract_answer(ref)
        
        # Check if they match within a small tolerance
        if not (math.isnan(pred_answer) or math.isnan(ref_answer)):
            if abs(pred_answer - ref_answer) < 1e-6:
                correct += 1
    
    return correct / len(predictions)

def calculate_confidence_interval(accuracy: float, n_samples: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for accuracy.
    
    Args:
        accuracy: Observed accuracy
        n_samples: Number of samples
        confidence: Confidence level (default: 0.95 for 95% confidence)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    import scipy.stats as stats
    
    # Standard error of the mean
    stderr = np.sqrt(accuracy * (1 - accuracy) / n_samples)
    
    # Z-score for the desired confidence level
    z_score = stats.norm.ppf((1 + confidence) / 2)
    
    # Confidence interval
    margin = z_score * stderr
    lower_bound = max(0.0, accuracy - margin)
    upper_bound = min(1.0, accuracy + margin)
    
    return (lower_bound, upper_bound)

def aggregate_results(all_run_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate results from multiple runs.
    
    Args:
        all_run_results: List of result dictionaries from multiple runs
        
    Returns:
        Aggregated results with mean, std, and confidence intervals
    """
    if not all_run_results:
        return {}
    
    # Initialize aggregated results
    agg_results = {
        "overall": {
            "accuracy": {
                "mean": 0.0,
                "std": 0.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0
            },
            "total": all_run_results[0]["overall"]["total"]
        },
        "subjects": {}
    }
    
    # Extract all subjects
    all_subjects = set()
    for run_result in all_run_results:
        if "subjects" in run_result:
            all_subjects.update(run_result["subjects"].keys())
    
    # Aggregate overall accuracy
    overall_accuracies = [run["overall"]["accuracy"] for run in all_run_results]
    agg_results["overall"]["accuracy"]["mean"] = np.mean(overall_accuracies)
    agg_results["overall"]["accuracy"]["std"] = np.std(overall_accuracies)
    
    # Calculate confidence interval
    n_samples = all_run_results[0]["overall"]["total"]
    ci_lower, ci_upper = calculate_confidence_interval(
        agg_results["overall"]["accuracy"]["mean"], 
        n_samples
    )
    agg_results["overall"]["accuracy"]["ci_lower"] = ci_lower
    agg_results["overall"]["accuracy"]["ci_upper"] = ci_upper
    
    # Aggregate subject accuracies
    for subject in all_subjects:
        subject_accuracies = []
        for run in all_run_results:
            if "subjects" in run and subject in run["subjects"]:
                subject_accuracies.append(run["subjects"][subject]["accuracy"])
        
        if subject_accuracies:
            agg_results["subjects"][subject] = {
                "accuracy": {
                    "mean": np.mean(subject_accuracies),
                    "std": np.std(subject_accuracies)
                },
                "total": all_run_results[0]["subjects"][subject]["total"]
            }
            
            # Calculate confidence interval
            n_samples = agg_results["subjects"][subject]["total"]
            ci_lower, ci_upper = calculate_confidence_interval(
                agg_results["subjects"][subject]["accuracy"]["mean"], 
                n_samples
            )
            agg_results["subjects"][subject]["accuracy"]["ci_lower"] = ci_lower
            agg_results["subjects"][subject]["accuracy"]["ci_upper"] = ci_upper
    
    return agg_results