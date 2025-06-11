"""
Core metrics and evaluation utilities for the benchmarking suite.
"""

import re
import math
from typing import List, Union, Optional


def extract_numerical_answer(text: str) -> float:
    """
    Extract numerical answer from text for math reasoning tasks with improved logic.
    
    This improved version:
    - Prioritizes answers after specific markers (####, answer:, etc.)
    - Handles multiple answer formats
    - Falls back to last number only when no clear markers found
    
    Args:
        text: Text containing a numerical answer
        
    Returns:
        Extracted number or NaN if no number found
    """
    try:
        # Look for GSM8K style answers first
        gsm8k_pattern = r"####\s*([0-9,]+(?:\.[0-9]+)?)"
        match = re.search(gsm8k_pattern, text)
        if match:
            return float(match.group(1).replace(',', ''))
        
        # Look for explicit answer markers
        answer_patterns = [
            r"answer is[:\s]+([0-9,]+(?:\.[0-9]+)?)",
            r"answer:[:\s]*([0-9,]+(?:\.[0-9]+)?)",
            r"= ([0-9,]+(?:\.[0-9]+)?)\s*$",
            r"total[:\s]+([0-9,]+(?:\.[0-9]+)?)",
            r"result[:\s]+([0-9,]+(?:\.[0-9]+)?)"
        ]
        
        text_lower = text.lower()
        for pattern in answer_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                return float(matches[-1].replace(',', ''))
        
        # Look for boxed answers
        boxed_pattern = r"\\boxed\{([0-9,]+(?:\.[0-9]+)?)\}"
        boxed_match = re.search(boxed_pattern, text)
        if boxed_match:
            return float(boxed_match.group(1).replace(',', ''))
        
        # Check last sentence for answer indicators
        sentences = text.strip().split('.')
        if sentences:
            last_sentence = sentences[-1].lower()
            if any(word in last_sentence for word in ['answer', 'total', 'result', 'therefore']):
                numbers = re.findall(r"([0-9,]+(?:\.[0-9]+)?)", sentences[-1])
                if numbers:
                    return float(numbers[-1].replace(',', ''))
        
        # Fallback: find the last number in the text
        all_numbers = re.findall(r"([0-9,]+(?:\.[0-9]+)?)", text)
        if all_numbers:
            return float(all_numbers[-1].replace(',', ''))
        
        return float('nan')
        
    except Exception:
        return float('nan')


def is_exact_match(predicted: Union[float, str], ground_truth: Union[float, str]) -> bool:
    """
    Check if predicted answer exactly matches ground truth.
    
    Args:
        predicted: Predicted answer (number or string)
        ground_truth: Ground truth answer (number or string)
        
    Returns:
        True if exact match, False otherwise
    """
    try:
        # Handle NaN values
        if isinstance(predicted, float) and math.isnan(predicted):
            return False
        if isinstance(ground_truth, float) and math.isnan(ground_truth):
            return False
            
        # Convert to float if possible for numerical comparison
        try:
            pred_num = float(predicted)
            gt_num = float(ground_truth)
            # Use small tolerance for floating point comparison
            return abs(pred_num - gt_num) < 1e-6
        except (ValueError, TypeError):
            # Fall back to string comparison
            return str(predicted).strip().lower() == str(ground_truth).strip().lower()
            
    except Exception:
        return False


def anls(prediction: str, ground_truths: List[str]) -> float:
    """
    Average Normalized Levenshtein Similarity (ANLS) for document VQA tasks.
    
    This is a common metric for evaluating OCR-related tasks where the
    exact string match is too rigid.
    
    Args:
        prediction: The predicted text from the model.
        ground_truths: A list of acceptable ground truth strings.
        
    Returns:
        The ANLS score between the prediction and the best matching ground truth.
    """
    if not ground_truths:
        return 0.0
        
    prediction_lower = prediction.lower().strip()
    
    # Find the ground truth with the highest similarity
    max_similarity = 0.0
    for gt in ground_truths:
        gt_lower = gt.lower().strip()
        
        # Simple similarity metric (Jaccard index as implementation)
        pred_tokens = set(prediction_lower.split())
        gt_tokens = set(gt_lower.split())
        
        intersection = len(pred_tokens.intersection(gt_tokens))
        union = len(pred_tokens.union(gt_tokens))
        
        if union == 0:
            similarity = 1.0 if prediction_lower == gt_lower else 0.0
        else:
            similarity = intersection / union
            
        if similarity > max_similarity:
            max_similarity = similarity
            
    return max_similarity


def extract_answer(text: str) -> float:
    """
    Legacy function name - delegates to extract_numerical_answer for backward compatibility.
    
    Args:
        text: Text containing a numerical answer
        
    Returns:
        Extracted number or NaN if no number found
    """
    return extract_numerical_answer(text)


def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate pass@k for code generation tasks.
    
    Args:
        n: Total number of samples generated.
        c: Number of correct samples.
        k: The "k" in pass@k.
        
    Returns:
        The pass@k score.
    """
    if n - c < k:
        return 1.0
        
    return 1.0 - _comb(n - c, k) / _comb(n, k)


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Alternative name for calculate_pass_at_k for compatibility.
    """
    return calculate_pass_at_k(n, c, k)


def calculate_accuracy(correct: int, total: int) -> float:
    """Calculate simple accuracy metric."""
    if total == 0:
        return 0.0
    return correct / total


def calculate_f1_score(precision: float, recall: float) -> float:
    """Calculate F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def calculate_confidence_interval(accuracy: float, n_samples: int, confidence: float = 0.95) -> tuple:
    """
    Calculate confidence interval for accuracy using normal approximation.
    
    Args:
        accuracy: Observed accuracy (0-1)
        n_samples: Number of samples
        confidence: Confidence level (default 0.95)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    import scipy.stats as stats
    
    # Use Wilson score interval for better performance with small samples
    z = stats.norm.ppf((1 + confidence) / 2)
    
    # Wilson score interval
    denominator = 1 + z**2 / n_samples
    centre_adjusted_probability = accuracy + z**2 / (2 * n_samples)
    adjusted_standard_deviation = math.sqrt((accuracy * (1 - accuracy) + z**2 / (4 * n_samples)) / n_samples)
    
    lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator
    
    return (max(0, lower_bound), min(1, upper_bound))


def aggregate_results(all_run_results: List[dict]) -> dict:
    """
    Aggregate results from multiple runs with statistics.
    
    Args:
        all_run_results: List of result dictionaries from multiple runs
        
    Returns:
        Aggregated results with mean, std, and confidence intervals
    """
    if not all_run_results:
        return {}
    
    # Extract all unique model-task combinations
    aggregated = {}
    
    for run_results in all_run_results:
        for model_name, model_results in run_results.items():
            if model_name not in aggregated:
                aggregated[model_name] = {}
                
            for task_name, task_results in model_results.items():
                if task_name not in aggregated[model_name]:
                    aggregated[model_name][task_name] = {
                        'all_runs': [],
                        'metrics': set()
                    }
                
                aggregated[model_name][task_name]['all_runs'].append(task_results)
                
                # Track which metrics are available
                if 'overall' in task_results:
                    aggregated[model_name][task_name]['metrics'].update(task_results['overall'].keys())
    
    # Calculate statistics for each metric
    final_results = {}
    for model_name, model_data in aggregated.items():
        final_results[model_name] = {}
        
        for task_name, task_data in model_data.items():
            final_results[model_name][task_name] = {
                'overall': {},
                'statistics': {},
                'num_runs': len(task_data['all_runs'])
            }
            
            # Calculate stats for each metric
            for metric in task_data['metrics']:
                values = []
                for run_result in task_data['all_runs']:
                    if 'overall' in run_result and metric in run_result['overall']:
                        values.append(run_result['overall'][metric])
                
                if values:
                    mean_val = sum(values) / len(values)
                    std_val = math.sqrt(sum((x - mean_val)**2 for x in values) / len(values)) if len(values) > 1 else 0
                    
                    final_results[model_name][task_name]['overall'][metric] = mean_val
                    final_results[model_name][task_name]['statistics'][metric] = {
                        'mean': mean_val,
                        'std': std_val,
                        'min': min(values),
                        'max': max(values),
                        'values': values
                    }
                    
                    # Add confidence interval for accuracy-like metrics
                    if 'accuracy' in metric.lower() and len(values) > 1:
                        ci = calculate_confidence_interval(mean_val, len(values))
                        final_results[model_name][task_name]['statistics'][metric]['confidence_interval_95'] = ci
    
    return final_results


def _comb(n, k):
    """
    Calculate combinations (n choose k).
    Helper for pass@k calculation.
    """
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    if k > n // 2:
        k = n - k
        
    res = 1
    for i in range(k):
        res = res * (n - i) // (i + 1)
        
    return res