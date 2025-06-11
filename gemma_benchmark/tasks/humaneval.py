"""
Benchmark for the HumanEval dataset for code generation.
"""

import os
import re
import json
import gzip
import logging
import multiprocessing
import platform
import resource
from typing import Dict, Any, List, Optional
from datasets import load_dataset, Dataset

# Core benchmark interfaces
from gemma_benchmark.core.interfaces import AbstractBenchmark, ModelInterface, BenchmarkResult

# Utilities for metrics
from gemma_benchmark.utils.metrics import pass_at_k


def check_correctness(prompt: str, completion: str, timeout: int) -> Dict[str, Any]:
    """
    Evaluates the correctness of a completion for a HumanEval problem.
    This function is designed to be run in a separate process.
    """
    
    def _run_test(code: str, test_code: str) -> bool:
        """Executes the provided code and its test."""
        # Note: This is a simplified execution environment.
        # A more robust solution would use a sandboxed environment.
        exec_globals = {}
        try:
            exec(code + "\n" + test_code, exec_globals)
            return True
        except Exception:
            return False
            
    # Combine prompt (which is the function header) with the completion
    full_code = prompt + completion
    
    # The test is part of the prompt in HumanEval format
    # It usually follows the function signature and docstring
    test_code = ""
    # Simplified extraction of the test code from the prompt
    # In HumanEval, this is often 'check(function_name)'
    if "check(" in prompt:
        # This is a placeholder for a more robust test extraction logic
        # For humaneval dataset, the test is provided with the problem
        pass 

    # We need to extract the test from the original dataset item, not the prompt.
    # This function should receive the test code as an argument.
    # For now, this is a simplified simulation. A complete implementation
    # would pass the 'test' field from the dataset.

    # This part needs to be improved in a real implementation
    # where the test code is passed explicitly.
    # Let's assume for now the test code is available globally.
    
    # A placeholder for the actual test execution logic
    # In a real scenario, you'd pass the test from the humaneval dataset
    # and execute it here.
    
    return {"passed": False, "result": "skipped"}


class HumanEvalBenchmark(AbstractBenchmark):
    """Benchmark for the HumanEval dataset."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.k_values = self.config.get('k_values', [1, 10, 100])
        self.timeout = self.config.get('timeout', 10)
        self.num_samples_per_task = max(self.k_values)

    def load_data(self) -> Dataset:
        """Load the HumanEval dataset."""
        self.logger.info("Loading HumanEval dataset...")
        # The HumanEval dataset is distributed as a .jsonl.gz file
        # We need a robust way to load it, possibly via a download script
        # For now, let's assume it's loaded via HuggingFace datasets
        try:
            dataset = load_dataset("openai_humaneval")
            return dataset['test'] # Use the test split
        except Exception as e:
            self.logger.error(f"Failed to load HumanEval dataset: {e}")
            self.logger.info("Please ensure 'openai_humaneval' is accessible or manually downloaded.")
            raise
    
    def extract_code(self, prompt: str, response: str) -> str:
        """
        Extracts Python code from a model's response, handling various formats like
        markdown blocks and ensuring the final output is a complete function.
        """
        # Case 1: Look for a python markdown block
        match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        if match:
            code = match.group(1)
            # If the extracted code is a full function, use it.
            if code.strip().startswith("def "):
                return code
            # Otherwise, assume it's the function body and append it to the prompt.
            else:
                return prompt + code

        # Case 2: Look for a generic markdown block if a python one isn't found
        match = re.search(r"```(.*?)```", response, re.DOTALL)
        if match:
            code = match.group(1)
            # If the model regenerates the function signature, use its version.
            if code.strip().startswith("def "):
                return code
            # Otherwise, append the assumed code body to the prompt.
            else:
                return prompt + code

        # Case 3: No markdown. Assume the response is the completion.
        # Truncate the response at common "stop" words that indicate the end of code.
        stop_words = ["\ndef", "\nclass", "\nif __name__", "\nprint"]
        min_stop_idx = len(response)
        for word in stop_words:
            stop_idx = response.find(word)
            if stop_idx != -1:
                min_stop_idx = min(min_stop_idx, stop_idx)
        
        # The final code is the prompt plus the (potentially truncated) response.
        return prompt + response

    def _evaluate_impl(self, model: ModelInterface) -> Dict[str, Any]:
        """Implementation-specific evaluation logic for HumanEval."""
        if self._data is None:
            self._data = self.load_data()
        
        # This is a simplified evaluation loop
        # A full implementation would use the official HumanEval evaluation script
        # which handles multiprocessing and secure execution.
        
        total_problems = len(self._data)
        total_passed = 0
        
        # Simplified loop for demonstration
        for i, problem in enumerate(self._data):
            prompt = problem['prompt']
            
            # For pass@k, we need to generate multiple samples
            # This is computationally intensive and should be done in parallel
            # For this example, we generate one sample (equivalent to pass@1)
            response = model.generate(
                prompt,
                max_new_tokens=256,
                temperature=0.2, # Recommended for HumanEval
                do_sample=True,
            )
            
            # Extract the code from the response
            generated_code = self.extract_code(prompt, response)
            
            # In a real implementation, you would run the 'test' code
            # from the dataset against the generated code in a sandbox.
            # This is a placeholder for that logic.
            # For now, we'll just check if the code is valid Python.
            try:
                compile(generated_code, '<string>', 'exec')
                # This doesn't check correctness, just syntax.
                # A full implementation requires running the problem['test']
                # total_passed += 1 # This would be based on test results
            except SyntaxError:
                pass # Code is invalid
        
        # The pass@k calculation is complex and requires multiple samples.
        # We'll return a placeholder result here.
        self.logger.warning("HumanEval evaluation is simplified. Results are for demonstration only.")
        
        return {
            "overall": {
                "pass_at_1": (total_passed / total_problems) * 100 if total_problems > 0 else 0,
                "pass_at_10": "Not Implemented",
                "pass_at_100": "Not Implemented",
            },
            "details": f"Evaluated {total_problems} problems. Correctness check is a placeholder."
        }