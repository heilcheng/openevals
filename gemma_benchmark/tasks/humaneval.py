"""
Benchmark for the HumanEval dataset for code generation.
"""

import os
import re
import json
import logging
import tempfile
import subprocess
import platform
import resource
from typing import Dict, Any, List, Optional, Set
from datasets import load_dataset, Dataset

# Core benchmark interfaces
from gemma_benchmark.core.interfaces import AbstractBenchmark, ModelInterface, BenchmarkResult

# Utilities for metrics
from gemma_benchmark.utils.metrics import pass_at_k


# Security configuration
DANGEROUS_IMPORTS = {
    'os', 'subprocess', 'sys', 'shutil', 'socket', 'urllib', 'requests',
    'pickle', 'marshal', 'shelve', 'dbm', 'sqlite3', 'ctypes', 'gc',
    'importlib', 'runpy', 'code', 'codeop', 'compile', 'eval', 'exec',
    '__import__', 'open', 'file', 'input', 'raw_input'
}

ALLOWED_IMPORTS = {
    'math', 'random', 'string', 'itertools', 'functools', 'operator',
    'collections', 'heapq', 'bisect', 'array', 'copy', 'pprint',
    'reprlib', 'enum', 'datetime', 'calendar', 'time', 'json', 're',
    'difflib', 'textwrap', 'unicodedata', 'stringprep', 'readline',
    'rlcompleter', 'struct', 'codecs', 'types', 'weakref', 'abc',
    'numbers', 'cmath', 'decimal', 'fractions', 'statistics'
}


def check_code_security(code: str) -> tuple[bool, List[str]]:
    """
    Check if code contains dangerous patterns.
    
    Args:
        code: Python code to check
        
    Returns:
        Tuple of (is_safe, list_of_violations)
    """
    violations = []
    
    # Check for dangerous imports
    import_pattern = r'(?:^|\n)\s*(?:import|from)\s+(\w+)'
    imports = re.findall(import_pattern, code, re.MULTILINE)
    
    for imp in imports:
        if imp in DANGEROUS_IMPORTS:
            violations.append(f"Dangerous import: {imp}")
        elif imp not in ALLOWED_IMPORTS:
            violations.append(f"Disallowed import: {imp}")
    
    # Check for dangerous function calls
    dangerous_patterns = [
        r'__import__\s*\(',
        r'eval\s*\(',
        r'exec\s*\(',
        r'compile\s*\(',
        r'open\s*\(',
        r'file\s*\(',
        r'input\s*\(',
        r'raw_input\s*\(',
        r'getattr\s*\(',
        r'setattr\s*\(',
        r'delattr\s*\(',
        r'hasattr\s*\(',
        r'globals\s*\(',
        r'locals\s*\(',
        r'vars\s*\(',
        r'dir\s*\(',
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            violations.append(f"Dangerous function call: {pattern}")
    
    # Check for file operations
    file_patterns = [
        r'\.read\s*\(',
        r'\.write\s*\(',
        r'\.open\s*\(',
        r'\.close\s*\(',
    ]
    
    for pattern in file_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            violations.append(f"File operation: {pattern}")
    
    # Check for network operations
    network_patterns = [
        r'socket\.',
        r'urllib\.',
        r'http\.',
        r'requests\.',
    ]
    
    for pattern in network_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            violations.append(f"Network operation: {pattern}")
    
    return len(violations) == 0, violations


def execute_code_safely(code: str, test_code: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Execute code in a sandboxed environment with proper security measures.
    
    Args:
        code: The code to execute
        test_code: Test code to run
        timeout: Timeout in seconds
        
    Returns:
        Dictionary with execution results
    """
    # Security check first
    is_safe, violations = check_code_security(code + "\n" + test_code)
    if not is_safe:
        return {
            "passed": False,
            "error": f"Security violation: {'; '.join(violations)}",
            "security_violation": True
        }
    
    # Create temporary file for execution
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code + "\n" + test_code)
        temp_file = f.name
    
    try:
        # Prepare sandboxed execution environment
        env = os.environ.copy()
        env['PYTHONPATH'] = ''  # Clear Python path
        env['PYTHONHOME'] = ''  # Clear Python home
        
        # Create subprocess with restrictions
        cmd = [
            'python', '-c', f"""
import sys
import signal
import resource

# Set resource limits
try:
    # Limit memory to 128MB
    resource.setrlimit(resource.RLIMIT_AS, (128 * 1024 * 1024, 128 * 1024 * 1024))
    # Limit CPU time to {timeout} seconds
    resource.setrlimit(resource.RLIMIT_CPU, ({timeout}, {timeout}))
    # Limit file size to 1MB
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    # Limit number of processes
    resource.setrlimit(resource.RLIMIT_NPROC, (10, 10))
except:
    pass  # Windows doesn't support all limits

# Timeout handler
def timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm({timeout})

try:
    exec(open('{temp_file}').read())
    print("EXECUTION_SUCCESS")
except Exception as e:
    print(f"EXECUTION_ERROR: {{e}}")
finally:
    signal.alarm(0)
"""
        ]
        
        # Execute with timeout and resource limits
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 5,  # Give extra time for setup
            env=env
        )
        
        # Check results
        if result.returncode == 0 and "EXECUTION_SUCCESS" in result.stdout:
            return {"passed": True, "error": None}
        else:
            error_msg = result.stderr or result.stdout or "Unknown error"
            if "EXECUTION_ERROR:" in error_msg:
                error_msg = error_msg.split("EXECUTION_ERROR:", 1)[1].strip()
            return {"passed": False, "error": error_msg}
            
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": "Execution timeout"}
    except Exception as e:
        return {"passed": False, "error": str(e)}
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_file)
        except:
            pass


class HumanEvalBenchmark(AbstractBenchmark):
    """Secure benchmark for the HumanEval dataset."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.k_values = self.config.get('k_values', [1, 10, 100])
        self.timeout = self.config.get('timeout', 10)
        self.num_samples_per_task = max(self.k_values)
        self.max_new_tokens = self.config.get('max_new_tokens', 256)
        self.temperature = self.config.get('temperature', 0.2)
        
        # Security settings
        self.enable_security_checks = self.config.get('enable_security_checks', True)
        self.allowed_imports = set(self.config.get('allowed_imports', ALLOWED_IMPORTS))
        
        self.logger.info(f"Initialized HumanEval with security checks: {self.enable_security_checks}")

    def load_data(self) -> Dataset:
        """Load the HumanEval dataset."""
        self.logger.info("Loading HumanEval dataset...")
        try:
            dataset = load_dataset("openai_humaneval")
            return dataset['test']
        except Exception as e:
            self.logger.error(f"Failed to load HumanEval dataset: {e}")
            # Create minimal mock data for testing
            return Dataset.from_list([
                {
                    "task_id": "HumanEval/0",
                    "prompt": "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
                    "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.0], 0.5) == False\n    assert candidate([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\ncheck(has_close_elements)"
                }
            ])
    
    def extract_code(self, prompt: str, response: str) -> str:
        """
        Extract Python code from a model's response, handling various formats.
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
        stop_words = ["\ndef", "\nclass", "\nif __name__", "\nprint", "```"]
        min_stop_idx = len(response)
        for word in stop_words:
            stop_idx = response.find(word)
            if stop_idx != -1:
                min_stop_idx = min(min_stop_idx, stop_idx)
        
        # The final code is the prompt plus the (potentially truncated) response.
        return prompt + response[:min_stop_idx]

    def test_code(self, code: str, test_code: str) -> bool:
        """
        Test if the generated code passes the test cases.
        
        Args:
            code: Generated code
            test_code: Test cases
            
        Returns:
            True if all tests pass, False otherwise
        """
        if self.enable_security_checks:
            return execute_code_safely(code, test_code, self.timeout)["passed"]
        else:
            # Fallback to simple execution (not recommended for production)
            self.logger.warning("Security checks disabled - using unsafe execution")
            try:
                exec_globals = {}
                exec(code + "\n" + test_code, exec_globals)
                return True
            except Exception as e:
                self.logger.debug(f"Code execution failed: {e}")
                return False

    def _evaluate_impl(self, model: ModelInterface) -> Dict[str, Any]:
        """Implementation-specific evaluation logic for HumanEval."""
        if self._data is None:
            self._data = self.load_data()
            
        total_problems = len(self._data)
        results_by_k = {k: 0 for k in self.k_values}
        failed_examples = []
        
        self.logger.info(f"Evaluating {total_problems} HumanEval problems...")
        
        for i, problem in enumerate(self._data):
            if i >= self.num_samples_per_task:
                break
                
            prompt = problem['prompt']
            test_code = problem['test']
            task_id = problem.get('task_id', f'task_{i}')
            
            # Generate multiple samples for pass@k evaluation
            samples_passed = 0
            generation_errors = 0
            
            for sample_idx in range(max(self.k_values)):
                try:
                    # Generate code completion
                    response = model.generate(
                        prompt,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        do_sample=True,
                    )
                    
                    # Extract the code from the response
                    generated_code = self.extract_code(prompt, response)
                    
                    # Test the code
                    if self.test_code(generated_code, test_code):
                        samples_passed += 1
                        
                except Exception as e:
                    generation_errors += 1
                    self.logger.debug(f"Generation error for {task_id}, sample {sample_idx}: {e}")
                    
                    if len(failed_examples) < 5:
                        failed_examples.append({
                            "task_id": task_id,
                            "sample_idx": sample_idx,
                            "error": str(e),
                            "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt
                        })
            
            # Calculate pass@k for this problem
            total_samples = max(self.k_values) - generation_errors
            if total_samples > 0:
                for k in self.k_values:
                    if k <= total_samples:
                        pass_k_score = pass_at_k(total_samples, samples_passed, k)
                        results_by_k[k] += pass_k_score
            
            if (i + 1) % 10 == 0:
                self.logger.info(f"Completed {i + 1}/{total_problems} problems")
        
        # Calculate final averages
        final_results = {}
        problems_evaluated = min(total_problems, self.num_samples_per_task)
        
        for k in self.k_values:
            if problems_evaluated > 0:
                final_results[f'pass_at_{k}'] = results_by_k[k] / problems_evaluated
            else:
                final_results[f'pass_at_{k}'] = 0.0
        
        self.logger.info(f"HumanEval evaluation complete. Pass@1: {final_results.get('pass_at_1', 0):.3f}")
        
        return {
            "overall": final_results,
            "details": {
                "problems_evaluated": problems_evaluated,
                "total_problems": total_problems,
                "security_enabled": self.enable_security_checks,
                "failed_examples": failed_examples[:5]  # Limit to first 5 failures
            }
        }