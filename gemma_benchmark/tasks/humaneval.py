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
from typing import Dict, Any, List, Optional, Set
from datasets import load_dataset, Dataset

# Import resource module only if not on Windows
if platform.system() != "Windows":
    import resource

# Core benchmark interfaces
from gemma_benchmark.core.interfaces import AbstractBenchmark, ModelInterface

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
        r'__import__\s*\(', r'eval\s*\(', r'exec\s*\(', r'compile\s*\(',
        r'open\s*\(', r'file\s*\(', r'input\s*\(', r'raw_input\s*\(',
        r'getattr\s*\(', r'setattr\s*\(', r'delattr\s*\(',
        r'globals\s*\(', r'locals\s*\(',
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            violations.append(f"Dangerous function call: {pattern}")

    return len(violations) == 0, violations


# --- HELPER FUNCTION FOR UNIX (with resource limits) ---
def _execute_code_unix(code: str, test_code: str, timeout: int) -> Dict[str, Any]:
    """Execute code on Unix-like systems with resource limits."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code + "\n" + test_code)
        temp_file = f.name

    try:
        # Prepare subprocess command with resource limits
        py_code = f"""
import sys
import signal
import resource

# Set resource limits
try:
    # Limit memory to 256MB
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
    # Limit CPU time
    resource.setrlimit(resource.RLIMIT_CPU, ({timeout}, {timeout}))
    # Limit number of processes to prevent fork bombs
    resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
except Exception as e:
    # Some environments might not allow setrlimit
    sys.stderr.write(f"Warning: Could not set resource limits: {{e}}\\n")

# Timeout handler
def timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm({timeout})

try:
    exec(open('{temp_file}').read(), {{'__name__': '__main__'}})
    sys.stdout.write("EXECUTION_SUCCESS")
except Exception as e:
    sys.stderr.write(f"EXECUTION_ERROR: {{e}}")
finally:
    signal.alarm(0)
"""
        cmd = [sys.executable, '-c', py_code]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout + 2
        )

        if result.returncode == 0 and "EXECUTION_SUCCESS" in result.stdout:
            return {"passed": True, "error": None}
        else:
            error_msg = result.stderr or "Unknown error during execution."
            return {"passed": False, "error": error_msg.strip()}

    except subprocess.TimeoutExpired:
        return {"passed": False, "error": "Execution timed out (subprocess)."}
    except Exception as e:
        return {"passed": False, "error": str(e)}
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


# --- HELPER FUNCTION FOR WINDOWS (without resource limits) ---
def _execute_code_windows(code: str, test_code: str, timeout: int) -> Dict[str, Any]:
    """Execute code on Windows with a simple timeout."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code + "\n" + test_code)
        temp_file = f.name

    try:
        cmd = [sys.executable, temp_file]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )

        if result.returncode == 0:
            return {"passed": True, "error": None}
        else:
            return {"passed": False, "error": result.stderr or "Execution failed with non-zero exit code."}

    except subprocess.TimeoutExpired:
        return {"passed": False, "error": "Execution timed out."}
    except Exception as e:
        return {"passed": False, "error": str(e)}
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)

def execute_code_safely(code: str, test_code: str, timeout: int = 10) -> Dict[str, Any]:
    """Execute code in a sandboxed environment with proper security measures."""
    # Security check first
    is_safe, violations = check_code_security(code + "\n" + test_code)
    if not is_safe:
        return {
            "passed": False,
            "error": f"Security violation: {'; '.join(violations)}",
            "security_violation": True
        }

    # Platform-specific execution
    if platform.system() == 'Windows':
        # Simplified execution for Windows without resource limits
        return _execute_code_windows(code, test_code, timeout)
    else:
        # Unix-like systems with full resource limits
        return _execute_code_unix(code, test_code, timeout)


class HumanEvalBenchmark(AbstractBenchmark):
    """Secure benchmark for the HumanEval dataset."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.k_values = self.config.get('k_values', [1, 10, 100])
        self.timeout = self.config.get('timeout', 10)
        # In HumanEval, num_samples_per_task is used to calculate pass@k
        self.num_samples_per_task = self.config.get('num_samples_per_task', 200)
        self.max_new_tokens = self.config.get('max_new_tokens', 256)
        self.temperature = self.config.get('temperature', 0.2)
        self.enable_security_checks = self.config.get('enable_security_checks', True)

        self.logger.info(f"Initialized HumanEval with security checks: {self.enable_security_checks}")

    def load_data(self) -> Dataset:
        """Load the HumanEval dataset."""
        self.logger.info("Loading HumanEval dataset...")
        dataset = load_dataset("openai_humaneval", split='test')
        return dataset

    def extract_code(self, prompt: str, response: str) -> str:
        """
        Extract Python code from a model's response.
        """
        # The model may return the full function including the prompt, or just the completion.
        # A common stop sequence for code generation is "\n\n", so we can use that.
        if "```" in response:
             match = re.search(r"```(?:python\n)?(.*?)```", response, re.DOTALL)
             if match:
                 return prompt + match.group(1)
        
        # Stop at the next function definition or common stop words
        stop_words = ["\ndef", "\nclass", "\nif __name__", "\nprint"]
        min_stop_idx = len(response)
        for word in stop_words:
            stop_idx = response.find(word)
            if stop_idx != -1:
                min_stop_idx = min(min_stop_idx, stop_idx)
        
        return prompt + response[:min_stop_idx]

    def test_code(self, code: str, test_code: str) -> bool:
        """Test if the generated code passes the test cases."""
        if self.enable_security_checks:
            result = execute_code_safely(code, test_code, self.timeout)
            if not result["passed"]:
                self.logger.debug(f"Code execution failed: {result['error']}")
            return result["passed"]
        else:
            # Unsafe execution (not recommended)
            try:
                exec(code + "\n" + test_code, globals())
                return True
            except Exception:
                return False

    def _evaluate_impl(self, model: ModelInterface) -> Dict[str, Any]:
        """Implementation-specific evaluation logic for HumanEval."""
        if self._data is None:
            self._data = self.load_data()

        total_problems = len(self._data)
        results_by_k = {k: 0 for k in self.k_values}
        
        self.logger.info(f"Evaluating {total_problems} HumanEval problems...")

        for i, problem in enumerate(self._data):
            prompt = problem['prompt']
            test_code = problem['test'] + f"\n\ncheck({problem['entry_point']})"
            
            samples_passed = 0
            # Generate n samples for each problem for pass@k
            for _ in range(self.num_samples_per_task):
                response = model.generate(
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                )
                generated_code = self.extract_code(prompt, response)
                if self.test_code(generated_code, test_code):
                    samples_passed += 1

            # Calculate pass@k for this problem and add to total
            for k in self.k_values:
                results_by_k[k] += pass_at_k(self.num_samples_per_task, samples_passed, k)
            
            if (i + 1) % 10 == 0:
                self.logger.info(f"Completed {i + 1}/{total_problems} problems")

        # Average the pass@k scores across all problems
        final_pass_at_k = {f"pass_at_{k}": (score / total_problems) for k, score in results_by_k.items()}
        
        self.logger.info(f"HumanEval Pass@k results: {final_pass_at_k}")
        
        return {"overall": final_pass_at_k}