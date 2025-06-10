"""
HumanEval benchmark implementation for code generation evaluation.
"""

import logging
import subprocess
import tempfile
import os
import sys
import platform
from typing import Dict, List, Any, Optional
from datasets import load_dataset
from contextlib import contextmanager
import threading
import multiprocessing

from ..core.model_loader import ModelWrapper


class TimeoutError(Exception):
    """Custom timeout error for cross-platform compatibility."""
    pass


class HumanevalBenchmark:
    """
    Implementation of the HumanEval benchmark for evaluating code generation.
    
    HumanEval contains Python programming problems with function signatures,
    docstrings, and test cases.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the HumanEval benchmark.
        
        Args:
            config: Configuration dictionary for the benchmark
        """
        self.logger = logging.getLogger("gemma_benchmark.tasks.humaneval")
        self.config = config
        self.timeout = config.get("timeout", 10)  # seconds per test
        self.temperature = config.get("temperature", 0.2)
        self.max_new_tokens = config.get("max_new_tokens", 256)
        self.data = None
        
        # Platform-specific setup
        self.is_windows = platform.system() == "Windows"
        if self.is_windows:
            self.logger.info("Running on Windows - using threading for timeout")
    
    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load HumanEval dataset from HuggingFace.
        
        Returns:
            List of programming problems
        """
        self.logger.info("Loading HumanEval data from HuggingFace...")
        
        try:
            # Load the HumanEval dataset
            dataset = load_dataset("openai_humaneval")
            
            data = []
            for item in dataset["test"]:
                data.append({
                    "task_id": item["task_id"],
                    "prompt": item["prompt"],
                    "canonical_solution": item["canonical_solution"],
                    "test": item["test"],
                    "entry_point": item["entry_point"]
                })
            
            self.logger.info(f"Loaded {len(data)} HumanEval problems")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load HumanEval data: {e}")
            return self._create_mock_data()
    
    def _create_mock_data(self) -> List[Dict[str, Any]]:
        """Create mock HumanEval data for demonstration."""
        self.logger.info("Creating mock HumanEval data...")
        
        return [
            {
                "task_id": "HumanEval/0",
                "prompt": "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
                "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n",
                "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n\ncheck(has_close_elements)",
                "entry_point": "has_close_elements"
            },
            {
                "task_id": "HumanEval/1", 
                "prompt": "def separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n",
                "canonical_solution": "    result = []\n    current_string = []\n    current_depth = 0\n\n    for c in paren_string:\n        if c == '(':\n            current_depth += 1\n            current_string.append(c)\n        elif c == ')':\n            current_depth -= 1\n            current_string.append(c)\n\n            if current_depth == 0:\n                result.append(''.join(current_string))\n                current_string = []\n\n    return result\n",
                "test": "def check(candidate):\n    assert candidate('(()()) ((())) () ((())()())') == [\n        '(()())', '((()))', '()', '((())()())'\n    ]\n    assert candidate('() (()) ((())) (((())))') == [\n        '()', '(())', '((()))', '(((())))'\n    ]\n    assert candidate('(()(())((())))') == [\n        '(()(())((())))'\n    ]\n    assert candidate('( ) (( )) (( )( ))') == ['()', '(())', '(()())']\n\ncheck(separate_paren_groups)",
                "entry_point": "separate_paren_groups"
            }
        ]
    
    @contextmanager
    def time_limit(self, seconds):
        """Context manager for executing code with a time limit (cross-platform)."""
        if self.is_windows:
            # Windows doesn't have SIGALRM, use threading approach
            timer = None
            timed_out = threading.Event()
            
            def timeout_handler():
                timed_out.set()
            
            timer = threading.Timer(seconds, timeout_handler)
            timer.start()
            
            try:
                yield timed_out
            finally:
                if timer:
                    timer.cancel()
        else:
            # Unix-based systems - use signal
            import signal
            
            def signal_handler(signum, frame):
                raise TimeoutError("Code execution timed out")
            
            # Set the signal handler and alarm
            old_handler = signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(seconds)
            
            try:
                yield None
            finally:
                # Restore the old signal handler and cancel the alarm
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
    
    def extract_code(self, response: str, prompt: str) -> str:
        """
        Extract the function implementation from the model's response.
        
        Args:
            response: The model's generated response
            prompt: The original prompt containing function signature
            
        Returns:
            The complete function implementation
        """
        # Combine prompt and response to get complete function
        full_code = prompt + response
        
        # Clean up common issues
        lines = full_code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove markdown code blocks
            if line.strip().startswith('```'):
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def test_code(self, code: str, test_code: str) -> bool:
        """
        Test if the generated code passes the test cases.
        
        Args:
            code: The complete function implementation
            test_code: The test code to run
            
        Returns:
            True if all tests pass, False otherwise
        """
        # Security check - don't run potentially dangerous code
        dangerous_patterns = [
            'import os', 'import subprocess', 'import sys', '__import__',
            'eval(', 'exec(', 'compile(', 'open(', 'file(',
            'input(', 'raw_input(', 'globals(', 'locals('
        ]
        
        code_lower = code.lower()
        for pattern in dangerous_patterns:
            if pattern in code_lower:
                self.logger.warning(f"Potentially dangerous code pattern detected: {pattern}")
                return False
        
        try:
            # Create a temporary file with the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # Add necessary imports
                f.write("from typing import List, Dict, Tuple, Optional, Any, Union, Set\n")
                f.write("import math\n")
                f.write("import re\n")
                f.write("import collections\n")
                f.write("import itertools\n\n")
                f.write(code)
                f.write("\n\n")
                f.write(test_code)
                temp_file = f.name
            
            try:
                if self.is_windows:
                    # Windows timeout handling with threading
                    with self.time_limit(self.timeout) as timed_out:
                        # Run the code
                        process = subprocess.Popen(
                            [sys.executable, temp_file],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True
                        )
                        
                        # Wait for completion or timeout
                        try:
                            stdout, stderr = process.communicate(timeout=self.timeout)
                            returncode = process.returncode
                        except subprocess.TimeoutExpired:
                            process.kill()
                            stdout, stderr = process.communicate()
                            self.logger.debug("Code execution timed out (subprocess)")
                            return False
                        
                        # Check if timed out via threading
                        if timed_out and timed_out.is_set():
                            self.logger.debug("Code execution timed out (threading)")
                            return False
                        
                        # Check if execution was successful
                        if returncode == 0:
                            return True
                        else:
                            self.logger.debug(f"Code execution failed: {stderr}")
                            return False
                else:
                    # Unix timeout handling with signal
                    with self.time_limit(self.timeout):
                        result = subprocess.run(
                            [sys.executable, temp_file],
                            capture_output=True,
                            text=True,
                            timeout=self.timeout
                        )
                        
                        # Check if execution was successful
                        if result.returncode == 0:
                            return True
                        else:
                            self.logger.debug(f"Code execution failed: {result.stderr}")
                            return False
                    
            except (subprocess.TimeoutExpired, TimeoutError):
                self.logger.debug("Code execution timed out")
                return False
            except Exception as e:
                self.logger.debug(f"Error during code execution: {e}")
                return False
            
        except Exception as e:
            self.logger.debug(f"Error testing code: {e}")
            return False
        
        finally:
            # Clean up temporary file
            try:
                if 'temp_file' in locals():
                    os.unlink(temp_file)
            except:
                pass
    
    def evaluate(self, model: ModelWrapper, num_samples: int = None) -> Dict[str, Any]:
        """
        Evaluate the model on HumanEval benchmark.
        
        Args:
            model: The model to evaluate
            num_samples: Number of problems to evaluate (None for all)
            
        Returns:
            Dictionary containing evaluation results
        """
        if self.data is None:
            self.data = self.load_data()
        
        # Use num_samples from config if not provided
        if num_samples is None:
            num_samples = self.config.get("num_samples", len(self.data))
        
        # Limit to available data
        num_samples = min(num_samples, len(self.data))
        
        self.logger.info(f"Evaluating HumanEval with {num_samples} problems")
        
        # Get test samples
        test_samples = self.data[:num_samples]
        
        passed = 0
        total = len(test_samples)
        failed_examples = []
        
        for i, problem in enumerate(test_samples):
            if (i + 1) % 10 == 0:
                self.logger.info(f"Progress: {i + 1}/{total}")
            
            try:
                # Generate code
                response = model.generate(
                    problem["prompt"], 
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature
                )
                
                # Extract complete function
                complete_code = self.extract_code(response, problem["prompt"])
                
                # Test the code
                is_correct = self.test_code(complete_code, problem["test"])
                
                if is_correct:
                    passed += 1
                else:
                    # Store failed example for analysis
                    if len(failed_examples) < 5:
                        failed_examples.append({
                            "task_id": problem["task_id"],
                            "prompt": problem["prompt"][:200] + "..." if len(problem["prompt"]) > 200 else problem["prompt"],
                            "generated_code": response[:200] + "..." if len(response) > 200 else response,
                            "issue": "Failed test cases"
                        })
                
            except Exception as e:
                self.logger.error(f"Error processing problem {problem['task_id']}: {e}")
                if len(failed_examples) < 5:
                    failed_examples.append({
                        "task_id": problem["task_id"],
                        "prompt": problem["prompt"][:200] + "..." if len(problem["prompt"]) > 200 else problem["prompt"],
                        "generated_code": "ERROR",
                        "issue": str(e)
                    })
        
        pass_at_1 = passed / total if total > 0 else 0.0
        
        results = {
            "overall": {
                "passed": passed,
                "total": total,
                "pass_at_1": pass_at_1
            },
            "config": {
                "temperature": self.temperature,
                "max_new_tokens": self.max_new_tokens,
                "timeout": self.timeout,
                "num_samples": num_samples
            },
            "failed_examples": failed_examples
        }
        
        self.logger.info(f"HumanEval evaluation complete. Pass@1: {pass_at_1:.4f} ({passed}/{total})")
        return results