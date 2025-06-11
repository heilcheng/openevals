"""
HumanEval benchmark implementation for evaluating code generation capabilities.
"""

import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
from typing import Dict, List, Any

from datasets import load_dataset

from ..core.model_loader import ModelWrapper
from ..utils.metrics import calculate_pass_at_k

class HumanevalBenchmark:
    """
    Implementation of the HumanEval benchmark for code generation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the HumanEval benchmark.
        
        Args:
            config: Configuration dictionary for the benchmark
        """
        self.logger = logging.getLogger("gemma_benchmark.tasks.humaneval")
        self.config = config
        self.data = None
        self.timeout = config.get("timeout", 10.0)
        self.is_windows = (os.name == 'nt')

    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load HumanEval dataset from HuggingFace.
        
        Returns:
            List of HumanEval problems
        """
        self.logger.info("Loading HumanEval data from HuggingFace...")
        
        try:
            dataset = load_dataset("openai_humaneval", split="test")
            data = list(dataset)
            self.logger.info(f"Loaded {len(data)} HumanEval examples")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load HumanEval data: {e}")
            return []

    def extract_code(self, prompt: str, response: str) -> str:
        """
        Extract the code from the model's response.
        
        Args:
            prompt: The full prompt including function signature
            response: The model's generated code
            
        Returns:
            The extracted code block
        """
        # A simple implementation: assume response completes the prompt
        return prompt + response

    def test_code(self, code: str, test_code: str) -> bool:
        """
        Test if the generated code passes the test cases with enhanced security.
        
        Uses subprocess with restricted permissions and resource limits for sandboxing.
        
        Args:
            code: The complete function implementation
            test_code: The test code to run
            
        Returns:
            True if all tests pass, False otherwise
        """
        # Basic security check - reject obviously malicious patterns
        dangerous_patterns = [
            '__import__', 'exec(', 'eval(', 'compile(', 'globals(', 'locals(',
            'open(', 'file(', 'input(', 'raw_input(', '__builtins__',
            'subprocess', 'os.system', 'commands.', 'popen'
        ]
        
        code_lower = code.lower()
        for pattern in dangerous_patterns:
            if pattern in code_lower:
                self.logger.warning(f"Potentially dangerous code pattern detected: {pattern}")
                return False
        
        # Whitelist of allowed imports for HumanEval
        allowed_imports = {
            'math', 're', 'collections', 'itertools', 'functools',
            'typing', 'string', 'datetime', 'random', 'bisect', 'heapq'
        }
        
        # Check 'import X' and 'from X import Y' style imports
        import_pattern = r'^\s*(?:import|from)\s+(\w+)'
        imports = re.findall(import_pattern, code, re.MULTILINE)
        for imp in imports:
            if imp not in allowed_imports:
                self.logger.warning(f"Disallowed import detected: {imp}")
                return False

        try:
            # Create a temporary file with the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write("from typing import List, Dict, Tuple, Optional, Any, Union, Set\n")
                f.write("import math\n")
                f.write(code)
                f.write("\n\n")
                f.write(test_code)
                temp_file = f.name
            
            try:
                # Prepare subprocess with security restrictions
                env = os.environ.copy()
                for var in ['LD_PRELOAD', 'LD_LIBRARY_PATH', 'PYTHONPATH']:
                    env.pop(var, None)
                env['PYTHONDONTWRITEBYTECODE'] = '1'
                
                if self.is_windows:
                    process = subprocess.Popen(
                        [sys.executable, '-E', '-s', temp_file],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                        creationflags=subprocess.CREATE_NO_WINDOW
                    )
                else:
                    # Unix-specific security enhancements
                    import resource
                    
                    def set_limits():
                        # Limit CPU time, memory, processes, and file operations
                        resource.setrlimit(resource.RLIMIT_CPU, (self.timeout, self.timeout))
                        resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
                        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
                        resource.setrlimit(resource.RLIMIT_NOFILE, (0, 0))
                    
                    process = subprocess.Popen(
                        [sys.executable, '-E', '-s', temp_file],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                        preexec_fn=set_limits
                    )
                
                # Wait for completion with timeout
                try:
                    stdout, stderr = process.communicate(timeout=self.timeout + 1)
                    returncode = process.returncode
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = process.communicate()
                    self.logger.debug("Code execution timed out")
                    return False
                
                # Check if execution was successful
                return returncode == 0 and "AssertionError" not in stderr

            except Exception as e:
                self.logger.debug(f"Error during code execution: {e}")
                return False
            
        except Exception as e:
            self.logger.debug(f"Error testing code: {e}")
            return False
            
        finally:
            # Clean up temporary file
            if 'temp_file' in locals() and os.path.exists(temp_file):
                os.unlink(temp_file)

    def evaluate(self, model: ModelWrapper, k_values: List[int] = [1, 10, 100]) -> Dict[str, Any]:
        """
        Evaluate the model on the HumanEval benchmark.
        
        Args:
            model: The model to evaluate
            k_values: List of k values for pass@k calculation
            
        Returns:
            Dictionary containing evaluation results
        """
        if self.data is None:
            self.data = self.load_data()
            if not self.data:
                return {"error": "Failed to load data."}

        results = {}
        for item in self.data:
            task_id = item["task_id"]
            prompt = item["prompt"]
            test_code = item["test"]
            
            num_samples = max(k_values)
            correct_count = 0
            
            # Generate multiple samples for pass@k
            for _ in range(num_samples):
                try:
                    response = model.generate(prompt, max_new_tokens=256, stop_sequences=["\n\n"])
                    generated_code = self.extract_code(prompt, response)
                    
                    if self.test_code(generated_code, test_code):
                        correct_count += 1
                except Exception as e:
                    self.logger.error(f"Error generating or testing for {task_id}: {e}")
            
            results[task_id] = {
                "n": num_samples,
                "c": correct_count
            }

        # Calculate pass@k for each k
        pass_at_k = {}
        total_problems = len(self.data)
        for k in k_values:
            pass_k_sum = 0
            for res in results.values():
                pass_k_sum += calculate_pass_at_k(res["n"], res["c"], k)
            pass_at_k[f"pass@{k}"] = pass_k_sum / total_problems if total_problems > 0 else 0.0

        return {
            "overall": pass_at_k,
            "config": {
                "k_values": k_values,
                "timeout": self.timeout
            },
            "per_task_results": results
        }