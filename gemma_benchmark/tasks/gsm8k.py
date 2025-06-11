"""
GSM8K (Grade School Math 8K) benchmark implementation.
"""

import logging
import re
from typing import Dict, List, Any, Optional
from datasets import load_dataset
import random

from ..core.model_loader import ModelWrapper

class Gsm8kBenchmark:
    """
    Implementation of the GSM8K benchmark.
    
    GSM8K is a dataset of 8.5K high quality linguistically diverse grade school math word problems.
    The dataset is designed to evaluate the mathematical reasoning capabilities of language models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the GSM8K benchmark.
        
        Args:
            config: Configuration dictionary for the benchmark
        """
        self.logger = logging.getLogger("gemma_benchmark.tasks.gsm8k")
        self.config = config
        self.shot_count = config.get("shot_count", 8)
        self.use_cot = config.get("use_chain_of_thought", True)
        self.data = None
    
    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load GSM8K dataset from HuggingFace.
        
        Returns:
            List of GSM8K questions and answers
        """
        self.logger.info("Loading GSM8K data from HuggingFace...")
        
        try:
            dataset = load_dataset("gsm8k", "main")
            
            # Process train and test sets
            train_data = list(dataset["train"])
            test_data = list(dataset["test"])
            
            # Extract numerical answers for few-shot examples
            for item in train_data:
                item["numerical_answer"] = self._extract_numerical_answer(item["answer"])
            
            # Attach few-shot examples to test data
            for item in test_data:
                item["examples"] = random.sample(train_data, self.shot_count)
            
            self.logger.info(f"Loaded {len(test_data)} test examples and {len(train_data)} training examples.")
            return test_data
            
        except Exception as e:
            self.logger.error(f"Failed to load GSM8K data: {e}")
            return [] # Return empty list on failure
            
    def _extract_numerical_answer(self, answer_text: str) -> float:
        """
        Extract the numerical answer from GSM8K answer text with improved logic.
        
        This improved version:
        - Prioritizes numbers after #### markers
        - Looks for numbers after answer indicators
        - Falls back to last number only if no clear answer markers found
        
        Args:
            answer_text: The full answer text containing reasoning and final answer
            
        Returns:
            The numerical answer as a float
        """
        # GSM8K answers typically end with "#### [number]"
        pattern = r"####\s*([0-9,]+(?:\.[0-9]+)?)"
        match = re.search(pattern, answer_text)
        
        if match:
            # Remove commas and convert to float
            number_str = match.group(1).replace(",", "")
            try:
                return float(number_str)
            except ValueError:
                pass
        
        # Look for answer indicators followed by numbers
        answer_patterns = [
            r"answer is[:\s]+([0-9,]+(?:\.[0-9]+)?)",
            r"answer:[:\s]*([0-9,]+(?:\.[0-9]+)?)",
            r"= ([0-9,]+(?:\.[0-9]+)?)\s*$",  # Equals sign at end
            r"total[:\s]+([0-9,]+(?:\.[0-9]+)?)",
            r"result[:\s]+([0-9,]+(?:\.[0-9]+)?)",
            r"therefore[,\s]+([0-9,]+(?:\.[0-9]+)?)",
            r"so[,\s]+([0-9,]+(?:\.[0-9]+)?)\s*$"  # "So" at end
        ]
        
        for pattern in answer_patterns:
            matches = re.findall(pattern, answer_text.lower())
            if matches:
                # Take the last match (closest to end)
                try:
                    return float(matches[-1].replace(",", ""))
                except ValueError:
                    continue
        
        # Look for boxed answers (sometimes used in math)
        boxed_pattern = r"\\boxed\{([0-9,]+(?:\.[0-9]+)?)\}"
        boxed_match = re.search(boxed_pattern, answer_text)
        if boxed_match:
            try:
                return float(boxed_match.group(1).replace(",", ""))
            except ValueError:
                pass
        
        # Fallback: look for the last standalone number
        all_numbers = re.findall(r"([0-9,]+(?:\.[0-9]+)?)", answer_text)
        if all_numbers:
            try:
                return float(all_numbers[-1].replace(",", ""))
            except ValueError:
                pass
        
        self.logger.warning(f"Could not extract numerical answer from: {answer_text[:100]}...")
        return float('nan')

    def format_prompt(self, question: str, examples: List[Dict[str, Any]]) -> str:
        """Format a prompt for GSM8K evaluation."""
        prompt = "Solve the following grade school math problems.\n\n"
        
        for ex in examples:
            prompt += f"Question: {ex['question']}\n"
            if self.use_cot:
                # Chain-of-thought prompt with full reasoning
                prompt += f"Answer: {ex['answer']}\n\n"
            else:
                # Standard prompt with just the numerical answer
                prompt += f"Answer: {ex['numerical_answer']}\n\n"
        
        prompt += f"Question: {question}\nAnswer:"
        
        if self.use_cot:
            prompt += " (Let's think step by step)"
            
        return prompt

    def evaluate(self, model: ModelWrapper) -> Dict[str, Any]:
        """Evaluate the model on GSM8K."""
        if self.data is None:
            self.data = self.load_data()
            if not self.data:
                return {"error": "Failed to load data."}
        
        correct = 0
        total = len(self.data)
        failed_examples = []
        
        for item in self.data:
            question = item["question"]
            true_answer_text = item["answer"]
            true_numerical_answer = self._extract_numerical_answer(true_answer_text)
            
            prompt = self.format_prompt(question, item["examples"])
            
            try:
                response = model.generate(prompt, max_new_tokens=256)
                predicted_numerical_answer = self._extract_numerical_answer(response)
                
                # Check for correctness (allowing for float comparison issues)
                if abs(predicted_numerical_answer - true_numerical_answer) < 1e-5:
                    correct += 1
                else:
                    if len(failed_examples) < 10:
                        failed_examples.append({
                            "question": question,
                            "true_answer": true_numerical_answer,
                            "predicted_answer": predicted_numerical_answer,
                            "model_response": response
                        })

            except Exception as e:
                self.logger.error(f"Error evaluating example: {e}")
                if len(failed_examples) < 10:
                     failed_examples.append({
                        "question": question,
                        "error": str(e)
                    })

        accuracy = correct / total if total > 0 else 0.0
        
        results = {
            "overall": {
                "correct": correct,
                "total": total,
                "accuracy": accuracy
            },
            "config": {
                "shot_count": self.shot_count,
                "use_chain_of_thought": self.use_cot
            },
            "failed_examples": failed_examples
        }
        
        self.logger.info(f"GSM8K evaluation complete. Accuracy: {accuracy:.4f}")
        return results