"""
GSM8K (Grade School Math 8K) benchmark implementation.
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional
from datasets import load_dataset

from ..core.model_loader import ModelWrapper

class Gsm8kBenchmark:
    """
    Implementation of the GSM8K benchmark for evaluating mathematical reasoning.
    
    GSM8K contains grade school math word problems that require multi-step reasoning.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the GSM8K benchmark.
        
        Args:
            config: Configuration dictionary for the benchmark
        """
        self.logger = logging.getLogger("gemma_benchmark.tasks.gsm8k")
        self.config = config
        self.shot_count = config.get("shot_count", 5)
        self.use_cot = config.get("use_chain_of_thought", True)
        self.data = None
    
    def load_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load GSM8K dataset from HuggingFace.
        
        Returns:
            Dictionary containing train and test data
        """
        self.logger.info("Loading GSM8K data from HuggingFace...")
        
        try:
            # Load the GSM8K dataset
            dataset = load_dataset("gsm8k", "main")
            
            data = {
                "train": [],
                "test": []
            }
            
            # Process training data for few-shot examples
            for item in dataset["train"]:
                data["train"].append({
                    "question": item["question"],
                    "answer": item["answer"],
                    "numerical_answer": self._extract_numerical_answer(item["answer"])
                })
            
            # Process test data
            for item in dataset["test"]:
                data["test"].append({
                    "question": item["question"],
                    "answer": item["answer"], 
                    "numerical_answer": self._extract_numerical_answer(item["answer"])
                })
            
            self.logger.info(f"Loaded {len(data['train'])} training examples and {len(data['test'])} test examples")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load GSM8K data: {e}")
            return self._create_mock_data()
    
    def _extract_numerical_answer(self, answer_text: str) -> float:
        """
        Extract the numerical answer from GSM8K answer text.
        
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
        
        # Fallback: look for numbers at the end
        numbers = re.findall(r"([0-9,]+(?:\.[0-9]+)?)", answer_text)
        if numbers:
            try:
                return float(numbers[-1].replace(",", ""))
            except ValueError:
                pass
        
        self.logger.warning(f"Could not extract numerical answer from: {answer_text[:100]}...")
        return float('nan')
    
    def _create_mock_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Create mock GSM8K data for demonstration."""
        self.logger.info("Creating mock GSM8K data...")
        
        mock_problems = [
            {
                "question": "Janet has 22 ducks. 9 ducks lay 1 egg each day. Janet collects all the eggs and eats 3 for breakfast. How many eggs does Janet have left?",
                "answer": "Janet has 22 ducks. 9 ducks lay 1 egg each day, so she gets 9 eggs per day. She eats 3 eggs for breakfast, so she has 9 - 3 = 6 eggs left. #### 6",
                "numerical_answer": 6.0
            },
            {
                "question": "Tom bought 12 books. Each book cost $8. How much did Tom spend in total?",
                "answer": "Tom bought 12 books at $8 each. The total cost is 12 × 8 = 96 dollars. #### 96",
                "numerical_answer": 96.0
            },
            {
                "question": "A school has 3 classes with 25 students each. How many students are there in total?",
                "answer": "There are 3 classes with 25 students each. Total students = 3 × 25 = 75. #### 75", 
                "numerical_answer": 75.0
            }
        ]
        
        return {
            "train": mock_problems * 3,  # More examples for few-shot
            "test": mock_problems
        }
    
    def format_prompt(self, question: str, examples: List[Dict[str, Any]] = None) -> str:
        """
        Format a prompt for GSM8K evaluation.
        
        Args:
            question: The math problem to solve
            examples: Few-shot examples (optional)
            
        Returns:
            Formatted prompt string
        """
        if self.use_cot:
            prompt = "Solve the following math problems step by step.\n\n"
        else:
            prompt = "Solve the following math problems.\n\n"
        
        # Add few-shot examples
        if examples:
            for i, example in enumerate(examples[:self.shot_count]):
                prompt += f"Question: {example['question']}\n"
                if self.use_cot:
                    prompt += f"Answer: {example['answer']}\n\n"
                else:
                    prompt += f"Answer: {example['numerical_answer']}\n\n"
        
        # Add the target question
        prompt += f"Question: {question}\n"
        prompt += "Answer:"
        
        return prompt
    
    def evaluate(self, model: ModelWrapper, num_samples: int = 100) -> Dict[str, Any]:
        """
        Evaluate the model on GSM8K benchmark.
        
        Args:
            model: The model to evaluate
            num_samples: Number of test samples to evaluate (for faster testing)
            
        Returns:
            Dictionary containing evaluation results
        """
        if self.data is None:
            self.data = self.load_data()
        
        self.logger.info(f"Evaluating GSM8K with {num_samples} samples")
        
        # Get few-shot examples from training set
        examples = self.data["train"][:self.shot_count] if self.shot_count > 0 else []
        
        # Get test samples
        test_samples = self.data["test"][:num_samples]
        
        correct = 0
        total = len(test_samples)
        failed_examples = []
        
        for i, example in enumerate(test_samples):
            if (i + 1) % 20 == 0:
                self.logger.info(f"Progress: {i + 1}/{total}")
            
            prompt = self.format_prompt(example["question"], examples)
            
            # Get model response
            try:
                response = model.generate(prompt, max_new_tokens=300, temperature=0.0)
                
                # Extract numerical answer from response
                predicted_answer = self._extract_numerical_answer(response)
                expected_answer = example["numerical_answer"]
                
                # Check if answers match (with small tolerance for floating point)
                if not (isinstance(predicted_answer, float) and isinstance(expected_answer, float)):
                    is_correct = False
                elif abs(predicted_answer - expected_answer) < 1e-6:
                    is_correct = True
                else:
                    is_correct = False
                
                if is_correct:
                    correct += 1
                else:
                    # Store failed example for analysis
                    if len(failed_examples) < 10:  # Limit stored examples
                        failed_examples.append({
                            "question": example["question"],
                            "expected": expected_answer,
                            "predicted": predicted_answer,
                            "response": response[:200] + "..." if len(response) > 200 else response
                        })
                
            except Exception as e:
                self.logger.error(f"Error processing example {i}: {e}")
                failed_examples.append({
                    "question": example["question"],
                    "expected": example["numerical_answer"],
                    "predicted": "ERROR",
                    "response": str(e)
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
                "use_chain_of_thought": self.use_cot,
                "num_samples": num_samples
            },
            "failed_examples": failed_examples[:5]  # Include a few for analysis
        }
        
        self.logger.info(f"GSM8K evaluation complete. Accuracy: {accuracy:.4f} ({correct}/{total})")
        return results