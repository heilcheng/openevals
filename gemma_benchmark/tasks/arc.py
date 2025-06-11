"""
ARC (AI2 Reasoning Challenge) benchmark implementation for science reasoning evaluation.
"""

import logging
import re
from typing import Dict, List, Any

from datasets import load_dataset

from ..core.model_loader import ModelWrapper

class ArcBenchmark:
    """
    Implementation of the ARC benchmark (AI2 Reasoning Challenge).
    
    This benchmark tests models on science questions from elementary to high school level.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ARC benchmark.
        
        Args:
            config: Configuration dictionary for the benchmark
        """
        self.logger = logging.getLogger("gemma_benchmark.tasks.arc")
        self.config = config
        self.subset = config.get("subset", "ARC-Challenge")  # "ARC-Easy" or "ARC-Challenge"
        self.data = None
        
    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load ARC dataset from HuggingFace.
        
        Returns:
            List of ARC questions and answers
        """
        self.logger.info(f"Loading ARC data for subset: {self.subset}")
        
        try:
            dataset = load_dataset("ai2_arc", self.subset)
            data = list(dataset["test"])
            self.logger.info(f"Loaded {len(data)} ARC examples for subset {self.subset}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load ARC data: {e}")
            return []

    def format_prompt(self, item: Dict[str, Any]) -> str:
        """
        Format a prompt for ARC evaluation.
        
        Args:
            item: A single data item from the ARC dataset
            
        Returns:
            A formatted prompt string
        """
        question = item["question"]
        choices = item["choices"]["text"]
        labels = item["choices"]["label"]
        
        prompt = f"Question: {question}\n"
        for i, choice_text in enumerate(choices):
            prompt += f"{labels[i]}. {choice_text}\n"
        prompt += "Answer:"
        
        return prompt

    def evaluate(self, model: ModelWrapper) -> Dict[str, Any]:
        """
        Evaluate the model on the ARC benchmark.
        
        Args:
            model: The model to evaluate
            
        Returns:
            Dictionary containing evaluation results
        """
        if self.data is None:
            self.data = self.load_data()
            if not self.data:
                return {"error": "Failed to load data."}

        correct = 0
        total = len(self.data)
        failed_examples = []
        
        for item in self.data:
            prompt = self.format_prompt(item)
            
            try:
                response = model.generate(prompt, max_new_tokens=5)
                
                # Robust parsing using regex
                response_clean = response.strip().upper()
                predicted_idx = None
                
                # Search for the first character that matches a choice label (e.g., A, B, C, D, 1, 2, 3)
                match = re.search(r"([A-Z0-9])", response_clean)
                if match:
                    predicted_char = match.group(1)
                    if predicted_char in item["choices"]["label"]:
                        predicted_idx = item["choices"]["label"].index(predicted_char)

                # Check if correct
                if predicted_idx is not None and item["choices"]["label"][predicted_idx] == item["answerKey"]:
                    correct += 1
                else:
                    if len(failed_examples) < 10:
                        failed_examples.append({
                            "question": item["question"],
                            "choices": item["choices"]["text"],
                            "answerKey": item["answerKey"],
                            "prediction": response,
                        })

            except Exception as e:
                self.logger.error(f"Error evaluating ARC example: {e}")
                if len(failed_examples) < 10:
                    failed_examples.append({
                        "question": item["question"],
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
                "subset": self.subset
            },
            "failed_examples": failed_examples
        }
        
        self.logger.info(f"ARC ({self.subset}) evaluation complete. Accuracy: {accuracy:.4f}")
        return results