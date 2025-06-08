"""
ARC (AI2 Reasoning Challenge) benchmark implementation for science reasoning evaluation.
"""

import logging
from typing import Dict, List, Any, Optional
from datasets import load_dataset

from ..core.model_loader import ModelWrapper

class ArcBenchmark:
    """
    Implementation of the ARC benchmark for evaluating science reasoning.
    
    ARC contains grade-school level science questions that require reasoning
    about the physical world. Includes both ARC-Easy and ARC-Challenge.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ARC benchmark.
        
        Args:
            config: Configuration dictionary for the benchmark
        """
        self.logger = logging.getLogger("gemma_benchmark.tasks.arc")
        self.config = config
        self.subset = config.get("subset", "ARC-Challenge")  # or "ARC-Easy"
        self.shot_count = config.get("shot_count", 5)
        self.data = None
    
    def load_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load ARC dataset from HuggingFace.
        
        Returns:
            Dictionary containing train, validation, and test data
        """
        self.logger.info(f"Loading ARC data ({self.subset}) from HuggingFace...")
        
        try:
            # Load the ARC dataset
            dataset = load_dataset("ai2_arc", self.subset)
            
            data = {
                "train": [],
                "validation": [],
                "test": []
            }
            
            # Process training data
            for item in dataset["train"]:
                data["train"].append({
                    "question": item["question"],
                    "choices": item["choices"]["text"],
                    "choice_labels": item["choices"]["label"],
                    "correct_answer": item["answerKey"],
                    "id": item["id"]
                })
            
            # Process validation data
            for item in dataset["validation"]:
                data["validation"].append({
                    "question": item["question"],
                    "choices": item["choices"]["text"],
                    "choice_labels": item["choices"]["label"], 
                    "correct_answer": item["answerKey"],
                    "id": item["id"]
                })
            
            # Process test data
            for item in dataset["test"]:
                data["test"].append({
                    "question": item["question"],
                    "choices": item["choices"]["text"],
                    "choice_labels": item["choices"]["label"],
                    "correct_answer": item["answerKey"],
                    "id": item["id"]
                })
            
            self.logger.info(f"Loaded {len(data['train'])} train, {len(data['validation'])} validation, {len(data['test'])} test examples")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load ARC data: {e}")
            return self._create_mock_data()
    
    def _create_mock_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Create mock ARC data for demonstration."""
        self.logger.info("Creating mock ARC data...")
        
        mock_examples = [
            {
                "question": "Which property of a mineral can be determined just by looking at it?",
                "choices": ["hardness", "color", "mass", "density"],
                "choice_labels": ["A", "B", "C", "D"],
                "correct_answer": "B",
                "id": "mock_arc_1"
            },
            {
                "question": "What happens to water when it freezes?",
                "choices": ["It becomes lighter", "It expands", "It becomes a gas", "It dissolves"],
                "choice_labels": ["A", "B", "C", "D"], 
                "correct_answer": "B",
                "id": "mock_arc_2"
            },
            {
                "question": "Which of these is an example of a physical change?",
                "choices": ["burning wood", "melting ice", "digesting food", "rusting iron"],
                "choice_labels": ["A", "B", "C", "D"],
                "correct_answer": "B", 
                "id": "mock_arc_3"
            }
        ]
        
        return {
            "train": mock_examples * 5,
            "validation": mock_examples * 2,
            "test": mock_examples
        }
    
    def format_prompt(self, question: str, choices: List[str], choice_labels: List[str], examples: List[Dict[str, Any]] = None) -> str:
        """
        Format a prompt for ARC evaluation.
        
        Args:
            question: The science question
            choices: List of answer choices
            choice_labels: Labels for choices (A, B, C, D)
            examples: Few-shot examples (optional)
            
        Returns:
            Formatted prompt string
        """
        prompt = "Answer the following science questions by choosing the best answer.\n\n"
        
        # Add few-shot examples
        if examples:
            for i, example in enumerate(examples[:self.shot_count]):
                prompt += f"Question: {example['question']}\n"
                for j, (label, choice) in enumerate(zip(example['choice_labels'], example['choices'])):
                    prompt += f"{label}. {choice}\n"
                prompt += f"Answer: {example['correct_answer']}\n\n"
        
        # Add the target question
        prompt += f"Question: {question}\n"
        for label, choice in zip(choice_labels, choices):
            prompt += f"{label}. {choice}\n"
        prompt += "Answer:"
        
        return prompt
    
    def evaluate(self, model: ModelWrapper, num_samples: int = None) -> Dict[str, Any]:
        """
        Evaluate the model on ARC benchmark.
        
        Args:
            model: The model to evaluate
            num_samples: Number of samples to evaluate (None for all test data)
            
        Returns:
            Dictionary containing evaluation results
        """
        if self.data is None:
            self.data = self.load_data()
        
        # Use test data for evaluation, train data for few-shot examples
        examples = self.data["train"][:self.shot_count] if self.shot_count > 0 else []
        eval_data = self.data["test"]
        
        if num_samples is not None:
            eval_data = eval_data[:num_samples]
        
        self.logger.info(f"Evaluating ARC {self.subset} with {len(eval_data)} samples")
        
        correct = 0
        total = len(eval_data)
        failed_examples = []
        
        for i, example in enumerate(eval_data):
            if (i + 1) % 50 == 0:
                self.logger.info(f"Progress: {i + 1}/{total}")
            
            prompt = self.format_prompt(
                example["question"],
                example["choices"], 
                example["choice_labels"],
                examples
            )
            
            try:
                # Get model response
                response = model.generate(prompt, max_new_tokens=10, temperature=0.0)
                
                # Extract the answer choice
                predicted_answer = None
                response_clean = response.strip().upper()
                
                # Look for the choice label in the response
                for label in example["choice_labels"]:
                    if response_clean.startswith(label.upper()):
                        predicted_answer = label.upper()
                        break
                
                # Check if correct
                if predicted_answer == example["correct_answer"].upper():
                    correct += 1
                else:
                    # Store failed example for analysis
                    if len(failed_examples) < 10:
                        failed_examples.append({
                            "id": example["id"],
                            "question": example["question"],
                            "choices": dict(zip(example["choice_labels"], example["choices"])),
                            "correct_answer": example["correct_answer"],
                            "predicted_answer": predicted_answer,
                            "response": response
                        })
                
            except Exception as e:
                self.logger.error(f"Error processing example {example['id']}: {e}")
                if len(failed_examples) < 10:
                    failed_examples.append({
                        "id": example["id"],
                        "question": example["question"],
                        "choices": dict(zip(example["choice_labels"], example["choices"])),
                        "correct_answer": example["correct_answer"],
                        "predicted_answer": "ERROR",
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
                "subset": self.subset,
                "shot_count": self.shot_count,
                "num_samples": len(eval_data)
            },
            "failed_examples": failed_examples[:5]  # Include a few for analysis
        }
        
        self.logger.info(f"ARC {self.subset} evaluation complete. Accuracy: {accuracy:.4f} ({correct}/{total})")
        return results