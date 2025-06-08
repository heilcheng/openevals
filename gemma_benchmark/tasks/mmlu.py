"""
MMLU (Massive Multitask Language Understanding) benchmark implementation.
"""

import logging
from typing import Dict, List, Any, Optional
from datasets import load_dataset
import random

from ..core.model_loader import ModelWrapper

class MMLUBenchmark:
    """Implementation of the MMLU benchmark using HuggingFace datasets."""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger("gemma_benchmark.tasks.mmlu")
        self.config = config
        self.subset = config.get("subset", "all")
        self.shot_count = config.get("shot_count", 5)
        self.data = None
    
    def load_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load MMLU dataset from HuggingFace."""
        self.logger.info("Loading MMLU data from HuggingFace...")
        
        try:
            # Load the full MMLU dataset
            dataset = load_dataset("cais/mmlu", "all")
            
            data = {}
            subjects = list(set(dataset["test"]["subject"]))
            
            if self.subset != "all" and self.subset in subjects:
                subjects = [self.subset]
            
            for subject in subjects:
                # Filter test data for this subject
                test_data = []
                dev_data = []
                
                # Process test set
                for i, item_subject in enumerate(dataset["test"]["subject"]):
                    if item_subject == subject:
                        test_data.append({
                            "question": dataset["test"]["question"][i],
                            "options": dataset["test"]["choices"][i],
                            "answer": dataset["test"]["answer"][i],
                            "subject": subject
                        })
                
                # Process dev set for few-shot examples
                for i, item_subject in enumerate(dataset["dev"]["subject"]):
                    if item_subject == subject and len(dev_data) < self.shot_count:
                        dev_data.append({
                            "question": dataset["dev"]["question"][i],
                            "options": dataset["dev"]["choices"][i],
                            "answer": dataset["dev"]["answer"][i]
                        })
                
                # Attach examples to test data
                for item in test_data:
                    item["examples"] = dev_data
                
                data[subject] = test_data
                self.logger.info(f"Loaded {len(test_data)} test examples for {subject}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load MMLU data: {e}")
            return self._create_mock_data()
    
    def format_prompt(self, example: Dict[str, Any]) -> str:
        """Format a prompt for MMLU evaluation."""
        prompt = f"The following are multiple choice questions (with answers) about {example['subject']}.\n\n"
        
        # Add few-shot examples
        if "examples" in example and example["examples"]:
            for i, ex in enumerate(example["examples"]):
                prompt += f"Question: {ex['question']}\n"
                for j, option in enumerate(ex['options']):
                    prompt += f"{chr(ord('A') + j)}. {option}\n"
                prompt += f"Answer: {chr(ord('A') + ex['answer'])}\n\n"
        
        # Add the target question
        prompt += f"Question: {example['question']}\n"
        for j, option in enumerate(example["options"]):
            prompt += f"{chr(ord('A') + j)}. {option}\n"
        prompt += "Answer:"
        
        return prompt
    
    def evaluate(self, model: ModelWrapper, runs: int = 1) -> Dict[str, Any]:
        """Evaluate model on MMLU."""
        if self.data is None:
            self.data = self.load_data()
        
        results = {
            "overall": {"correct": 0, "total": 0, "accuracy": 0.0},
            "subjects": {}
        }
        
        for subject, subject_data in self.data.items():
            self.logger.info(f"Evaluating {subject} ({len(subject_data)} questions)")
            
            subject_correct = 0
            subject_total = len(subject_data)
            
            for example in subject_data:
                prompt = self.format_prompt(example)
                
                # Get model response
                response = model.generate(prompt, max_new_tokens=10, temperature=0.0)
                
                # Extract answer (look for A, B, C, D at start of response)
                predicted_answer = None
                response_clean = response.strip().upper()
                
                for i, letter in enumerate(['A', 'B', 'C', 'D']):
                    if response_clean.startswith(letter):
                        predicted_answer = i
                        break
                
                # Check if correct
                if predicted_answer == example["answer"]:
                    subject_correct += 1
                    results["overall"]["correct"] += 1
            
            # Calculate subject accuracy
            subject_accuracy = subject_correct / subject_total if subject_total > 0 else 0
            results["subjects"][subject] = {
                "correct": subject_correct,
                "total": subject_total,
                "accuracy": subject_accuracy
            }
            results["overall"]["total"] += subject_total
        
        # Calculate overall accuracy
        if results["overall"]["total"] > 0:
            results["overall"]["accuracy"] = results["overall"]["correct"] / results["overall"]["total"]
        
        self.logger.info(f"MMLU evaluation complete. Overall accuracy: {results['overall']['accuracy']:.4f}")
        return results