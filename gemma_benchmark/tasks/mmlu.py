"""
MMLU (Massive Multitask Language Understanding) benchmark implementation.
"""

import os
import csv
import logging
import random
from typing import Dict, List, Any, Optional, Tuple

from ..core.model_loader import ModelWrapper

class MMLUBenchmark:
    """
    Implementation of the MMLU benchmark for evaluating knowledge and reasoning.
    
    MMLU covers 57 subjects across STEM, humanities, social sciences, and more.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MMLU benchmark.
        
        Args:
            config: Configuration dictionary for the benchmark
        """
        self.logger = logging.getLogger("gemma_benchmark.tasks.mmlu")
        self.config = config
        self.data_path = config.get("data_path", "data/mmlu")
        self.subset = config.get("subset", "all")
        self.shot_count = config.get("shot_count", 5)
        self.data = None
    
    def load_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load MMLU dataset.
        
        Returns:
            Dictionary mapping subject names to lists of questions
        """
        self.logger.info("Loading MMLU data...")
        
        # Check if data path exists
        if not os.path.exists(self.data_path):
            self.logger.warning(f"Data path {self.data_path} doesn't exist. Using mock data.")
            self.logger.warning("Consider running: python -m gemma_benchmark.scripts.download_data --mmlu")
            return self._create_mock_data()
        
        subjects = []
        if self.subset == "all":
            # Get all subject directories/files based on test files
            for root, _, files in os.walk(self.data_path):
                for file in files:
                    if file.endswith("_test.csv"):
                        subject = file.replace("_test.csv", "")
                        subjects.append(subject)
        else:
            subjects = [self.subset]
        
        if not subjects:
            self.logger.warning("No subjects found. Using mock data.")
            return self._create_mock_data()
        
        data = {}
        for subject in subjects:
            # Paths for test and dev files
            test_path = os.path.join(self.data_path, f"{subject}_test.csv")
            dev_path = os.path.join(self.data_path, f"{subject}_dev.csv")
            val_path = os.path.join(self.data_path, f"{subject}_val.csv")  # Some datasets use val instead of dev
            
            # Try to find the development set file
            dev_file = None
            if os.path.exists(dev_path):
                dev_file = dev_path
            elif os.path.exists(val_path):
                dev_file = val_path
            
            if not os.path.exists(test_path):
                self.logger.warning(f"Test file {test_path} doesn't exist. Skipping subject.")
                continue
            
            # Load test data
            test_data = []
            try:
                with open(test_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 6:  # Question, A, B, C, D, Answer
                            test_data.append({
                                "question": row[0],
                                "options": row[1:5],
                                "answer": ord(row[5]) - ord('A'),  # Convert A,B,C,D to 0,1,2,3
                                "subject": subject
                            })
            except Exception as e:
                self.logger.error(f"Error loading test data for {subject}: {e}")
                continue
            
            # Load few-shot examples if needed
            if self.shot_count > 0 and dev_file and os.path.exists(dev_file):
                examples = []
                try:
                    with open(dev_file, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        for i, row in enumerate(reader):
                            if i >= self.shot_count:
                                break
                            if len(row) >= 6:
                                examples.append({
                                    "question": row[0],
                                    "options": row[1:5],
                                    "answer": ord(row[5]) - ord('A')
                                })
                    
                    # Attach examples to each test question
                    for item in test_data:
                        item["examples"] = examples
                except Exception as e:
                    self.logger.error(f"Error loading dev data for {subject}: {e}")
            
            data[subject] = test_data
            self.logger.info(f"Loaded {len(test_data)} test examples for subject: {subject}")
        
        if not data:
            self.logger.warning("No data loaded. Using mock data.")
            return self._create_mock_data()
            
        return data
    
    def _create_mock_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Create mock MMLU data for demonstration."""
        self.logger.info("Creating mock MMLU data...")
        
        # Define subjects and questions
        mock_subjects = ["mathematics", "physics", "computer_science", "history"]
        mock_data = {}
        
        for subject in mock_subjects:
            questions = []
            
            # Create 10 mock questions per subject
            for i in range(10):
                if subject == "mathematics":
                    question = f"What is the derivative of x^{i+1}?"
                    options = [f"{i+1}x^{i}", f"{i+1}x^{i-1}", f"{i}x^{i+1}", f"{i}x^{i-1}"]
                    answer = 1  # B is correct
                elif subject == "physics":
                    question = f"What is the SI unit of {['force', 'energy', 'power', 'pressure'][i % 4]}?"
                    options = ["Newton", "Joule", "Watt", "Pascal"]
                    answer = i % 4
                elif subject == "computer_science":
                    question = f"Which data structure is best for {['searching', 'sorting', 'insertion', 'deletion'][i % 4]}?"
                    options = ["Array", "Linked List", "Heap", "Binary Search Tree"]
                    answer = (i + 1) % 4
                else:  # history
                    question = f"Who was the {i+1}th President of the United States?"
                    options = ["George Washington", "Thomas Jefferson", "Abraham Lincoln", "Franklin D. Roosevelt"]
                    answer = i % 4
                
                questions.append({
                    "question": question,
                    "options": options,
                    "answer": answer,
                    "subject": subject
                })
            
            # Create examples for few-shot learning
            examples = []
            for i in range(self.shot_count):
                examples.append({
                    "question": f"Example question {i+1} for {subject}?",
                    "options": [f"Option A for example {i+1}", 
                                f"Option B for example {i+1}",
                                f"Option C for example {i+1}", 
                                f"Option D for example {i+1}"],
                    "answer": i % 4
                })
            
            # Attach examples to each question
            for item in questions:
                item["examples"] = examples
            
            mock_data[subject] = questions
        
        return mock_data
    
    def format_prompt(self, example: Dict[str, Any]) -> str:
        """
        Format a prompt for the model from an MMLU example.
        
        Args:
            example: Dictionary containing the question, options, and examples
            
        Returns:
            Formatted prompt string
        """
        prompt = "The following is a multiple choice question. Please choose the correct answer.\n\n"
        
        # Add few-shot examples if available
        if "examples" in example and example["examples"]:
            for i, ex in enumerate(example["examples"]):
                prompt += f"Example {i+1}:\n"
                prompt += f"Question: {ex['question']}\n"
                prompt += "Options:\n"
                for j, option in enumerate(ex['options']):
                    option_letter = chr(ord('A') + j)
                    prompt += f"{option_letter}. {option}\n"
                
                answer_letter = chr(ord('A') + ex["answer"])
                prompt += f"Answer: {answer_letter}\n\n"
        
        # Add the actual question
        prompt += f"Question: {example['question']}\n"
        prompt += "Options:\n"
        
        for j, option in enumerate(example["options"]):
            option_letter = chr(ord('A') + j)
            prompt += f"{option_letter}. {option}\n"
        
        prompt += "Answer:"
        return prompt
    
    def evaluate(self, model: ModelWrapper, runs: int = 1) -> Dict[str, Any]:
        """
        Evaluate the model on MMLU benchmark.
        
        Args:
            model: The model to evaluate
            runs: Number of runs to perform for statistical significance
            
        Returns:
            Dictionary containing evaluation results
        """
        if self.data is None:
            self.data = self.load_data()
        
        # For multiple runs, we'll collect results for statistical analysis
        all_run_results = []
        
        for run in range(runs):
            run_results = {
                "overall": {
                    "correct": 0,
                    "total": 0,
                    "accuracy": 0.0
                },
                "subjects": {}
            }
            all_run_results.append(run_results)
        
        # Aggregate results
        results = {
            "overall": {
                "correct": 0,
                "total": 0,
                "accuracy": 0.0
            },
            "subjects": {}
        }
        
        for subject, subject_data in self.data.items():
            self.logger.info(f"Evaluating subject: {subject} with {len(subject_data)} questions")
            
            subject_correct = 0
            subject_total = len(subject_data)
            failed_examples = []
            
            for example in subject_data:
                prompt = self.format_prompt(example)
                
                # Get model response
                response = model.generate(prompt, max_new_tokens=10)
                
                # Extract the answer (A, B, C, or D)
                answer_matches = [
                    response.strip().startswith(letter)
                    for letter in ["A", "B", "C", "D"]
                ]
                
                # Check if correct answer was given
                if sum(answer_matches) == 1:
                    predicted_answer = answer_matches.index(True)
                    if predicted_answer == example["answer"]:
                        subject_correct += 1
                        results["overall"]["correct"] += 1
                    else:
                        # Record failure for error analysis
                        failed_examples.append({
                            "question": example["question"],
                            "options": example["options"],
                            "expected": chr(ord('A') + example["answer"]),
                            "predicted": chr(ord('A') + predicted_answer),
                            "response": response
                        })
                else:
                    # Could not determine answer
                    failed_examples.append({
                        "question": example["question"],
                        "options": example["options"],
                        "expected": chr(ord('A') + example["answer"]),
                        "predicted": "unclear",
                        "response": response
                    })
            
            # Calculate subject accuracy
            subject_accuracy = subject_correct / subject_total if subject_total > 0 else 0
            
            results["subjects"][subject] = {
                "correct": subject_correct,
                "total": subject_total,
                "accuracy": subject_accuracy,
                "failed_examples": failed_examples[:5]  # Include a few examples for analysis
            }
            
            results["overall"]["total"] += subject_total
        
        # Calculate overall accuracy
        if results["overall"]["total"] > 0:
            results["overall"]["accuracy"] = results["overall"]["correct"] / results["overall"]["total"]
        
        # Generate summary statistics
        total_questions = results["overall"]["total"]
        correct_questions = results["overall"]["correct"]
        accuracy = results["overall"]["accuracy"]
        
        # Find best and worst performing subjects
        if results["subjects"]:
            best_subject = max(results["subjects"].items(), key=lambda x: x[1]["accuracy"])
            worst_subject = min(results["subjects"].items(), key=lambda x: x[1]["accuracy"])
            
            results["summary"] = {
                "total_questions": total_questions,
                "correct_questions": correct_questions,
                "accuracy": accuracy,
                "best_subject": {
                    "name": best_subject[0],
                    "accuracy": best_subject[1]["accuracy"]
                },
                "worst_subject": {
                    "name": worst_subject[0],
                    "accuracy": worst_subject[1]["accuracy"]
                }
            }
        
        self.logger.info(f"MMLU Evaluation complete. Overall accuracy: {accuracy:.4f}")
        return results