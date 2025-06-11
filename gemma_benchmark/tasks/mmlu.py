"""
Benchmark for the MMLU (Massive Multitask Language Understanding) dataset.
"""

import re
import random
from typing import Dict, Any, List, Optional
from datasets import load_dataset, Dataset

# Core benchmark interfaces
from gemma_benchmark.core.interfaces import AbstractBenchmark, ModelInterface, BenchmarkResult

# Prompt templates
FEW_SHOT_PROMPT = """
The following are multiple choice questions (with answers) about {subject}.

{question}
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}
Answer: {answer}
""".strip()

FINAL_PROMPT = """
The following are multiple choice questions (with answers) about {subject}.

{few_shot_examples}
{question}
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}
Answer:
""".strip()


class MMLUBenchmark(AbstractBenchmark):
    """Benchmark for the MMLU dataset."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.subset = self.config.get('subset', 'all')
        self.shot_count = self.config.get('shot_count', 5)

    def load_data(self) -> Dataset:
        """Load data for the specified MMLU subset."""
        self.logger.info(f"Loading MMLU dataset for subset: {self.subset}")
        # Load the dev set for few-shot examples and the test set for evaluation
        dev_data = load_dataset("cais/mmlu", self.subset, split='dev')
        test_data = load_dataset("cais/mmlu", self.subset, split='test')
        
        # Prepare prompts
        prompts = []
        subjects = list(set(test_data['subject']))
        
        for subject in subjects:
            # Get few-shot examples from the dev set for this subject
            dev_examples = [ex for ex in dev_data if ex['subject'] == subject]
            if len(dev_examples) > self.shot_count:
                few_shot_samples = random.sample(dev_examples, self.shot_count)
            else:
                few_shot_samples = dev_examples
            
            few_shot_str = "\n\n".join([
                FEW_SHOT_PROMPT.format(
                    subject=subject,
                    question=ex['question'],
                    choice_a=ex['choices'][0],
                    choice_b=ex['choices'][1],
                    choice_c=ex['choices'][2],
                    choice_d=ex['choices'][3],
                    answer=chr(ord('A') + ex['answer'])
                ) for ex in few_shot_samples
            ])
            
            # Create prompts for test examples of this subject
            for item in [t for t in test_data if t['subject'] == subject]:
                prompt = FINAL_PROMPT.format(
                    subject=subject,
                    few_shot_examples=few_shot_str,
                    question=item['question'],
                    choice_a=item['choices'][0],
                    choice_b=item['choices'][1],
                    choice_c=item['choices'][2],
                    choice_d=item['choices'][3],
                )
                prompts.append({
                    "prompt": prompt,
                    "correct_letter": chr(ord('A') + item['answer']),
                    "subject": subject
                })
                
        return Dataset.from_list(prompts)

    def _evaluate_impl(self, model: ModelInterface) -> Dict[str, Any]:
        """Implementation-specific evaluation logic for MMLU."""
        if self._data is None:
            self._data = self.load_data()
            
        prompts = [item['prompt'] for item in self._data]
        
        self.logger.info(f"Generating responses for {len(prompts)} MMLU prompts...")
        responses = model.generate_batch(prompts, max_new_tokens=5)
        
        # Calculate accuracy
        correct_by_subject = {}
        total_by_subject = {}
        results_details = []
        
        for i, (item, response) in enumerate(zip(self._data, responses)):
            correct_letter = item['correct_letter']
            subject = item['subject']
            
            # Robustly parse the answer from the model's response using regex
            response_clean = response.strip()
            match = re.search(r'\b([A-D])\b', response_clean)
            predicted_letter = match.group(1) if match else None
            
            is_correct = (predicted_letter == correct_letter)
            
            # Track correct answers by subject
            if subject not in correct_by_subject:
                correct_by_subject[subject] = 0
                total_by_subject[subject] = 0
            
            if is_correct:
                correct_by_subject[subject] += 1
            total_by_subject[subject] += 1

            results_details.append({
                "index": i,
                "subject": subject,
                "prompt": item['prompt'],
                "response": response,
                "predicted_answer": predicted_letter,
                "correct_answer": correct_letter,
                "correct": is_correct
            })
            
        # Calculate overall and per-subject accuracy
        total_correct = sum(correct_by_subject.values())
        total_questions = len(self._data)
        overall_accuracy = (total_correct / total_questions) * 100 if total_questions > 0 else 0
        
        subject_accuracies = {
            subject: (correct_by_subject[subject] / total_by_subject[subject]) * 100
            for subject in total_by_subject
        }
        
        self.logger.info(f"MMLU Overall Accuracy: {overall_accuracy:.2f}%")
        
        return {
            "overall": {
                "accuracy": overall_accuracy,
                "total_questions": total_questions,
                "correct_answers": total_correct,
            },
            "subjects": subject_accuracies,
            "details": results_details,
        }