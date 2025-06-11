"""
Benchmark for the MMLU (Massive Multitask Language Understanding) dataset.
"""

import re
import random
from typing import Dict, Any, List
from datasets import load_dataset, Dataset

# Core benchmark interfaces
from gemma_benchmark.core.interfaces import AbstractBenchmark, ModelInterface

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
        self.subset = self.config.get("subset", "all")
        self.shot_count = self.config.get("shot_count", 5)

    # --- HELPER METHOD ADDED TO SUPPORT THE NEW load_data ---
    def _format_full_prompt(
        self, question: str, choices: list, subject: str, few_shot_examples: list
    ) -> str:
        """Formats the complete prompt with few-shot examples."""

        # Format the few-shot examples from the dev set
        few_shot_str = "\n\n".join(
            [
                FEW_SHOT_PROMPT.format(
                    subject=subject,
                    question=ex["question"],
                    choice_a=ex["choices"][0],
                    choice_b=ex["choices"][1],
                    choice_c=ex["choices"][2],
                    choice_d=ex["choices"][3],
                    answer=chr(ord("A") + ex["answer"]),
                )
                for ex in few_shot_examples
            ]
        )

        # Format the final prompt for the current test question
        full_prompt = FINAL_PROMPT.format(
            subject=subject,
            few_shot_examples=few_shot_str,
            question=question,
            choice_a=choices[0],
            choice_b=choices[1],
            choice_c=choices[2],
            choice_d=choices[3],
        )
        return full_prompt

    # --- REPLACED METHOD ---
    def load_data(self) -> Dataset:
        """Load data for the specified MMLU subset."""
        self.logger.info(f"Loading MMLU dataset for subset: {self.subset}")
        # Load dev and test splits from the HuggingFace dataset
        dataset = load_dataset("cais/mmlu", self.subset)
        dev_data = dataset["dev"]
        test_data = dataset["test"]

        # Prepare prompts with correct field names
        prompts = []
        for item in test_data:
            # Get few-shot examples from dev set
            few_shot_examples = []
            # Use random sampling for few-shot examples for variety
            potential_shots = list(dev_data)
            if len(potential_shots) > self.shot_count:
                few_shot_samples = random.sample(potential_shots, self.shot_count)
            else:
                few_shot_samples = potential_shots

            for dev_item in few_shot_samples:
                few_shot_examples.append(
                    {
                        "question": dev_item["question"],
                        "choices": dev_item["choices"],
                        "answer": dev_item["answer"],
                    }
                )

            prompt = self._format_full_prompt(
                question=item["question"],
                choices=item["choices"],
                subject=item.get(
                    "subject", self.subset
                ),  # Use item's subject if available
                few_shot_examples=few_shot_examples,
            )

            prompts.append(
                {
                    "prompt": prompt,
                    "correct_answer": item["answer"],  # 0-3 index
                    "correct_letter": chr(ord("A") + item["answer"]),
                    "subject": item.get("subject", self.subset),
                }
            )
        return Dataset.from_list(prompts)

    def _evaluate_impl(self, model: ModelInterface) -> Dict[str, Any]:
        """Implementation-specific evaluation logic for MMLU."""
        if self._data is None:
            self._data = self.load_data()

        prompts = [item["prompt"] for item in self._data]

        self.logger.info(f"Generating responses for {len(prompts)} MMLU prompts...")
        responses = model.generate_batch(prompts, max_new_tokens=5)

        # Calculate accuracy
        correct_by_subject = {}
        total_by_subject = {}
        results_details = []

        for i, (item, response) in enumerate(zip(self._data, responses)):
            correct_letter = item["correct_letter"]
            subject = item["subject"]

            # Robustly parse the answer from the model's response using regex
            response_clean = response.strip()
            match = re.search(r"\b([A-D])\b", response_clean)
            predicted_letter = match.group(1) if match else None

            is_correct = predicted_letter == correct_letter

            # Track correct answers by subject
            if subject not in correct_by_subject:
                correct_by_subject[subject] = 0
                total_by_subject[subject] = 0

            if is_correct:
                correct_by_subject[subject] += 1
            total_by_subject[subject] += 1

            results_details.append(
                {
                    "index": i,
                    "subject": subject,
                    "prompt": item["prompt"],
                    "response": response,
                    "predicted_answer": predicted_letter,
                    "correct_answer": correct_letter,
                    "correct": is_correct,
                }
            )

        # Calculate overall and per-subject accuracy
        total_correct = sum(correct_by_subject.values())
        total_questions = len(self._data)
        overall_accuracy = (
            (total_correct / total_questions) * 100 if total_questions > 0 else 0
        )

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
