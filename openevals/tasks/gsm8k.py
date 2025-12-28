"""
Benchmark for the GSM8K dataset.
"""

import random
import re
from typing import Any, Dict, List, Optional, Union

from datasets import Dataset, load_dataset

# Core benchmark interfaces
from openevals.core.interfaces import AbstractBenchmark, BenchmarkResult, ModelInterface

# Utility for parsing numerical answers
from openevals.utils.metrics import extract_numerical_answer, is_exact_match

FEW_SHOT_PROMPT = """
A grade school math problem.
**Question:**
{question}
**Answer:**
{answer}
""".strip()

FINAL_PROMPT = """
{few_shot_examples}
A grade school math problem.
**Question:**
{question}
**Answer:**
""".strip()


class GSM8KBenchmark(AbstractBenchmark):
    """Benchmark for the GSM8K dataset."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.shot_count = self.config.get("shot_count", 8)

        # Initialize a dedicated random number generator for reproducibility
        self.random_seed = self.config.get("random_seed", 42)
        self.rng = random.Random(self.random_seed)

        if self.shot_count < 0 or self.shot_count > 8:
            raise ValueError(
                "Number of few-shot examples must be between 0 and 8 for GSM8K."
            )

    def load_data(self) -> Dataset:
        """Load both train and test splits of GSM8K."""
        self.logger.info("Loading GSM8K dataset...")
        train_data = load_dataset("gsm8k", "main", split="train")
        test_data = load_dataset("gsm8k", "main", split="test")

        # Prepare prompts for each test item
        prompts = []
        for item in test_data:
            few_shot_examples_str = ""
            if self.shot_count > 0:
                # Avoid using the test example itself in the few-shot selection
                potential_shots = [
                    s for s in train_data if s["question"] != item["question"]
                ]
                if len(potential_shots) >= self.shot_count:
                    # Use the seeded RNG for sampling to ensure reproducibility
                    few_shot_examples = self.rng.sample(
                        potential_shots, self.shot_count
                    )
                    few_shot_examples_str = (
                        "\n".join(
                            [FEW_SHOT_PROMPT.format(**ex) for ex in few_shot_examples]
                        )
                        + "\n"
                    )

            final_prompt = FINAL_PROMPT.format(
                few_shot_examples=few_shot_examples_str, question=item["question"]
            )
            prompts.append({"prompt": final_prompt, "ground_truth": item["answer"]})

        return Dataset.from_list(prompts)

    def _evaluate_impl(self, model: ModelInterface) -> Dict[str, Any]:
        """Implementation-specific evaluation logic for GSM8K."""
        if self._data is None:
            self._data = self.load_data()

        prompts = [item["prompt"] for item in self._data]
        ground_truths = [item["ground_truth"] for item in self._data]

        # Generate responses in batches
        self.logger.info(f"Generating responses for {len(prompts)} prompts...")
        responses = model.generate_batch(prompts, max_new_tokens=256)

        # Calculate accuracy
        correct = 0
        results_details = []

        for i, (prompt, response, truth) in enumerate(
            zip(prompts, responses, ground_truths)
        ):
            # Extract numerical answers from both the ground truth and the model's response
            true_answer = extract_numerical_answer(truth)
            model_answer = extract_numerical_answer(response)

            match = is_exact_match(model_answer, true_answer)
            if match:
                correct += 1

            results_details.append(
                {
                    "index": i,
                    "prompt": prompt,
                    "response": response,
                    "ground_truth": truth,
                    "model_answer": model_answer,
                    "true_answer": true_answer,
                    "correct": match,
                }
            )

        accuracy = (correct / len(prompts)) * 100
        self.logger.info(f"GSM8K Accuracy: {accuracy:.2f}%")

        return {
            "overall": {
                "accuracy": accuracy,
                "total_questions": len(prompts),
                "correct_answers": correct,
            },
            "details": results_details,
        }
