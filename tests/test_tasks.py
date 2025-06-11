"""Tests for benchmark tasks."""

import pytest
from unittest.mock import Mock, patch
import tempfile
import sys
import platform

# Mock the datasets library if it's not available
try:
    import datasets
except ImportError:
    sys.modules["datasets"] = Mock()

from gemma_benchmark.tasks.mmlu import MMLUBenchmark
from gemma_benchmark.tasks.arc import ArcBenchmark
from gemma_benchmark.tasks.efficiency import EfficiencyBenchmark
from gemma_benchmark.tasks.gsm8k import GSM8KBenchmark
from gemma_benchmark.tasks.humaneval import HumanevalBenchmark
from gemma_benchmark.core.model_loader import ModelWrapper


class TestMMLUBenchmark:
    """Test MMLU benchmark implementation."""

    def test_init(self):
        """Test MMLU benchmark initialization."""
        config = {"subset": "mathematics", "shot_count": 3}
        benchmark = MMLUBenchmark(config)
        assert benchmark.subset == "mathematics"
        assert benchmark.shot_count == 3
        assert benchmark.data is None

    def test_format_prompt(self):
        """Test MMLU prompt formatting."""
        config = {"subset": "all", "shot_count": 1}
        benchmark = MMLUBenchmark(config)

        example = {
            "question": "What is 2+2?",
            "options": ["3", "4", "5", "6"],
            "answer": 1,
            "subject": "mathematics",
            "examples": [
                {
                    "question": "What is 1+1?",
                    "options": ["1", "2", "3", "4"],
                    "answer": 1,
                }
            ],
        }

        prompt = benchmark.format_prompt(example)

        assert "mathematics" in prompt
        assert "What is 2+2?" in prompt
        assert "A. 3" in prompt
        assert "B. 4" in prompt
        assert "What is 1+1?" in prompt  # Few-shot example

    @patch("datasets.load_dataset")
    def test_load_data_success(self, mock_load_dataset):
        """Test successful data loading."""
        mock_dataset = {
            "test": {
                "subject": ["math"],
                "question": ["Q1"],
                "choices": [["A", "B"]],
                "answer": [0],
            },
            "dev": {
                "subject": ["math"],
                "question": ["Q_dev1"],
                "choices": [["A_dev", "B_dev"]],
                "answer": [1],
            },
        }
        mock_load_dataset.return_value = mock_dataset

        config = {"subset": "math", "shot_count": 1}
        benchmark = MMLUBenchmark(config)
        data = benchmark.load_data()

        assert "math" in data
        assert len(data["math"]) == 1
        assert data["math"][0]["question"] == "Q1"


class TestEfficiencyBenchmark:
    """Test efficiency benchmark implementation."""

    def test_init(self):
        """Test efficiency benchmark initialization."""
        config = {"sample_prompts": ["Test prompt 1"], "output_lengths": [64]}
        benchmark = EfficiencyBenchmark(config)
        assert len(benchmark.sample_prompts) == 1
        assert "os" in benchmark.system_info

    @patch("time.time", side_effect=[0.0, 1.0, 2.0, 3.0])
    @patch("psutil.Process")
    def test_evaluate_mock_model(self, mock_process, mock_time):
        """Test efficiency evaluation with mock model."""
        mock_process.return_value.memory_info.return_value.rss = 1024**3
        mock_model = Mock(spec=ModelWrapper)
        mock_model.model_name = "test-model"
        mock_model.generate.return_value = "Generated text"

        config = {"sample_prompts": ["Test"], "output_lengths": [128]}
        benchmark = EfficiencyBenchmark(config)
        results = benchmark.evaluate(mock_model)

        assert "latency" in results
        assert "tokens_per_second" in results
        assert "memory_usage" in results


class TestGSM8KBenchmark:
    """Test GSM8K benchmark implementation."""

    def test_init(self):
        """Test GSM8K benchmark initialization."""
        config = {"shot_count": 3, "use_chain_of_thought": True}
        benchmark = Gsm8kBenchmark(config)
        assert benchmark.shot_count == 3
        assert benchmark.use_cot is True


### New Tests for Improved Coverage ###


def test_mmlu_answer_extraction_robustness():
    """Test MMLU answer extraction with various formats by mocking the evaluation."""
    config = {"subset": "all", "shot_count": 0}
    benchmark = MMLUBenchmark(config)

    # This test assumes the MMLU evaluate method uses robust regex parsing.
    # We mock a single question and test different model responses.
    mock_data = {
        "science": [
            {
                "question": "What is H2O?",
                "options": ["Water", "Air", "Fire", "Earth"],
                "answer": 0,
                "subject": "science",
                "examples": [],
            }
        ]
    }
    benchmark.data = mock_data

    test_cases = [
        ("A", 1),
        ("The answer is A", 1),
        ("(A)", 1),
        ("A.", 1),
        ("B", 0),  # Incorrect case
    ]

    for response, expected_correct in test_cases:
        mock_model = Mock(spec=ModelWrapper)
        mock_model.generate.return_value = response
        results = benchmark.evaluate(mock_model)
        assert results["overall"]["correct"] == expected_correct


def test_efficiency_token_counting():
    """Test that efficiency benchmark uses actual token counting when available."""
    mock_model = Mock(spec=ModelWrapper)
    mock_model.model_name = "test-model"
    mock_model.generate.return_value = "This is a test response with multiple words"

    # Add mock tokenizer
    mock_tokenizer = Mock()
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5, 6, 7, 8]  # 8 tokens
    mock_model.tokenizer = mock_tokenizer  # Use the public attribute

    config = {
        "sample_prompts": ["Test"],
        "output_lengths": [128],
        "warmup_runs": 0,
        "measurement_runs": 1,
    }
    benchmark = EfficiencyBenchmark(config)

    # Mock time to be exactly 1 second
    with patch("time.perf_counter", side_effect=[0.0, 1.0]):
        metrics = benchmark._measure_single_generation(mock_model, "Test", 128)

    # Should have used actual token count (8) not estimation
    assert metrics.tokens_per_second == 8.0  # 8 tokens / 1 second


def test_gsm8k_improved_answer_extraction():
    """Test improved GSM8K answer extraction."""
    config = {"shot_count": 0}
    benchmark = Gsm8kBenchmark(config)

    test_cases = [
        ("The calculation is 5 + 3 = 8. #### 8", 8.0),
        ("First we add 10 + 20 = 30. The answer is 30.", 30.0),
        ("After solving: \\boxed{42}", 42.0),
        ("The final result is 1,234 dollars. #### 1,234", 1234.0),
        ("100 - 25 = 75.", 75.0),  # Fallback to last number
    ]

    for text, expected in test_cases:
        result = benchmark._extract_numerical_answer(text)
        assert result == expected, f"Failed for text: '{text}'"


def test_humaneval_secure_code_execution():
    """Test HumanEval security improvements."""
    # This test might be platform-specific, skip if not on unix-like system for resource check
    if platform.system() == "Windows":
        pytest.skip("Skipping resource limit tests on Windows")

    config = {"timeout": 5}
    benchmark = HumanevalBenchmark(config)

    # Test dangerous code detection
    dangerous_codes = [
        "import os\nos.system('rm -rf /')",
        "__import__('subprocess').call(['ls'])",
        "from sys import exit",  # Test disallowed from-import
    ]

    for code in dangerous_codes:
        result = benchmark.test_code(code, "pass")
        assert result is False, f"Dangerous code not blocked: {code}"

    # Test allowed, safe code execution by mocking the subprocess call
    allowed_code = "import math\n\ndef add(a, b):\n    return a + b"
    test_code = "assert add(1, 2) == 3"

    with patch("subprocess.Popen") as mock_popen:
        mock_process = Mock()
        mock_process.communicate.return_value = ("", "")  # No error
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        result = benchmark.test_code(allowed_code, test_code)
        # Should not be rejected for security reasons and should attempt to run
        assert mock_popen.called
        assert result is True
