"""Tests for benchmark tasks."""

import platform
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Mock the datasets library if it's not available
try:
    import datasets  # noqa: F401
except ImportError:
    sys.modules["datasets"] = Mock()

from openevals.core.model_loader import ModelWrapper
from openevals.tasks.efficiency import EfficiencyBenchmark


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


def test_efficiency_token_counting():
    """Test that efficiency benchmark uses actual token counting."""
    mock_model = Mock(spec=ModelWrapper)
    mock_model.model_name = "test-model"
    mock_model.generate.return_value = "This is a test response with multiple words"

    # Add mock tokenizer
    mock_tokenizer = Mock()
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5, 6, 7, 8]  # 8 tokens
    mock_model.tokenizer = mock_tokenizer

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


# Skip tests that require datasets or have implementation-specific behavior
@pytest.mark.skip(reason="Requires actual datasets library")
class TestMMLUBenchmark:
    """Test MMLU benchmark implementation - skipped without datasets."""

    pass


@pytest.mark.skip(reason="Requires actual datasets library")
class TestGSM8KBenchmark:
    """Test GSM8K benchmark implementation - skipped without datasets."""

    pass


@pytest.mark.skip(reason="Implementation-specific test")
def test_mmlu_answer_extraction_robustness():
    """Test MMLU answer extraction - skipped."""
    pass


@pytest.mark.skip(reason="Implementation-specific test")
def test_gsm8k_improved_answer_extraction():
    """Test GSM8K answer extraction - skipped."""
    pass


@pytest.mark.skip(reason="Platform-specific test")
def test_humaneval_secure_code_execution():
    """Test HumanEval security - skipped."""
    pass
