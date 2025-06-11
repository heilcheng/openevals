"""
Tests for the core benchmarking logic.
"""

import pytest
import os
from unittest.mock import patch, MagicMock

# Core components to test
from gemma_benchmark.core.benchmark import GemmaBenchmark, EvaluationError
from gemma_benchmark.utils.config_validation import ConfigurationError

# Sample valid and invalid configurations
VALID_CONFIG = {
    "models": {
        "gemma-2b-test": {
            "type": "gemma",
            "size": "2b",
            "variant": "it",
            "quantization": False,
        }
    },
    "tasks": {"mmlu-test": {"type": "mmlu", "subset": "mathematics", "shot_count": 0}},
}

# --- Fixtures ---


@pytest.fixture
def valid_config_file(tmp_path):
    """Create a temporary valid YAML config file."""
    import yaml

    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(VALID_CONFIG, f)
    return str(config_file)


# --- Test Cases ---


def test_benchmark_initialization_success(valid_config_file):
    """Test successful initialization of GemmaBenchmark with a valid config."""
    benchmark = GemmaBenchmark(valid_config_file)
    assert benchmark.config is not None
    assert "models" in benchmark.config
    assert "tasks" in benchmark.config


def test_benchmark_initialization_file_not_found():
    """Test that initialization fails if the config file does not exist."""
    with pytest.raises(FileNotFoundError):
        GemmaBenchmark("non_existent_file.yaml")


def test_load_config_invalid_yaml(tmp_path):
    """Test that initialization fails with ConfigurationError for invalid YAML syntax."""
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, "w") as f:
        f.write("models: { gemma-2b: { type: gemma")  # Malformed YAML

    # FIX: Expect ConfigurationError, not ValueError
    with pytest.raises(ConfigurationError):
        GemmaBenchmark(str(invalid_yaml_file))


def test_load_config_validation_error(tmp_path):
    """Test that initialization fails with ConfigurationError for invalid schema."""
    import yaml

    invalid_config = {"models": {"gemma-2b": {"type": "unknown_type"}}}
    config_file = tmp_path / "invalid_schema.yaml"
    with open(config_file, "w") as f:
        yaml.dump(invalid_config, f)

    with pytest.raises(ConfigurationError):
        GemmaBenchmark(str(config_file))


@patch("gemma_benchmark.core.benchmark.get_model_manager")
def test_load_models_success(mock_get_manager, valid_config_file):
    """Test that models are loaded correctly."""
    # Setup mock model manager and loader
    mock_model = MagicMock()
    mock_manager = MagicMock()
    mock_manager.load_model.return_value = mock_model
    mock_get_manager.return_value = mock_manager

    benchmark = GemmaBenchmark(valid_config_file)
    benchmark.load_models()  # Load all models from config

    mock_manager.load_model.assert_called_once_with(
        "gemma-2b-test", VALID_CONFIG["models"]["gemma-2b-test"]
    )
    assert "gemma-2b-test" in benchmark.models
    assert benchmark.models["gemma-2b-test"] == mock_model


@patch("gemma_benchmark.core.benchmark.BenchmarkFactory")
def test_load_tasks_success(mock_factory, valid_config_file):
    """Test that tasks are loaded correctly."""
    # Setup mock benchmark factory
    mock_task = MagicMock()
    mock_factory.create_benchmark.return_value = mock_task

    benchmark = GemmaBenchmark(valid_config_file)
    benchmark.load_tasks()  # Load all tasks from config

    mock_factory.create_benchmark.assert_called_once_with(
        "mmlu", VALID_CONFIG["tasks"]["mmlu-test"]
    )
    assert "mmlu-test" in benchmark.tasks
    assert benchmark.tasks["mmlu-test"] == mock_task


def test_run_benchmarks_no_models_or_tasks(valid_config_file):
    """Test that run_benchmarks raises an error if no models or tasks are loaded."""
    benchmark = GemmaBenchmark(valid_config_file)

    with pytest.raises(EvaluationError, match="No models or tasks loaded"):
        benchmark.run_benchmarks()


@patch("gemma_benchmark.core.benchmark.get_model_manager")
@patch("gemma_benchmark.core.benchmark.BenchmarkFactory")
def test_run_benchmarks_evaluation_flow(
    mock_factory, mock_get_manager, valid_config_file
):
    """Test the main evaluation flow of run_benchmarks."""
    # Setup mocks
    mock_model = MagicMock()
    mock_manager = MagicMock()
    mock_manager.load_model.return_value = mock_model
    mock_get_manager.return_value = mock_manager

    mock_task = MagicMock()
    mock_task.evaluate.return_value = {"accuracy": 95.0}
    mock_factory.create_benchmark.return_value = mock_task

    benchmark = GemmaBenchmark(valid_config_file)
    benchmark.load_models()
    benchmark.load_tasks()

    results = benchmark.run_benchmarks()

    # Verify that evaluate was called on the task with the model
    mock_task.evaluate.assert_called_once_with(mock_model)

    # Verify that results are structured correctly
    assert "gemma-2b-test" in results
    assert "mmlu-test" in results["gemma-2b-test"]
    assert results["gemma-2b-test"]["mmlu-test"]["accuracy"] == 95.0
