"""Tests for core functionality."""

import pytest
import tempfile
import os
import yaml
from unittest.mock import Mock, patch

from gemma_benchmark.core.benchmark import GemmaBenchmark
from gemma_benchmark.core.model_loader import ModelWrapper


class TestGemmaBenchmark:
    """Test the core benchmark orchestration."""
    
    def test_init_with_config(self):
        """Test benchmark initialization with config file."""
        config_data = {
            "models": {
                "test-model": {
                    "type": "gemma",
                    "size": "2b"
                }
            },
            "tasks": {
                "mmlu": {
                    "type": "mmlu",
                    "subset": "all"
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            benchmark = GemmaBenchmark(config_path)
            assert benchmark.config == config_data
            assert benchmark.models == {}
            assert benchmark.tasks == {}
            assert benchmark.results == {}
        finally:
            os.unlink(config_path)
    
    def test_load_config_file_not_found(self):
        """Test handling of missing config file."""
        with pytest.raises(FileNotFoundError):
            GemmaBenchmark("nonexistent_config.yaml")
    
    def test_load_config_invalid_yaml(self):
        """Test handling of invalid YAML config."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name
        
        try:
            with pytest.raises(yaml.YAMLError):
                GemmaBenchmark(config_path)
        finally:
            os.unlink(config_path)


class TestModelWrapper:
    """Test the model wrapper functionality."""
    
    def test_model_wrapper_init(self):
        """Test model wrapper initialization."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        wrapper = ModelWrapper("test-model", mock_model, mock_tokenizer)
        
        assert wrapper.model_name == "test-model"
        assert wrapper.model == mock_model
        assert wrapper.tokenizer == mock_tokenizer
    
    def test_generate_with_none_model(self):
        """Test generate method with None model."""
        wrapper = ModelWrapper("test-model", None, None)
        
        with pytest.raises(ValueError, match="Model or tokenizer not loaded"):
            wrapper.generate("test prompt")
    
    @patch('torch.cuda.is_available')
    def test_generate_mock_model(self, mock_cuda):
        """Test generate method with mock model and tokenizer."""
        mock_cuda.return_value = False
        
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        # Setup mock tokenizer
        mock_tokenizer.return_value = {"input_ids": Mock()}
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.decode.return_value = "Generated text"
        
        # Setup mock model
        mock_output = Mock()
        mock_output.__getitem__.return_value = [1, 2, 3, 4, 5]  # Mock token IDs
        mock_model.generate.return_value = [mock_output]
        
        wrapper = ModelWrapper("test-model", mock_model, mock_tokenizer)
        
        result = wrapper.generate("test prompt", max_new_tokens=10)
        
        assert result == "Generated text"
        mock_model.generate.assert_called_once()
        mock_tokenizer.decode.assert_called_once()


@pytest.mark.unit
class TestConfigValidation:
    """Test configuration validation."""
    
    def test_valid_model_config(self):
        """Test valid model configuration."""
        config = {
            "models": {
                "gemma-2b": {
                    "type": "gemma",
                    "size": "2b",
                    "variant": "it"
                }
            }
        }
        # This would be expanded with actual validation logic
        assert "models" in config
        assert "gemma-2b" in config["models"]
    
    def test_valid_task_config(self):
        """Test valid task configuration."""
        config = {
            "tasks": {
                "mmlu": {
                    "type": "mmlu",
                    "subset": "all",
                    "shot_count": 5
                }
            }
        }
        # This would be expanded with actual validation logic
        assert "tasks" in config
        assert "mmlu" in config["tasks"]


@pytest.mark.integration
class TestBenchmarkIntegration:
    """Integration tests for benchmark functionality."""
    
    @pytest.fixture
    def sample_config_file(self):
        """Create a sample config file for testing."""
        config_data = {
            "models": {
                "mock-model": {
                    "type": "mock",
                    "size": "small"
                }
            },
            "tasks": {
                "mock-task": {
                    "type": "mock",
                    "subset": "test"
                }
            },
            "output": {
                "path": "test_results"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            yield f.name
        
        os.unlink(f.name)
    
    def test_benchmark_initialization(self, sample_config_file):
        """Test full benchmark initialization."""
        benchmark = GemmaBenchmark(sample_config_file)
        
        assert benchmark.config is not None
        assert "models" in benchmark.config
        assert "tasks" in benchmark.config


if __name__ == "__main__":
    pytest.main([__file__])
