"""Tests for benchmark tasks."""

import pytest
from unittest.mock import Mock, patch
import tempfile
import sys

from gemma_benchmark.tasks.mmlu import MMLUBenchmark
from gemma_benchmark.tasks.efficiency import EfficiencyBenchmark
from gemma_benchmark.tasks.gsm8k import Gsm8kBenchmark
from gemma_benchmark.core.model_loader import ModelWrapper


class TestMMLUBenchmark:
    """Test MMLU benchmark implementation."""
    
    def test_init(self):
        """Test MMLU benchmark initialization."""
        config = {
            "subset": "mathematics",
            "shot_count": 3
        }
        
        benchmark = MMLUBenchmark(config)
        
        assert benchmark.subset == "mathematics"
        assert benchmark.shot_count == 3
        assert benchmark.data is None
    
    def test_format_prompt(self):
        """Test MMLU prompt formatting."""
        config = {"subset": "all", "shot_count": 5}
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
                    "answer": 1
                }
            ]
        }
        
        prompt = benchmark.format_prompt(example)
        
        assert "mathematics" in prompt
        assert "What is 2+2?" in prompt
        assert "A. 3" in prompt
        assert "B. 4" in prompt
        assert "What is 1+1?" in prompt  # Few-shot example
    
    @patch('datasets.load_dataset')
    def test_load_data_success(self, mock_load_dataset):
        """Test successful data loading."""
        # Mock dataset structure
        mock_dataset = {
            "test": {
                "subject": ["math", "math", "science"],
                "question": ["Q1", "Q2", "Q3"],
                "choices": [["A", "B"], ["C", "D"], ["E", "F"]],
                "answer": [0, 1, 0]
            },
            "dev": {
                "subject": ["math", "science"],
                "question": ["Q_dev1", "Q_dev2"],
                "choices": [["A_dev", "B_dev"], ["C_dev", "D_dev"]],
                "answer": [1, 0]
            }
        }
        mock_load_dataset.return_value = mock_dataset
        
        config = {"subset": "math", "shot_count": 1}
        benchmark = MMLUBenchmark(config)
        
        data = benchmark.load_data()
        
        assert "math" in data
        assert len(data["math"]) == 2  # Two math questions
        assert data["math"][0]["question"] == "Q1"
    
    @patch('datasets.load_dataset')
    def test_load_data_fallback_to_mock(self, mock_load_dataset):
        """Test fallback to mock data when loading fails."""
        mock_load_dataset.side_effect = Exception("Network error")
        
        config = {"subset": "all", "shot_count": 5}
        benchmark = MMLUBenchmark(config)
        
        data = benchmark.load_data()
        
        # Should return mock data
        assert isinstance(data, dict)
        assert len(data) > 0


class TestEfficiencyBenchmark:
    """Test efficiency benchmark implementation."""
    
    def test_init(self):
        """Test efficiency benchmark initialization."""
        config = {
            "sample_prompts": ["Test prompt 1", "Test prompt 2"],
            "output_lengths": [64, 128]
        }
        
        benchmark = EfficiencyBenchmark(config)
        
        assert len(benchmark.sample_prompts) == 2
        assert benchmark.output_lengths == [64, 128]
        assert "os" in benchmark.system_info
    
    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_count')
    @patch('platform.system')
    def test_get_system_info(self, mock_system, mock_cpu_count, mock_memory):
        """Test system information gathering."""
        mock_system.return_value = "Linux"
        mock_cpu_count.return_value = 8
        mock_memory.return_value.total = 16 * 1024**3  # 16 GB
        
        config = {"sample_prompts": ["test"], "output_lengths": [128]}
        benchmark = EfficiencyBenchmark(config)
        
        info = benchmark.system_info
        
        assert info["os"] == "Linux"
        assert info["cpu_count"] == 8
        assert info["memory_total"] == 16.0
    
    @patch('time.time')
    @patch('psutil.Process')
    def test_evaluate_mock_model(self, mock_process, mock_time):
        """Test efficiency evaluation with mock model."""
        # Setup time mock
        mock_time.side_effect = [0.0, 1.0, 2.0, 3.0]  # Simulate time progression
        
        # Setup memory mock
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024**3  # 1 GB
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        # Create mock model
        mock_model = Mock(spec=ModelWrapper)
        mock_model.model_name = "test-model"
        mock_model.generate.return_value = "Generated text"
        
        config = {
            "sample_prompts": ["Test prompt"],
            "output_lengths": [128]
        }
        benchmark = EfficiencyBenchmark(config)
        
        results = benchmark.evaluate(mock_model)
        
        assert "latency" in results
        assert "memory_usage" in results
        assert "tokens_per_second" in results
        assert "system_info" in results
        assert "tokens_128" in results["latency"]


class TestGSM8KBenchmark:
    """Test GSM8K benchmark implementation."""
    
    def test_init(self):
        """Test GSM8K benchmark initialization."""
        config = {
            "shot_count": 3,
            "use_chain_of_thought": True
        }
        
        benchmark = Gsm8kBenchmark(config)
        
        assert benchmark.shot_count == 3
        assert benchmark.use_cot is True
        assert benchmark.data is None
    
    def test_extract_numerical_answer(self):
        """Test numerical answer extraction."""
        config = {"shot_count": 0}
        benchmark = Gsm8kBenchmark(config)
        
        # Test with GSM8K format
        answer1 = "Janet has 9 ducks. She eats 3 eggs. So she has 9 - 3 = 6 eggs left. #### 6"
        result1 = benchmark._extract_numerical_answer(answer1)
        assert result1 == 6.0
        
        # Test with comma-separated number
        answer2 = "The total is 1,234 dollars. #### 1,234"
        result2 = benchmark._extract_numerical_answer(answer2)
        assert result2 == 1234.0
        
        # Test with no clear answer
        answer3 = "This is unclear text without numbers"
        result3 = benchmark._extract_numerical_answer(answer3)
        assert str(result3) == "nan"
    
    def test_format_prompt_with_cot(self):
        """Test prompt formatting with chain of thought."""
        config = {"shot_count": 1, "use_chain_of_thought": True}
        benchmark = Gsm8kBenchmark(config)
        
        examples = [{
            "question": "Tom has 5 apples. He eats 2. How many left?",
            "answer": "Tom starts with 5 apples. He eats 2. So 5 - 2 = 3. #### 3",
            "numerical_answer": 3.0
        }]
        
        prompt = benchmark.format_prompt("Mary has 10 cookies. She gives away 4. How many left?", examples)
        
        assert "step by step" in prompt
        assert "Tom has 5 apples" in prompt
        assert "Tom starts with 5 apples. He eats 2. So 5 - 2 = 3. #### 3" in prompt
        assert "Mary has 10 cookies" in prompt
    
    def test_format_prompt_without_cot(self):
        """Test prompt formatting without chain of thought."""
        config = {"shot_count": 1, "use_chain_of_thought": False}
        benchmark = Gsm8kBenchmark(config)
        
        examples = [{
            "question": "Tom has 5 apples. He eats 2. How many left?",
            "answer": "Tom starts with 5 apples. He eats 2. So 5 - 2 = 3. #### 3",
            "numerical_answer": 3.0
        }]
        
        prompt = benchmark.format_prompt("Mary has 10 cookies. She gives away 4. How many left?", examples)
        
        assert "step by step" not in prompt
        assert "3.0" in prompt  # Just the numerical answer
        assert "Tom starts with 5" not in prompt  # No reasoning


@pytest.mark.slow
class TestTasksIntegration:
    """Integration tests for tasks (marked as slow)."""
    
    def test_mmlu_with_mock_model(self):
        """Test MMLU evaluation with mock model."""
        mock_model = Mock(spec=ModelWrapper)
        mock_model.model_name = "test-model"
        mock_model.generate.return_value = "A"  # Always answer A
        
        config = {"subset": "all", "shot_count": 0}
        benchmark = MMLUBenchmark(config)
        
        # Use mock data
        benchmark.data = benchmark._create_mock_data()
        
        # This would be a full evaluation in the real test
        # For now, just ensure no exceptions
        assert benchmark.data is not None
        assert len(benchmark.data) > 0


if __name__ == "__main__":
    pytest.main([__file__])
