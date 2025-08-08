"""
Unit tests for Tabula-8B model implementation.

Tests core functionality including configuration, data formatting,
and model operations.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from tabula_model import TabulaModel, TabulaConfig, check_environment


class TestTabulaConfig:
    """Test configuration class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TabulaConfig()
        
        assert config.model_name == "mlfoundations/tabula-8b-v1.2"
        assert config.device == "auto"
        assert config.use_8bit == False
        assert config.use_4bit == False
        assert config.max_length == 2048
        assert config.temperature == 0.1
        assert config.top_p == 0.95
        
    def test_custom_config(self):
        """Test custom configuration."""
        config = TabulaConfig(
            device="cpu",
            use_4bit=True,
            max_length=512,
            temperature=0.5
        )
        
        assert config.device == "cpu"
        assert config.use_4bit == True
        assert config.max_length == 512
        assert config.temperature == 0.5
        
    def test_config_validation(self):
        """Test configuration validation."""
        # Test that config accepts various valid values
        config = TabulaConfig(temperature=0.0)
        assert config.temperature == 0.0
        
        config = TabulaConfig(max_length=1)
        assert config.max_length == 1
        
        # Test edge cases
        config = TabulaConfig(temperature=2.0)
        assert config.temperature == 2.0


class TestTabulaModel:
    """Test main model class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance for testing."""
        config = TabulaConfig(device="cpu")
        return TabulaModel(config)
        
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            'age': [25, 32, 45, 28, 56],
            'income': [35000, 52000, 78000, 41000, 92000],
            'category': ['A', 'B', 'A', 'C', 'B'],
            'score': [0.5, 0.8, 0.9, 0.6, 0.95],
            'target': ['Yes', 'No', 'Yes', 'No', 'Yes']
        })
        
    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model.config is not None
        assert model.model is None  # Not loaded yet
        assert model.tokenizer is None  # Not loaded yet
        assert model.device is not None
        
    def test_format_tabular_data_classification(self, model, sample_df):
        """Test data formatting for classification."""
        prompt = model.format_tabular_data(
            sample_df.drop('target', axis=1),
            target_column='target',
            task_type='classification',
            include_column_descriptions=True
        )
        
        # Check prompt structure
        assert "Task: Predict 'target' (classification)" in prompt
        assert "Column Information:" in prompt
        assert "Data:" in prompt
        assert "age,income,category,score" in prompt
        
        # Check column descriptions
        assert "age: int64" in prompt
        assert "category: object (unique: 3)" in prompt
        
    def test_format_tabular_data_regression(self, model, sample_df):
        """Test data formatting for regression."""
        prompt = model.format_tabular_data(
            sample_df.drop('income', axis=1),
            target_column='income',
            task_type='regression',
            include_column_descriptions=False
        )
        
        # Check prompt structure
        assert "Task: Predict 'income' (regression)" in prompt
        assert "Column Information:" not in prompt
        assert "Data:" in prompt
        
    def test_format_tabular_data_no_target(self, model, sample_df):
        """Test data formatting without target column."""
        prompt = model.format_tabular_data(
            sample_df,
            target_column=None,
            task_type='classification'
        )
        
        assert "Task: Analyze the following tabular data:" in prompt
        assert "Predict the" not in prompt
        
    def test_parse_predictions_classification(self, model):
        """Test prediction parsing for classification."""
        generated_text = """
        Row 1: class_A
        Row 2: class_B
        Row 3: class_A
        """
        
        predictions = model._parse_predictions(
            generated_text,
            task_type='classification',
            expected_count=3
        )
        
        assert len(predictions) == 3
        assert predictions[0] == 'class_A'
        assert predictions[1] == 'class_B'
        assert predictions[2] == 'class_A'
        
    def test_parse_predictions_regression(self, model):
        """Test prediction parsing for regression."""
        generated_text = """
        Prediction 1: 42.5
        Prediction 2: 38.2
        Prediction 3: 51.7
        """
        
        predictions = model._parse_predictions(
            generated_text,
            task_type='regression',
            expected_count=3
        )
        
        assert len(predictions) == 3
        assert predictions[0] == 42.5
        assert predictions[1] == 38.2
        assert predictions[2] == 51.7
        
    def test_parse_predictions_padding(self, model):
        """Test prediction parsing with padding."""
        generated_text = "Row 1: value_1"
        
        predictions = model._parse_predictions(
            generated_text,
            task_type='classification',
            expected_count=3
        )
        
        assert len(predictions) == 3
        assert predictions[0] == 'value_1'
        assert predictions[1] is None
        assert predictions[2] is None
        
    def test_parse_predictions_trimming(self, model):
        """Test prediction parsing with trimming."""
        generated_text = """
        Row 1: A
        Row 2: B
        Row 3: C
        Row 4: D
        Row 5: E
        """
        
        predictions = model._parse_predictions(
            generated_text,
            task_type='classification',
            expected_count=3
        )
        
        assert len(predictions) == 3
        assert predictions[0] == 'A'
        assert predictions[1] == 'B'
        assert predictions[2] == 'C'
        
    def test_handling_missing_values(self, model):
        """Test handling of missing values in data."""
        df_with_missing = pd.DataFrame({
            'col1': [1, 2, None, 4],
            'col2': ['A', None, 'C', 'D'],
            'target': ['X', 'Y', 'X', 'Y']
        })
        
        prompt = model.format_tabular_data(
            df_with_missing.drop('target', axis=1),
            target_column='target',
            task_type='classification'
        )
        
        # Check that missing values are mentioned
        assert "(missing: " in prompt
        
    def test_predict_without_model_loaded(self, model, sample_df):
        """Test that predict raises error when model not loaded."""
        with pytest.raises(ValueError, match="Model not loaded"):
            model.predict(
                sample_df.drop('target', axis=1),
                target_column='target',
                task_type='classification'
            )


class TestEnvironmentCheck:
    """Test environment checking functionality."""
    
    def test_check_environment(self):
        """Test environment check function."""
        env_info = check_environment()
        
        # Check required keys
        assert 'python_version' in env_info
        assert 'cuda_available' in env_info
        assert 'torch_version' in env_info
        assert 'device_count' in env_info
        
        # Check types
        assert isinstance(env_info['cuda_available'], bool)
        assert isinstance(env_info['device_count'], int)
        
        # If CUDA available, check additional info
        if env_info['cuda_available']:
            assert 'gpu_name' in env_info
            assert 'gpu_memory_gb' in env_info
            assert env_info['gpu_memory_gb'] > 0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def model(self):
        """Create model instance for testing."""
        config = TabulaConfig(device="cpu")
        return TabulaModel(config)
        
    def test_empty_dataframe(self, model):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        
        prompt = model.format_tabular_data(
            empty_df,
            target_column='target',
            task_type='classification'
        )
        
        assert "Data:" in prompt
        
    def test_single_column_dataframe(self, model):
        """Test handling of single column DataFrame."""
        single_col_df = pd.DataFrame({'col1': [1, 2, 3]})
        
        prompt = model.format_tabular_data(
            single_col_df,
            target_column='target',
            task_type='classification'
        )
        
        assert "col1" in prompt
        
    def test_large_dataframe(self, model):
        """Test handling of large DataFrame."""
        large_df = pd.DataFrame(
            np.random.randn(1000, 50),
            columns=[f'col_{i}' for i in range(50)]
        )
        
        prompt = model.format_tabular_data(
            large_df,
            target_column='target',
            task_type='regression',
            include_column_descriptions=False
        )
        
        # Should handle without error
        assert len(prompt) > 0
        assert "Data:" in prompt
        
    def test_special_characters_in_columns(self, model):
        """Test handling of special characters in column names."""
        special_df = pd.DataFrame({
            'col-1': [1, 2, 3],
            'col.2': [4, 5, 6],
            'col 3': [7, 8, 9],
            'col@4': [10, 11, 12]
        })
        
        prompt = model.format_tabular_data(
            special_df,
            target_column='target',
            task_type='classification'
        )
        
        # Should handle special characters
        assert 'col-1' in prompt
        assert 'col.2' in prompt
        assert 'col 3' in prompt
        assert 'col@4' in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])