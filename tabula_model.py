"""
Tabula-8B Model Implementation

This module provides the core implementation for loading and using the Tabula-8B model
for zero-shot tabular prediction tasks.

Based on the paper: https://arxiv.org/pdf/2406.12031
Model: https://huggingface.co/mlfoundations/tabula-8b
"""

import os
import json
import warnings
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from pydantic import BaseModel, Field
from tqdm import tqdm


class TabulaConfig(BaseModel):
    """Configuration for Tabula-8B model."""
    
    model_name: str = Field(
        default="mlfoundations/tabula-8b-v1.2",
        description="Hugging Face model identifier"
    )
    device: str = Field(
        default="auto",
        description="Device to use: 'cuda', 'cpu', or 'auto'"
    )
    use_8bit: bool = Field(
        default=False,
        description="Use 8-bit quantization for reduced memory"
    )
    use_4bit: bool = Field(
        default=False,
        description="Use 4-bit quantization for minimal memory"
    )
    max_length: int = Field(
        default=2048,
        description="Maximum sequence length"
    )
    temperature: float = Field(
        default=0.1,
        description="Temperature for generation"
    )
    top_p: float = Field(
        default=0.95,
        description="Top-p for nucleus sampling"
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Directory to cache the model"
    )


class TabulaModel:
    """
    Main class for Tabula-8B model operations.
    
    This class handles model loading, data preprocessing, and inference
    for zero-shot tabular prediction tasks.
    """
    
    def __init__(self, config: Optional[TabulaConfig] = None):
        """
        Initialize Tabula-8B model.
        
        Args:
            config: Configuration object for the model
        """
        self.config = config or TabulaConfig()
        self.model = None
        self.tokenizer = None
        self._setup_device()
        
    def _setup_device(self):
        """Setup the computation device."""
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
            
        if self.device.type == "cuda":
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("Using CPU - Note: Inference will be slower")
            
    def load_model(self):
        """
        Load the Tabula-8B model and tokenizer.
        
        This method handles different quantization options and memory requirements.
        """
        print(f"Loading model: {self.config.model_name}")
        
        # Configure quantization if needed
        quantization_config = None
        if self.config.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            print("Using 4-bit quantization (memory efficient)")
        elif self.config.use_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
            print("Using 8-bit quantization")
            
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir,
            trust_remote_code=True
        )
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model with appropriate configuration
        model_kwargs = {
            "cache_dir": self.config.cache_dir,
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device.type == "cuda" else torch.float32,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        elif self.device.type == "cuda":
            model_kwargs["device_map"] = "auto"
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        # Move to device if not using device_map
        if not quantization_config and self.device.type == "cpu":
            self.model = self.model.to(self.device)
            
        self.model.eval()
        print("Model loaded successfully!")
        
    def format_tabular_data(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        task_type: str = "classification",
        include_column_descriptions: bool = True
    ) -> str:
        """
        Format tabular data for Tabula-8B input.
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column for prediction
            task_type: Type of task ('classification' or 'regression')
            include_column_descriptions: Whether to include column metadata
            
        Returns:
            Formatted string prompt for the model
        """
        # Build the prompt following RTFM methodology
        prompt_parts = []
        
        # Task description
        if target_column:
            prompt_parts.append(
                f"Task: Predict '{target_column}' ({task_type}) based on the following data:\n"
            )
        else:
            prompt_parts.append("Task: Analyze the following tabular data:\n")
            
        # Column descriptions if requested
        if include_column_descriptions:
            prompt_parts.append("\nColumn Information:")
            for col in df.columns:
                if col != target_column:
                    dtype = str(df[col].dtype)
                    unique_count = df[col].nunique()
                    null_count = df[col].isnull().sum()
                    
                    col_info = f"- {col}: {dtype}"
                    if dtype == "object":
                        col_info += f" (unique: {unique_count})"
                    if null_count > 0:
                        col_info += f" (missing: {null_count})"
                    prompt_parts.append(col_info)
                    
        # Add the actual data
        prompt_parts.append("\nData:")
        prompt_parts.append(df.to_csv(index=False))
        
        # Add prediction request
        if target_column:
            prompt_parts.append(f"\nPredict the '{target_column}' value for each row.")
            
        return "\n".join(prompt_parts)
        
    def predict(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str = "classification",
        batch_size: int = 1,
        return_probabilities: bool = False
    ) -> Union[List[Any], Dict[str, Any]]:
        """
        Make predictions on tabular data.
        
        Args:
            df: Input DataFrame
            target_column: Column to predict
            task_type: 'classification' or 'regression'
            batch_size: Number of samples to process at once
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Predictions or dictionary with predictions and probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        predictions = []
        probabilities = []
        
        # Process in batches
        for i in tqdm(range(0, len(df), batch_size), desc="Predicting"):
            batch_df = df.iloc[i:i+batch_size]
            
            # Format input
            prompt = self.format_tabular_data(
                batch_df,
                target_column,
                task_type
            )
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
                padding=True
            ).to(self.device)
            
            # Generate predictions
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            # Parse predictions from generated text
            batch_predictions = self._parse_predictions(
                generated_text,
                task_type,
                len(batch_df)
            )
            predictions.extend(batch_predictions)
            
        if return_probabilities and task_type == "classification":
            return {"predictions": predictions, "probabilities": probabilities}
        return predictions
        
    def _parse_predictions(
        self,
        generated_text: str,
        task_type: str,
        expected_count: int
    ) -> List[Any]:
        """
        Parse predictions from generated text.
        
        Args:
            generated_text: Raw output from the model
            task_type: Type of prediction task
            expected_count: Expected number of predictions
            
        Returns:
            List of parsed predictions
        """
        # Simple parsing logic - can be enhanced based on specific output format
        lines = generated_text.strip().split('\n')
        predictions = []
        
        for line in lines:
            if ':' in line or '=' in line:
                # Extract value after delimiter
                parts = line.split(':' if ':' in line else '=')
                if len(parts) > 1:
                    value = parts[-1].strip()
                    
                    if task_type == "regression":
                        try:
                            predictions.append(float(value))
                        except ValueError:
                            continue
                    else:
                        predictions.append(value)
                        
        # Pad or trim to expected count
        if len(predictions) < expected_count:
            predictions.extend([None] * (expected_count - len(predictions)))
        elif len(predictions) > expected_count:
            predictions = predictions[:expected_count]
            
        return predictions
        
    def fine_tune(
        self,
        train_df: pd.DataFrame,
        target_column: str,
        validation_df: Optional[pd.DataFrame] = None,
        output_dir: str = "./fine_tuned_model",
        num_epochs: int = 3
    ):
        """
        Fine-tune Tabula-8B on specific tabular data.
        
        Note: This is a placeholder for fine-tuning functionality.
        The base Tabula-8B achieves strong zero-shot performance,
        but task-specific fine-tuning can improve results.
        
        Args:
            train_df: Training data
            target_column: Target column name
            validation_df: Optional validation data
            output_dir: Directory to save fine-tuned model
            num_epochs: Number of training epochs
        """
        raise NotImplementedError(
            "Fine-tuning implementation requires additional setup. "
            "Tabula-8B is designed for zero-shot performance. "
            "See the paper for fine-tuning strategies."
        )


def check_environment():
    """
    Check if the environment is properly configured for Tabula-8B.
    
    Returns:
        Dictionary with environment information
    """
    env_info = {
        "python_version": os.sys.version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if env_info["cuda_available"]:
        env_info["gpu_name"] = torch.cuda.get_device_name(0)
        env_info["gpu_memory_gb"] = (
            torch.cuda.get_device_properties(0).total_memory / 1e9
        )
        
    return env_info


if __name__ == "__main__":
    # Quick environment check
    print("Tabula-8B Environment Check")
    print("-" * 40)
    env_info = check_environment()
    for key, value in env_info.items():
        print(f"{key}: {value}")