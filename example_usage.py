#!/usr/bin/env python
"""
Tabula-8B Example Usage Script

This script demonstrates how to use Tabula-8B for zero-shot tabular prediction.
It includes examples for both classification and regression tasks.

Usage: python example_usage.py
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

from tabula_model import TabulaModel, TabulaConfig


def example_classification():
    """
    Example of using Tabula-8B for classification task.
    Uses the Iris dataset as a simple demonstration.
    """
    print("\n" + "="*60)
    print("CLASSIFICATION EXAMPLE: Iris Dataset")
    print("="*60)
    
    # Load dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = [iris.target_names[i] for i in iris.target]
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Features: {list(iris.feature_names)}")
    print(f"Classes: {list(iris.target_names)}")
    
    # Configure model for classification
    config = TabulaConfig(
        device="cpu",  # Use CPU for demo; change to "cuda" if GPU available
        use_4bit=False,  # Enable for memory-constrained environments
        temperature=0.1,  # Low temperature for more deterministic predictions
    )
    
    # Initialize and load model
    print("\nLoading Tabula-8B model...")
    model = TabulaModel(config)
    
    # NOTE: Model loading is commented out for demo
    # Uncomment the following line to actually load the model:
    # model.load_model()
    
    print("\nModel configuration:")
    print(f"- Device: {config.device}")
    print(f"- Max length: {config.max_length}")
    print(f"- Temperature: {config.temperature}")
    
    # Format data for model input
    prompt = model.format_tabular_data(
        test_df.drop('species', axis=1),
        target_column='species',
        task_type='classification'
    )
    
    print("\nExample prompt (first 500 chars):")
    print(prompt[:500])
    print("...")
    
    # Make predictions (would work with loaded model)
    # predictions = model.predict(
    #     test_df.drop('species', axis=1),
    #     target_column='species',
    #     task_type='classification'
    # )
    
    # For demo, show expected output format
    print("\nExpected prediction format:")
    print("['setosa', 'versicolor', 'virginica', ...]")
    

def example_regression():
    """
    Example of using Tabula-8B for regression task.
    Uses the Diabetes dataset as a demonstration.
    """
    print("\n" + "="*60)
    print("REGRESSION EXAMPLE: Diabetes Dataset")
    print("="*60)
    
    # Load dataset
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df['target'] = diabetes.target
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Features: {list(diabetes.feature_names)}")
    print(f"Target range: [{df['target'].min():.1f}, {df['target'].max():.1f}]")
    
    # Configure model for regression
    config = TabulaConfig(
        device="cpu",
        use_8bit=False,  # Can use 8-bit for moderate memory savings
        temperature=0.3,  # Slightly higher for regression
    )
    
    # Initialize model
    print("\nInitializing Tabula-8B for regression...")
    model = TabulaModel(config)
    
    # Format data
    prompt = model.format_tabular_data(
        test_df.head(5).drop('target', axis=1),
        target_column='target',
        task_type='regression'
    )
    
    print("\nExample prompt structure (truncated):")
    lines = prompt.split('\n')
    for line in lines[:15]:
        print(line)
    print("...")
    

def example_custom_data():
    """
    Example of using Tabula-8B with custom tabular data.
    Shows how to prepare and format arbitrary CSV data.
    """
    print("\n" + "="*60)
    print("CUSTOM DATA EXAMPLE")
    print("="*60)
    
    # Create sample custom data
    custom_data = pd.DataFrame({
        'age': [25, 32, 45, 28, 56, 39],
        'income': [35000, 52000, 78000, 41000, 92000, 66000],
        'education_years': [12, 16, 18, 14, 20, 16],
        'credit_score': [650, 720, 780, 680, 800, 740],
        'loan_approved': ['No', 'Yes', 'Yes', 'No', 'Yes', 'Yes']
    })
    
    print("Custom dataset:")
    print(custom_data)
    
    # Configure model
    config = TabulaConfig(
        device="cpu",
        use_4bit=True,  # Maximum memory efficiency
        max_length=1024,  # Shorter for small datasets
    )
    
    model = TabulaModel(config)
    
    # Format for prediction
    test_row = custom_data.drop('loan_approved', axis=1).iloc[[0]]
    prompt = model.format_tabular_data(
        test_row,
        target_column='loan_approved',
        task_type='classification',
        include_column_descriptions=True
    )
    
    print("\nFormatted input for prediction:")
    print(prompt)
    

def example_batch_processing():
    """
    Example of batch processing large datasets efficiently.
    """
    print("\n" + "="*60)
    print("BATCH PROCESSING EXAMPLE")
    print("="*60)
    
    # Create larger dataset
    n_samples = 1000
    large_df = pd.DataFrame({
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'feature_3': np.random.choice(['A', 'B', 'C'], n_samples),
        'feature_4': np.random.randint(0, 100, n_samples),
        'target': np.random.choice(['class_0', 'class_1'], n_samples)
    })
    
    print(f"Dataset size: {len(large_df)} samples")
    print(f"Memory usage: {large_df.memory_usage().sum() / 1024:.1f} KB")
    
    # Configure for batch processing
    config = TabulaConfig(
        device="cuda" if False else "cpu",  # Set to True if CUDA available
        use_8bit=True,
        max_length=512,  # Smaller context for efficiency
    )
    
    model = TabulaModel(config)
    
    # Demonstrate batch processing approach
    batch_size = 32
    n_batches = (len(large_df) + batch_size - 1) // batch_size
    
    print(f"\nBatch processing configuration:")
    print(f"- Batch size: {batch_size}")
    print(f"- Number of batches: {n_batches}")
    print(f"- Estimated time: ~{n_batches * 2} seconds (CPU)")
    
    # Show how batching works
    for i in range(min(3, n_batches)):
        batch_start = i * batch_size
        batch_end = min(batch_start + batch_size, len(large_df))
        batch_df = large_df.iloc[batch_start:batch_end]
        print(f"\nBatch {i+1}: rows {batch_start}-{batch_end}")
        print(f"  Shape: {batch_df.shape}")


def main():
    """
    Run all examples.
    """
    print("TABULA-8B USAGE EXAMPLES")
    print("========================")
    print("\nThese examples demonstrate how to use Tabula-8B for:")
    print("1. Classification tasks")
    print("2. Regression tasks")
    print("3. Custom data processing")
    print("4. Batch processing of large datasets")
    
    # Check environment
    from tabula_model import check_environment
    print("\nEnvironment Check:")
    env_info = check_environment()
    for key, value in env_info.items():
        print(f"  {key}: {value}")
    
    # Run examples
    example_classification()
    example_regression()
    example_custom_data()
    example_batch_processing()
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Install dependencies: poetry install")
    print("2. Uncomment model.load_model() to load the actual model")
    print("3. Run predictions on your data")
    print("4. Fine-tune for specific tasks if needed")
    

if __name__ == "__main__":
    main()