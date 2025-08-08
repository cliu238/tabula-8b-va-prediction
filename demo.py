#!/usr/bin/env python
"""
Demo script for Tabula-8B VA Cause of Death Prediction

Demonstrates the complete pipeline from data loading to predictions.
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.preprocessor import PHMRCPreprocessor
from src.data.tabula_formatter import TabulaVAFormatter


def check_model_available():
    """Check if Tabula-8B model is available."""
    try:
        from tabula_model import TabulaModel, TabulaConfig
        import torch
        return True
    except ImportError:
        return False


def load_model():
    """Load Tabula-8B model with appropriate configuration."""
    from tabula_model import TabulaModel, TabulaConfig
    import torch
    
    print("\n" + "="*60)
    print("Loading Tabula-8B Model")
    print("="*60)
    
    # Check device availability
    # Note: MPS has issues with some operations, using CPU for stability
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("ℹ Using CPU for inference")
        if torch.backends.mps.is_available():
            print("  (MPS available but using CPU for stability)")
    
    # Configure model for available hardware
    config = TabulaConfig(
        model_name="mlfoundations/tabula-8b",
        device=device,
        use_4bit=False,  # Set to True if memory constrained
        use_8bit=False,  # Not available on macOS
        max_length=2048,
        temperature=0.1,
        top_p=0.95
    )
    
    # Initialize model
    model = TabulaModel(config)
    
    try:
        # Load model (will download if not cached)
        print("Loading model weights (this may take a moment)...")
        model.load_model()
        print("✓ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("\nTo download the model, run:")
        print("  poetry run python download_model_mac.py")
        return None


def run_predictions(model, processed_df: pd.DataFrame, formatter: TabulaVAFormatter, batch_size: int = 4) -> List[Dict]:
    """Run predictions using CSV format for Tabula-8B."""
    from tqdm import tqdm
    import torch
    
    predictions = []
    
    print("\n" + "="*60)
    print("Running Predictions")
    print("="*60)
    
    for i in tqdm(range(0, len(processed_df), batch_size), desc="Processing batches"):
        batch_df = processed_df.iloc[i:i+batch_size]
        
        for idx, row in batch_df.iterrows():
            try:
                # Create CSV format for single prediction
                row_df = pd.DataFrame([row])
                row_df = row_df.drop(columns=['cause_of_death', 'cause_of_death_code'], errors='ignore')
                
                # Format as CSV for Tabula-8B
                prompt = (
                    "Task: Predict cause of death from verbal autopsy data.\n\n"
                    "Data (CSV format):\n"
                    f"{row_df.to_csv(index=False)}\n"
                    "Predict the cause of death. Common causes include: "
                    "TB, AIDS/HIV, Malaria, Cardiovascular disease, Respiratory infections.\n\n"
                    "Cause of death:"
                )
                
                # Tokenize the prompt
                inputs = model.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                )
                
                # Move to appropriate device
                if hasattr(model, 'device'):
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # Generate prediction
                with torch.no_grad():
                    outputs = model.model.generate(
                        **inputs,
                        max_new_tokens=20,  # Short response for cause name
                        temperature=0.1,
                        do_sample=True,
                        top_p=0.95,
                        pad_token_id=model.tokenizer.pad_token_id
                    )
                
                # Decode the output
                generated_text = model.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                # Extract just the cause (first line, remove extra text)
                cause = generated_text.split('\n')[0].strip()
                if ':' in cause:
                    cause = cause.split(':')[-1].strip()
                
                # Get a preview of the CSV data
                csv_preview = row_df.to_csv(index=False)
                csv_preview = csv_preview[:100] + "..." if len(csv_preview) > 100 else csv_preview
                
                predictions.append({
                    'input': csv_preview,
                    'prediction': cause if cause else 'Unknown',
                    'confidence': 0.85  # Placeholder confidence
                })
                
            except Exception as e:
                print(f"\nError processing record {i}: {str(e)[:100]}")
                predictions.append({
                    'input': 'Error loading data',
                    'prediction': 'Error',
                    'confidence': 0.0
                })
    
    return predictions


def evaluate_predictions(df: pd.DataFrame, predictions: List[Dict]) -> Dict:
    """Evaluate prediction accuracy."""
    if 'cause_of_death' not in df.columns:
        return {}
    
    true_labels = df['cause_of_death'].tolist()
    pred_labels = [p['prediction'] for p in predictions]
    
    # Calculate accuracy
    correct = sum(1 for true, pred in zip(true_labels, pred_labels) 
                  if true.lower() == pred.lower())
    accuracy = correct / len(true_labels) if true_labels else 0
    
    # Get unique causes
    unique_true = set(true_labels)
    unique_pred = set(pred_labels)
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': len(true_labels),
        'unique_true_causes': len(unique_true),
        'unique_predicted_causes': len(unique_pred)
    }


def run_demo(sample_size: int = 10, model_only: bool = False):
    """Run the complete demo pipeline."""
    
    print("\n" + "="*60)
    print("Tabula-8B VA Cause of Death Prediction Demo")
    print("="*60)
    
    # Check data file
    data_path = Path("data/raw/PHMRC/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv")
    if not data_path.exists():
        print(f"❌ Data file not found at: {data_path}")
        print("\nPlease download the PHMRC data first.")
        return
    
    if not model_only:
        # Step 1: Load and preprocess data
        print("\n[Step 1/4] Loading and preprocessing data...")
        preprocessor = PHMRCPreprocessor()
        df = preprocessor.load_data(str(data_path))
        
        # Sample data
        sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
        print(f"✓ Sampled {len(sample_df)} records")
        
        # Preprocess
        processed_df = preprocessor.preprocess(sample_df)
        print(f"✓ Preprocessed to {len(processed_df.columns)} features")
        
        # Step 2: Serialize to text
        print("\n[Step 2/4] Converting to natural language...")
        serializer = VADataSerializer(verbose=False)
        serialized_texts = serializer.serialize_batch(processed_df, show_progress=True)
        print(f"✓ Serialized {len(serialized_texts)} records")
        
        # Show sample
        print("\nSample serialized record:")
        print("-" * 40)
        print(serialized_texts[0][:300] + "..." if len(serialized_texts[0]) > 300 else serialized_texts[0])
        print("-" * 40)
        
        # Save preprocessed data
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        processed_df.to_csv(output_dir / f"demo_sample_{sample_size}.csv", index=False)
        print(f"\n✓ Saved processed data to: {output_dir / f'demo_sample_{sample_size}.csv'}")
        
        with open(output_dir / f"demo_serialized_{sample_size}.txt", 'w') as f:
            for text in serialized_texts:
                f.write(text + "\n")
        print(f"✓ Saved serialized text to: {output_dir / f'demo_serialized_{sample_size}.txt'}")
    
    # Step 3: Load model and run predictions
    print("\n[Step 3/4] Loading Tabula-8B model...")
    
    if not check_model_available():
        print("❌ Model dependencies not available")
        print("\nPlease install transformers and torch:")
        print("  poetry install")
        return
    
    model = load_model()
    if model is None:
        print("\n⚠ Model not loaded. Skipping predictions.")
        print("\nTo continue with predictions:")
        print("1. Download the model: poetry run python download_model_mac.py")
        print("2. Run demo again: poetry run python demo.py")
        return
    
    # Step 4: Run predictions
    print("\n[Step 4/4] Running predictions...")
    
    if model_only:
        # Load pre-processed data if running model-only
        processed_path = Path("data/processed") / f"demo_sample_{sample_size}.csv"
        if not processed_path.exists():
            print(f"❌ No preprocessed data found. Run without --model-only first.")
            return
        
        processed_df = pd.read_csv(processed_path)
        formatter = TabulaVAFormatter()
    
    predictions = run_predictions(model, processed_df, formatter, batch_size=2)
    
    # Display results
    print("\n" + "="*60)
    print("Prediction Results")
    print("="*60)
    
    for i, pred in enumerate(predictions[:5]):  # Show first 5
        print(f"\nRecord {i+1}:")
        print(f"  Input: {pred['input']}")
        print(f"  Prediction: {pred['prediction']}")
        print(f"  Confidence: {pred['confidence']:.2%}")
        
        if processed_df is not None and 'cause_of_death' in processed_df.columns:
            true_cause = processed_df.iloc[i]['cause_of_death']
            print(f"  True cause: {true_cause}")
            match = "✓" if pred['prediction'].lower() == true_cause.lower() else "✗"
            print(f"  Match: {match}")
    
    # Evaluate if ground truth available
    if processed_df is not None and 'cause_of_death' in processed_df.columns:
        metrics = evaluate_predictions(processed_df, predictions)
        
        print("\n" + "="*60)
        print("Evaluation Metrics")
        print("="*60)
        print(f"Accuracy: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")
        print(f"Unique true causes: {metrics['unique_true_causes']}")
        print(f"Unique predicted causes: {metrics['unique_predicted_causes']}")
    
    # Save predictions
    predictions_df = pd.DataFrame(predictions)
    predictions_path = Path("data/processed") / f"demo_predictions_{sample_size}.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"\n✓ Saved predictions to: {predictions_path}")
    
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Tabula-8B VA Prediction Demo")
    parser.add_argument(
        "--sample",
        type=int,
        default=10,
        help="Number of samples to process (default: 10)"
    )
    parser.add_argument(
        "--model-only",
        action="store_true",
        help="Skip preprocessing and use cached data"
    )
    
    args = parser.parse_args()
    
    try:
        run_demo(sample_size=args.sample, model_only=args.model_only)
    except KeyboardInterrupt:
        print("\n\n⚠ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()