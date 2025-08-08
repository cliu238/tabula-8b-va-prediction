#!/usr/bin/env python
"""
Test script for PHMRC data preprocessing and serialization
Tests with 100 sample records without requiring the model
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.preprocessor import PHMRCPreprocessor
from src.data.serializer import VADataSerializer


def test_preprocessing():
    """Test data preprocessing on PHMRC data."""
    
    print("="*60)
    print("Testing PHMRC Data Preprocessing")
    print("="*60)
    
    # Check if data file exists
    data_path = Path("data/raw/PHMRC/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv")
    
    if not data_path.exists():
        print(f"❌ Data file not found at: {data_path}")
        print("\nPlease ensure the PHMRC data is downloaded to:")
        print("  data/raw/PHMRC/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv")
        return False
    
    try:
        # Initialize preprocessor
        print("\n1. Initializing preprocessor...")
        preprocessor = PHMRCPreprocessor()
        
        # Load data
        print("\n2. Loading PHMRC data...")
        df = preprocessor.load_data(str(data_path))
        
        # Take sample
        print("\n3. Taking sample of 100 records...")
        sample_df = df.sample(n=min(100, len(df)), random_state=42)
        print(f"Sample size: {len(sample_df)}")
        
        # Preprocess
        print("\n4. Preprocessing data...")
        processed_df = preprocessor.preprocess(sample_df)
        
        # Show statistics
        print("\n5. Dataset statistics:")
        stats = preprocessor.create_summary_stats(processed_df)
        for key, value in stats.items():
            if key != 'top_causes':
                print(f"  {key}: {value}")
        
        if 'top_causes' in stats:
            print("\n  Top 5 causes of death in sample:")
            for cause, count in list(stats['top_causes'].items())[:5]:
                print(f"    - {cause}: {count}")
        
        # Test serialization
        print("\n6. Testing serialization...")
        serializer = VADataSerializer(verbose=False)
        
        # Serialize first 5 records
        print("\n  Sample serialized records:")
        for i in range(min(5, len(processed_df))):
            row = processed_df.iloc[i]
            text = serializer.serialize_row(row)
            print(f"\n  Record {i+1}:")
            print(f"    {text[:200]}..." if len(text) > 200 else f"    {text}")
            if 'cause_of_death' in row:
                print(f"    Target: {row['cause_of_death']}")
        
        # Test batch serialization
        print("\n7. Testing batch serialization...")
        all_serialized = serializer.serialize_batch(processed_df, show_progress=True)
        print(f"✓ Serialized {len(all_serialized)} records")
        
        # Show sample prompt
        print("\n8. Sample model prompt:")
        sample_prompt = serializer.create_prompt(all_serialized[0])
        print("-"*40)
        print(sample_prompt)
        print("-"*40)
        
        print("\n✅ All preprocessing tests passed!")
        
        # Save processed sample for later use
        output_path = Path("data/processed/sample_100.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        processed_df.to_csv(output_path, index=False)
        print(f"\n✓ Saved processed sample to: {output_path}")
        
        # Save serialized text
        text_output_path = Path("data/processed/sample_100_serialized.txt")
        with open(text_output_path, 'w') as f:
            for text in all_serialized:
                f.write(text + "\n")
        print(f"✓ Saved serialized text to: {text_output_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("Starting PHMRC preprocessing test...")
    print("This will process 100 sample records without requiring the model.\n")
    
    success = test_preprocessing()
    
    if success:
        print("\n" + "="*60)
        print("SUCCESS: Data preprocessing is working correctly!")
        print("="*60)
        print("\nNext steps:")
        print("1. Download the model: poetry run python download_model_mac.py")
        print("2. Run inference on the processed data")
    else:
        print("\n" + "="*60)
        print("FAILED: Please check the errors above")
        print("="*60)


if __name__ == "__main__":
    main()