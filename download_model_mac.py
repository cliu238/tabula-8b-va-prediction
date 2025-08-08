#!/usr/bin/env python
"""
Download Tabula-8B model for Apple Silicon (M3 Max)
Optimized for macOS without bitsandbytes
"""

import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def download_tabula_model():
    """Download and cache Tabula-8B model for M3 Max."""
    
    model_name = "mlfoundations/tabula-8b"
    
    print("="*60)
    print("Downloading Tabula-8B Model for Apple M3 Max")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Device: Apple M3 Max with 48GB RAM")
    print(f"Configuration: Using MPS (Metal Performance Shaders)")
    print(f"Expected download size: ~16GB")
    print("="*60)
    
    try:
        # Step 1: Download tokenizer (small, quick)
        print("\n[1/2] Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        print("✓ Tokenizer downloaded successfully")
        
        # Step 2: Download model
        print("\n[2/2] Downloading model (this may take 20-40 minutes)...")
        print("Progress will update as each model shard downloads...")
        print("Tip: The progress bar may appear stuck at 0% initially - this is normal\n")
        
        # For M3 Max, we'll use float16 for memory efficiency
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use float16 for memory efficiency
            device_map="auto",  # Will use MPS on M3
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        print("\n✓ Model downloaded and loaded successfully!")
        print(f"✓ Model cached at: ~/.cache/huggingface/hub/")
        
        # Check if MPS is available
        if torch.backends.mps.is_available():
            print("✓ MPS (Metal Performance Shaders) is available for acceleration")
            device = torch.device("mps")
        else:
            print("ℹ Using CPU (MPS not available)")
            device = torch.device("cpu")
        
        # Test the model
        print("\n" + "="*60)
        print("Testing model with sample input...")
        test_input = "The patient is a 65 year old male with fever and cough. The likely diagnosis is:"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        # Move inputs to device
        if device.type == "mps":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.1,
                do_sample=True,
                top_p=0.95
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Input: {test_input}")
        print(f"Model response: {response}")
        print("\n✓ Model is working correctly!")
        
        # Print memory usage
        print("\n" + "="*60)
        print("Model Information:")
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {param_count/1e9:.1f}B")
        print(f"Model dtype: {model.dtype}")
        print(f"Device: {device}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n⚠ Download interrupted by user")
        print("Run this script again to resume download")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Ensure you have at least 40GB free disk space")
        print("3. If out of memory, try closing other applications")
        print("4. Try setting: export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0")
        sys.exit(1)

if __name__ == "__main__":
    print("Starting Tabula-8B download for Apple Silicon...")
    print("Note: You can interrupt with Ctrl+C and resume later\n")
    
    # Set environment variable for better MPS memory management
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    
    download_tabula_model()