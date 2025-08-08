#!/usr/bin/env python
"""
Setup script for Tabula-8B environment

This script handles the complete setup process including:
- Dependency installation
- Model download
- Environment verification
- Troubleshooting common issues

Usage: python setup.py [--check-only] [--download-model] [--test]
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
import shutil


def check_python_version():
    """Check if Python version meets requirements."""
    version_info = sys.version_info
    print(f"Python version: {version_info.major}.{version_info.minor}.{version_info.micro}")
    
    if version_info.major < 3 or (version_info.major == 3 and version_info.minor < 9):
        print("ERROR: Python 3.9+ is required")
        return False
    
    print("✓ Python version is compatible")
    return True


def check_poetry():
    """Check if Poetry is installed."""
    try:
        result = subprocess.run(
            ["poetry", "--version"],
            capture_output=True,
            text=True
        )
        print(f"Poetry version: {result.stdout.strip()}")
        print("✓ Poetry is installed")
        return True
    except FileNotFoundError:
        print("ERROR: Poetry is not installed")
        print("Install with: curl -sSL https://install.python-poetry.org | python3 -")
        return False


def install_dependencies():
    """Install project dependencies using Poetry."""
    print("\n" + "="*60)
    print("Installing dependencies...")
    print("="*60)
    
    try:
        # Install dependencies
        subprocess.run(["poetry", "install"], check=True)
        print("✓ Dependencies installed successfully")
        
        # Show installed packages
        result = subprocess.run(
            ["poetry", "show"],
            capture_output=True,
            text=True
        )
        
        print("\nKey packages installed:")
        for line in result.stdout.split('\n'):
            if any(pkg in line for pkg in ['torch', 'transformers', 'pandas', 'numpy']):
                print(f"  - {line.split()[0]} {line.split()[1] if len(line.split()) > 1 else ''}")
                
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install dependencies: {e}")
        return False


def check_gpu():
    """Check GPU availability and CUDA setup."""
    print("\n" + "="*60)
    print("Checking GPU/CUDA configuration...")
    print("="*60)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1e9
                print(f"  GPU {i}: {props.name}")
                print(f"    Memory: {memory_gb:.1f} GB")
                print(f"    Compute capability: {props.major}.{props.minor}")
                
                # Memory requirements for Tabula-8B
                if memory_gb < 16:
                    print(f"    ⚠ Warning: GPU has <16GB memory. Use quantization:")
                    print(f"      - 8-bit: ~8GB required")
                    print(f"      - 4-bit: ~4GB required")
                elif memory_gb < 32:
                    print(f"    ✓ Sufficient for 8-bit or 4-bit quantization")
                else:
                    print(f"    ✓ Sufficient for full precision")
                    
        else:
            print("⚠ CUDA is not available - will use CPU")
            print("  Note: CPU inference will be significantly slower")
            print("  For GPU support, ensure:")
            print("    1. NVIDIA GPU with CUDA support")
            print("    2. CUDA toolkit installed")
            print("    3. PyTorch with CUDA support: poetry add torch --with-cuda")
            
    except ImportError:
        print("ERROR: PyTorch not installed yet")
        print("Run: poetry install")
        return False
        
    return True


def download_model(model_name="mlfoundations/tabula-8b-v1.2"):
    """Download the Tabula-8B model from Hugging Face."""
    print("\n" + "="*60)
    print("Downloading Tabula-8B model...")
    print("="*60)
    
    try:
        from huggingface_hub import snapshot_download
        import torch
        
        cache_dir = Path.home() / ".cache" / "huggingface"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Model: {model_name}")
        print(f"Cache directory: {cache_dir}")
        print("\nNote: This is an 8B parameter model (~16GB download)")
        print("Download time depends on your internet connection...")
        
        # Download model files
        snapshot_download(
            repo_id=model_name,
            cache_dir=str(cache_dir),
            resume_download=True,
            local_files_only=False
        )
        
        print("✓ Model downloaded successfully")
        
        # Check disk space
        cache_size = sum(
            f.stat().st_size for f in cache_dir.rglob('*') if f.is_file()
        ) / 1e9
        
        print(f"Cache size: {cache_size:.1f} GB")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to download model: {e}")
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Ensure sufficient disk space (>20GB recommended)")
        print("3. Try manual download: huggingface-cli download mlfoundations/tabula-8b-v1.2")
        return False


def test_setup():
    """Test the setup with a simple example."""
    print("\n" + "="*60)
    print("Testing Tabula-8B setup...")
    print("="*60)
    
    try:
        # Import required modules
        import torch
        import pandas as pd
        from tabula_model import TabulaModel, TabulaConfig, check_environment
        
        # Check environment
        print("Environment check:")
        env_info = check_environment()
        for key, value in env_info.items():
            print(f"  {key}: {value}")
            
        # Create test configuration
        config = TabulaConfig(
            device="cpu",  # Use CPU for test
            use_4bit=True,  # Use quantization for test
            max_length=512
        )
        
        # Initialize model (without loading weights)
        print("\nInitializing model structure...")
        model = TabulaModel(config)
        print("✓ Model initialized successfully")
        
        # Create test data
        test_df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': ['A', 'B', 'A']
        })
        
        # Test data formatting
        prompt = model.format_tabular_data(
            test_df.drop('target', axis=1),
            target_column='target',
            task_type='classification'
        )
        
        print("\nTest prompt generated:")
        print(prompt[:200] + "...")
        print("\n✓ Setup test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Setup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def troubleshoot_common_issues():
    """Provide solutions for common setup issues."""
    print("\n" + "="*60)
    print("TROUBLESHOOTING GUIDE")
    print("="*60)
    
    issues = {
        "Out of Memory (OOM)": [
            "Use quantization: config.use_4bit=True or config.use_8bit=True",
            "Reduce batch size: batch_size=1",
            "Use CPU instead of GPU: config.device='cpu'",
            "Clear GPU cache: torch.cuda.empty_cache()",
        ],
        "Model Download Fails": [
            "Check disk space: df -h",
            "Use VPN if behind firewall",
            "Set HF_TOKEN environment variable for gated models",
            "Try: huggingface-cli download mlfoundations/tabula-8b-v1.2",
        ],
        "CUDA Not Available": [
            "Check NVIDIA drivers: nvidia-smi",
            "Reinstall PyTorch with CUDA: poetry add torch --with-cuda",
            "Verify CUDA toolkit: nvcc --version",
            "Use CPU fallback: config.device='cpu'",
        ],
        "Import Errors": [
            "Ensure in poetry environment: poetry shell",
            "Reinstall dependencies: poetry install --no-cache",
            "Check Python version: python --version (needs 3.9+)",
            "Clear poetry cache: poetry cache clear pypi --all",
        ],
        "Slow Inference": [
            "Use GPU if available: config.device='cuda'",
            "Enable quantization for faster inference",
            "Reduce max_length: config.max_length=512",
            "Process in larger batches if memory allows",
        ],
    }
    
    for issue, solutions in issues.items():
        print(f"\n{issue}:")
        for solution in solutions:
            print(f"  • {solution}")


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup Tabula-8B environment")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check environment without installing"
    )
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Download the Tabula-8B model"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run setup test"
    )
    parser.add_argument(
        "--troubleshoot",
        action="store_true",
        help="Show troubleshooting guide"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("TABULA-8B SETUP")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
        
    # Check Poetry
    if not check_poetry():
        sys.exit(1)
        
    if args.check_only:
        check_gpu()
        print("\nEnvironment check complete!")
        return
        
    if args.troubleshoot:
        troubleshoot_common_issues()
        return
        
    # Install dependencies
    if not install_dependencies():
        print("\nSetup failed. Run with --troubleshoot for help.")
        sys.exit(1)
        
    # Check GPU after dependencies installed
    check_gpu()
    
    # Download model if requested
    if args.download_model:
        if not download_model():
            print("\nModel download failed. Check troubleshooting guide.")
            sys.exit(1)
            
    # Run test if requested
    if args.test:
        if not test_setup():
            print("\nTest failed. Check troubleshooting guide.")
            sys.exit(1)
            
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Copy .env.example to .env and configure")
    print("2. Download model: python setup.py --download-model")
    print("3. Test setup: python setup.py --test")
    print("4. Run examples: poetry run python example_usage.py")
    print("\nFor issues, run: python setup.py --troubleshoot")


if __name__ == "__main__":
    main()