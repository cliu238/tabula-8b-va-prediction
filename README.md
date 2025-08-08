# Tabula-8B for Verbal Autopsy Cause of Death Prediction

Zero-shot cause of death prediction from PHMRC verbal autopsy data using the Tabula-8B foundation model.

## 🎯 Project Overview

This project implements Tabula-8B, an 8-billion parameter language model specialized for tabular data, to predict cause of death from verbal autopsy (VA) records. The system processes PHMRC (Population Health Metrics Research Consortium) datasets by converting symptom data into natural language descriptions for zero-shot classification.

## 💻 System Requirements

### Minimum Requirements
- **Python**: 3.9+
- **RAM**: 16GB (with 4-bit quantization)
- **Storage**: 40GB free space
- **OS**: macOS (Apple Silicon) or Linux with CUDA

### Recommended (Your System ✅)
- **Device**: Apple M3 Max
- **RAM**: 48GB
- **Configuration**: Float16 precision (~16GB RAM usage)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd tabula-8b

# Install Poetry dependencies
poetry install
```

### 2. Test Data Preprocessing

Test the data pipeline without downloading the model:

```bash
poetry run python test_preprocessing.py
```

This will:
- Load 100 sample records from PHMRC data
- Clean and preprocess the data
- Convert records to natural language
- Save processed outputs to `data/processed/`

### 3. Download Tabula-8B Model

Download the model (16GB, takes 20-40 minutes):

```bash
poetry run python download_model_mac.py
```

**Note**: The progress bar may appear stuck at 0% initially - this is normal. The download is progressing in the background.

### 4. Run Inference

After model download, run predictions on sample data:

```bash
poetry run python demo.py
```

## 📁 Project Structure

```
tabula-8b/
├── src/
│   └── data/
│       ├── preprocessor.py    # PHMRC data cleaning
│       └── serializer.py      # Convert to natural language
├── data/
│   ├── raw/
│   │   └── PHMRC/            # Place PHMRC CSV files here
│   └── processed/            # Processed outputs
├── download_model_mac.py     # Model download script
├── test_preprocessing.py     # Test data pipeline
├── demo.py                   # Run predictions
└── TABULA_IMPLEMENTATION_PLAN.md  # Detailed plan
```

## 🔧 Configuration

The system is pre-configured for Apple M3 Max with these optimizations:
- **Float16 precision** for memory efficiency
- **Metal Performance Shaders (MPS)** for acceleration
- **Automatic batch sizing** based on available memory

To modify settings, edit `tabula_model.py`:

```python
config = TabulaConfig(
    device="auto",           # auto, cpu, or mps
    use_4bit=False,         # Enable for <16GB RAM
    max_length=2048,        # Maximum sequence length
    temperature=0.1         # Generation temperature
)
```

## 📊 Data Format

### Input: PHMRC CSV
The system expects PHMRC verbal autopsy data with columns:
- `gs_text34`: Cause of death (target)
- `g1_05`: Gender
- `g1_07a/b/c`: Age (years/months/days)
- `a1_*`: Symptom indicators
- `word_*`: Narrative-derived features

### Output: Natural Language
Records are converted to descriptive text:
```
Patient: 65 year old male | Symptoms present: fever, cough, chest pain | Illness duration: 7 days (acute)
```

## 🧪 Testing

Run the test suite:

```bash
# Unit tests
poetry run pytest tests/ -v

# Data preprocessing test
poetry run python test_preprocessing.py

# Full pipeline test (requires model)
poetry run python demo.py --sample 10
```

## 📈 Performance

On Apple M3 Max with 48GB RAM:
- **Model loading**: ~30 seconds
- **Preprocessing**: 1000 records/second
- **Inference**: 10-20 records/second
- **Memory usage**: ~16GB with float16

## 🐛 Troubleshooting

### Model Download Issues
If download is stuck or fails:
1. Check internet connection
2. Ensure 40GB free disk space
3. Cancel (Ctrl+C) and resume - downloads are resumable

### Memory Issues
If you encounter out-of-memory errors:
1. Close other applications
2. Enable 4-bit quantization in config
3. Reduce batch size

### Mac-Specific
For MPS memory issues:
```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
poetry run python your_script.py
```

## 📚 Documentation

- [Implementation Plan](TABULA_IMPLEMENTATION_PLAN.md) - Detailed technical plan
- [Setup Instructions](SETUP_INSTRUCTIONS.md) - Environment setup guide
- [API Documentation](docs/api.md) - Code reference (coming soon)

## 🤝 Contributing

1. Check existing issues in GitHub
2. Create feature branches: `feature/issue-123-description`
3. Follow code style (PEP8, type hints)
4. Add unit tests for new features
5. Keep files under 350 lines

## 📄 License

This project uses the Tabula-8B model which is released under the Llama 3 Community License. Users must comply with the license terms.

## 🔗 References

- [Tabula-8B Paper](https://arxiv.org/abs/2406.19308)
- [RTFM Repository](https://github.com/mlfoundations/rtfm)
- [PHMRC Dataset](https://ghdx.healthdata.org/record/population-health-metrics-research-consortium-gold-standard-verbal-autopsy-data-2005-2011)

## ⚡ Quick Commands Reference

```bash
# Install
poetry install

# Test data processing (no model needed)
poetry run python test_preprocessing.py

# Download model
poetry run python download_model_mac.py

# Run predictions
poetry run python demo.py

# Run tests
poetry run pytest tests/ -v

# Update dependencies
poetry update
```

---

*Last updated: 2025-08-08*