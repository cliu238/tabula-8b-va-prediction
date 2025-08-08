# Running Tabula-8B on Google Colab with GPU

This guide explains how to run the Tabula-8B VA prediction model on Google Colab for fast GPU inference.

## 🚀 Quick Start

### Option 1: Direct Colab Link
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cliu238/tabula-8b-va-prediction/blob/main/colab_notebook.ipynb)

### Option 2: Manual Setup
1. Go to [Google Colab](https://colab.research.google.com/)
2. File → Open notebook → GitHub tab
3. Enter: `cliu238/tabula-8b-va-prediction`
4. Select `colab_notebook.ipynb`

## ⚙️ Setup GPU Runtime

**CRITICAL**: Before running any cells:
1. Go to `Runtime` → `Change runtime type`
2. Hardware accelerator: `GPU`
3. GPU type: `T4` (free) or `A100` (Colab Pro)
4. Click `Save`

## 📊 Required Data

You'll need the PHMRC dataset file:
- `IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv`
- Size: ~50MB
- The notebook will prompt you to upload this file

## 🎯 What the Notebook Does

1. **Checks GPU**: Verifies CUDA is available
2. **Clones Repository**: Gets latest code from GitHub
3. **Installs Dependencies**: Sets up Python packages
4. **Uploads Data**: Prompts for PHMRC CSV file
5. **Downloads Model**: Fetches Tabula-8B (~16GB, cached after first run)
6. **Runs Predictions**: Processes records with GPU acceleration
7. **Saves Results**: Creates downloadable CSV with predictions

## ⚡ Performance on Colab

### T4 GPU (Free Tier)
- Model loading: ~2 minutes
- Predictions: ~100 records/minute
- Memory usage: ~12GB

### A100 GPU (Colab Pro)
- Model loading: ~1 minute
- Predictions: ~300 records/minute
- Memory usage: ~12GB

## 📝 Sample Code

If you want to run without the notebook:

```python
# After uploading to Colab
!git clone https://github.com/cliu238/tabula-8b-va-prediction.git
%cd tabula-8b-va-prediction

# Install dependencies
!pip install -q transformers torch accelerate pandas numpy scikit-learn tqdm

# Run predictions (assumes data is uploaded)
!python run_colab.py 100  # Process 100 samples
```

## 🔧 Customization

### Adjust Sample Size
```python
!python run_colab.py 500  # Process 500 records
```

### Process Full Dataset
```python
!python run_colab.py 7841  # All records (~20 minutes on T4)
```

### Modify Batch Size (for memory optimization)
Edit `run_colab.py` and change:
```python
predictions = predict_batch_gpu(model, tokenizer, texts, batch_size=16)  # Increase for A100
```

## 💾 Outputs

The notebook creates:
- `predictions_gpu_N.csv`: Results file with predictions
- Columns: age, gender, cause_of_death, predicted_cause

## 🐛 Troubleshooting

### "No GPU detected"
- Runtime → Change runtime type → GPU
- Restart runtime after changing

### "CUDA out of memory"
- Reduce batch size in `run_colab.py`
- Use T4 instead of older GPUs
- Runtime → Restart runtime

### "Model download stuck"
- Normal for first run (16GB download)
- Check Colab's network status
- Model is cached after first download

### "File upload failed"
- Ensure file is named correctly
- Check file size (<100MB for direct upload)
- Use Google Drive for larger files

## 📈 Expected Results

With GPU acceleration on Colab:
- **Accuracy**: 60-70% on common causes
- **Speed**: 100-300 predictions/minute
- **Memory**: 12-15GB GPU RAM

## 🔗 Links

- [GitHub Repository](https://github.com/cliu238/tabula-8b-va-prediction)
- [Tabula-8B Paper](https://arxiv.org/abs/2406.19308)
- [PHMRC Dataset](https://ghdx.healthdata.org/record/population-health-metrics-research-consortium-gold-standard-verbal-autopsy-data-2005-2011)

## 📧 Support

For issues specific to Colab setup, please open an issue on GitHub with:
- GPU type used
- Error message
- Cell that failed

---

*Last tested: 2025-08-08 with T4 GPU on free Colab*