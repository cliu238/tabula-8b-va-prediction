# Tabula-8B Implementation Plan for PHMRC Cause of Death Prediction

## Executive Summary
This document outlines a comprehensive plan for implementing Tabula-8B to predict cause of death from the PHMRC (Population Health Metrics Research Consortium) verbal autopsy dataset. The plan leverages Tabula-8B's zero-shot capabilities to classify causes of death based on symptom data without requiring model fine-tuning.

---

## 1. Dataset Analysis Summary

### PHMRC Dataset Overview
- **Dataset**: IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv
- **Size**: 7,841 records × 946 columns
- **Structure**: Mixed data types (binary symptoms, categorical features, text fields)

### Key Column Identification

#### Target Variable (Label)
- **Primary**: `gs_text34` (column 4) - Gold standard cause of death text
- **Alternative**: `gs_code34` (column 3) - ICD-10 code format
- **Note**: Multiple physician annotations exist (gs_text46, gs_text55) but should be dropped to avoid data leakage

#### Features to Keep
1. **Demographics** (columns 15-27):
   - Birth/death dates (g1_01d/m/y, g1_06d/m/y)
   - Gender (g1_05)
   - Age at death (g1_07a/b/c)
   - Education, marital status (g1_08-g1_10)

2. **Symptom Indicators** (columns 72-266):
   - Initial symptoms (a1_01_1 to a1_01_14)
   - Detailed symptoms (a2_01 to a2_87_10b)
   - Additional categories (a3_*, a4_*, a5_*, a6_*, a7_*)

3. **Word Count Features** (columns 267-945):
   - Derived from narrative text (word_fever, word_cough, etc.)

#### Columns to Drop
```python
columns_to_drop = [
    # Administrative
    'site', 'module', 'newid',
    
    # Duplicate targets
    'gs_code46', 'gs_text46', 'va46',
    'gs_code55', 'gs_text55', 'va55', 'va34',
    
    # Metadata
    'gs_comorbid1', 'gs_comorbid2', 'gs_level',
    
    # Interview data (g2_*, g3_*, g4_*, g5_*)
    # All interview dates and respondent information
]
```

### Cause of Death Distribution
- Top causes: Stroke (630), Other NCDs (599), Pneumonia (540), AIDS (502)
- 34 distinct cause categories in adult dataset

---

## 2. Tabula-8B Setup Instructions

### System Requirements
- **Python**: 3.9+ (tested with 3.12.4)
- **Memory**: 
  - Full precision: 32GB RAM
  - 8-bit quantization: 16GB RAM
  - 4-bit quantization: 8GB RAM
- **Storage**: ~20GB for model weights
- **GPU** (optional): CUDA 11.8+ with 24GB+ VRAM

### Installation Steps

#### Step 1: Install Dependencies
```bash
# Using poetry (recommended)
poetry add transformers torch pillow pandas scikit-learn tqdm python-dotenv bitsandbytes accelerate

# Or update pyproject.toml and run
poetry install
```

#### Step 2: Verify Environment
```bash
poetry run python setup.py --check-only
```

#### Step 3: Download Model (Optional - auto-downloads on first use)
```bash
# Download Tabula-8B model (~16GB)
poetry run python setup.py --download-model
```

#### Step 4: Configure Environment
Create `.env` file:
```env
TABULA_MODEL_PATH=mlfoundations/tabula-8b
TABULA_CACHE_DIR=~/.cache/huggingface
TABULA_DEVICE=auto
TABULA_USE_QUANTIZATION=false
```

### Basic Model Loading
```python
from tabula_model import TabulaModel, TabulaConfig

# Configure for available resources
config = TabulaConfig(
    device="cpu",  # or "cuda" if available
    use_4bit=True,  # Enable 4-bit quantization for memory efficiency
    max_length=2048
)

# Initialize and load model
model = TabulaModel(config)
model.load_model()  # Downloads on first run
```

---

## 3. Implementation Plan

### Phase 1: Data Preparation (30 minutes)

#### Step 1.1: Dataset Exploration
```python
# File: explore_data.py
import pandas as pd

df = pd.read_csv('data/raw/PHMRC/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv')
print(f"Dataset shape: {df.shape}")
print(f"Target distribution:\n{df['gs_text34'].value_counts().head(10)}")
print(f"Missing values:\n{df.isnull().sum().describe()}")
```

#### Step 1.2: Data Preprocessor Module
```python
# File: src/data/preprocessor.py (≤200 lines)

class PHMRCPreprocessor:
    def __init__(self):
        self.column_mapping = self._create_column_mapping()
        self.columns_to_drop = self._get_columns_to_drop()
    
    def _create_column_mapping(self):
        """Map cryptic column names to descriptive ones"""
        return {
            'a1_01_1': 'fever_present',
            'a1_01_2': 'cough_present',
            'a2_01': 'illness_duration_days',
            'g1_05': 'gender',
            'g1_07a': 'age_years',
            # ... comprehensive mapping
        }
    
    def preprocess(self, df):
        """Clean and prepare dataframe"""
        # Drop administrative columns
        df = df.drop(columns=self.columns_to_drop, errors='ignore')
        
        # Rename to descriptive names
        df = df.rename(columns=self.column_mapping)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        return df
    
    def _handle_missing_values(self, df):
        """Preserve semantic meaning of missing data"""
        # Don't impute - keep as informative strings
        symptom_cols = [col for col in df.columns if 'symptom' in col]
        for col in symptom_cols:
            df[col] = df[col].fillna('not_assessed')
        return df
```

#### Step 1.3: Text Serialization Module
```python
# File: src/data/serializer.py (≤150 lines)

class VADataSerializer:
    def serialize_row(self, row):
        """Convert patient record to natural language"""
        text_parts = []
        
        # Demographics
        text_parts.append(f"Patient: {row['age_years']} year old {row['gender']}")
        
        # Present symptoms
        symptoms = self._extract_symptoms(row)
        if symptoms:
            text_parts.append(f"Symptoms: {', '.join(symptoms)}")
        
        # Duration
        if 'illness_duration_days' in row:
            text_parts.append(f"Illness duration: {row['illness_duration_days']} days")
        
        # Comorbidities if present
        if 'comorbidities' in row and row['comorbidities']:
            text_parts.append(f"Medical history: {row['comorbidities']}")
        
        return " | ".join(text_parts)
    
    def _extract_symptoms(self, row):
        """Extract present symptoms from row"""
        symptoms = []
        for col, value in row.items():
            if 'symptom' in col and value == 'Yes':
                # Extract symptom name from column
                symptom = col.replace('_symptom', '').replace('_', ' ')
                symptoms.append(symptom)
        return symptoms
```

### Phase 2: Model Integration (45 minutes)

#### Step 2.1: Tabula-8B Predictor
```python
# File: src/models/tabula_predictor.py (≤200 lines)

from tabula_model import TabulaModel, TabulaConfig

class TabulaCODPredictor:
    def __init__(self, use_quantization=True):
        """Initialize Tabula-8B for cause of death prediction"""
        self.config = TabulaConfig(
            device="auto",
            use_4bit=use_quantization,
            max_length=2048
        )
        self.model = None
        self.cause_list = self._load_cause_list()
    
    def load_model(self):
        """Load Tabula-8B model"""
        self.model = TabulaModel(self.config)
        self.model.load_model()
    
    def predict_cause(self, serialized_text):
        """Predict cause of death from serialized patient data"""
        prompt = self._create_prompt(serialized_text)
        
        # Get prediction from model
        prediction = self.model.predict(
            data=prompt,
            task="classification",
            labels=self.cause_list
        )
        
        return self._parse_prediction(prediction)
    
    def _create_prompt(self, text):
        """Create zero-shot prompt for COD prediction"""
        prompt = (
            "Based on the following patient information and symptoms, "
            "predict the most likely cause of death.\n\n"
            f"Patient Information:\n{text}\n\n"
            "Cause of death:"
        )
        return prompt
    
    def _parse_prediction(self, raw_output):
        """Extract cause and confidence from model output"""
        # Parse model response
        if isinstance(raw_output, dict):
            return {
                'cause': raw_output.get('label', 'Unknown'),
                'confidence': raw_output.get('score', 0.0)
            }
        return {'cause': str(raw_output), 'confidence': 0.5}
```

#### Step 2.2: Prediction Pipeline
```python
# File: src/pipeline/predict.py (≤150 lines)

class PredictionPipeline:
    def __init__(self):
        self.preprocessor = PHMRCPreprocessor()
        self.serializer = VADataSerializer()
        self.predictor = TabulaCODPredictor(use_quantization=True)
        
    def initialize(self):
        """Load model and prepare pipeline"""
        print("Loading Tabula-8B model...")
        self.predictor.load_model()
        print("Model loaded successfully")
    
    def predict_batch(self, df, batch_size=50):
        """Process dataframe in batches"""
        # Preprocess data
        df_clean = self.preprocessor.preprocess(df)
        
        predictions = []
        for i in range(0, len(df_clean), batch_size):
            batch = df_clean.iloc[i:i+batch_size]
            
            # Serialize each row
            for _, row in batch.iterrows():
                text = self.serializer.serialize_row(row)
                pred = self.predictor.predict_cause(text)
                predictions.append(pred)
            
            print(f"Processed {min(i+batch_size, len(df_clean))}/{len(df_clean)} records")
        
        return predictions
    
    def save_predictions(self, predictions, output_path):
        """Save predictions to CSV"""
        import pandas as pd
        df_pred = pd.DataFrame(predictions)
        df_pred.to_csv(output_path, index=False)
        print(f"Saved predictions to {output_path}")
```

### Phase 3: Evaluation (30 minutes)

#### Step 3.1: Evaluation Script
```python
# File: src/evaluation/evaluate.py (≤100 lines)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np

class Evaluator:
    def evaluate(self, true_labels, predictions):
        """Calculate evaluation metrics"""
        pred_labels = [p['cause'] for p in predictions]
        
        # Basic metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        
        # Detailed report
        report = classification_report(
            true_labels, 
            pred_labels, 
            output_dict=True
        )
        
        # Confusion matrix for top causes
        cm = confusion_matrix(true_labels, pred_labels)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm
        }
    
    def print_summary(self, metrics):
        """Print evaluation summary"""
        print(f"\n=== Evaluation Results ===")
        print(f"Overall Accuracy: {metrics['accuracy']:.2%}")
        print(f"\nTop Causes Performance:")
        
        # Show metrics for most common causes
        report = metrics['classification_report']
        for cause, stats in sorted(
            report.items(), 
            key=lambda x: x[1].get('support', 0) if isinstance(x[1], dict) else 0,
            reverse=True
        )[:10]:
            if isinstance(stats, dict) and 'precision' in stats:
                print(f"  {cause}: Precision={stats['precision']:.2f}, "
                      f"Recall={stats['recall']:.2f}, N={stats['support']}")
```

#### Step 3.2: Demo Script
```python
# File: demo.py (≤50 lines)

def run_demo(sample_size=10):
    """Run end-to-end demo on sample data"""
    print("=== Tabula-8B VA Cause of Death Prediction Demo ===\n")
    
    # Load sample data
    df = pd.read_csv('data/raw/PHMRC/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv')
    sample = df.sample(n=sample_size, random_state=42)
    
    # Initialize pipeline
    pipeline = PredictionPipeline()
    pipeline.initialize()
    
    # Make predictions
    predictions = pipeline.predict_batch(sample, batch_size=5)
    
    # Show results
    for i, (idx, row) in enumerate(sample.iterrows()):
        print(f"\n--- Record {i+1} ---")
        print(f"True Cause: {row['gs_text34']}")
        print(f"Predicted: {predictions[i]['cause']}")
        print(f"Confidence: {predictions[i]['confidence']:.2%}")
    
    # Evaluate if ground truth available
    evaluator = Evaluator()
    metrics = evaluator.evaluate(
        sample['gs_text34'].tolist(),
        predictions
    )
    evaluator.print_summary(metrics)

if __name__ == "__main__":
    run_demo()
```

### Phase 4: Testing (15 minutes)

#### Step 4.1: Unit Tests
```python
# File: tests/test_preprocessor.py

def test_column_mapping():
    """Test column name mapping"""
    preprocessor = PHMRCPreprocessor()
    assert 'fever_present' in preprocessor.column_mapping.values()

def test_drop_columns():
    """Test administrative columns are dropped"""
    preprocessor = PHMRCPreprocessor()
    df = pd.DataFrame({'site': [1], 'module': [1], 'a1_01_1': ['Yes']})
    df_clean = preprocessor.preprocess(df)
    assert 'site' not in df_clean.columns
    assert 'fever_present' in df_clean.columns

# File: tests/test_serializer.py

def test_serialization():
    """Test row serialization to text"""
    serializer = VADataSerializer()
    row = pd.Series({
        'age_years': 65,
        'gender': 'Male',
        'fever_symptom': 'Yes',
        'cough_symptom': 'No'
    })
    text = serializer.serialize_row(row)
    assert '65 year old Male' in text
    assert 'fever' in text.lower()

# File: tests/test_pipeline.py

def test_pipeline_initialization():
    """Test pipeline components initialize"""
    pipeline = PredictionPipeline()
    assert pipeline.preprocessor is not None
    assert pipeline.serializer is not None
    assert pipeline.predictor is not None
```

#### Step 4.2: Integration Test
```bash
# Run integration test
poetry run python -m pytest tests/ -v

# Run demo on small sample
poetry run python demo.py --sample 100
```

---

## 4. File Structure

```
tabula-8b/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessor.py      # Data cleaning and column mapping
│   │   └── serializer.py        # Convert rows to text
│   ├── models/
│   │   ├── __init__.py
│   │   └── tabula_predictor.py  # Tabula-8B interface
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── predict.py           # End-to-end prediction pipeline
│   └── evaluation/
│       ├── __init__.py
│       └── evaluate.py          # Metrics and evaluation
├── tests/
│   ├── test_preprocessor.py
│   ├── test_serializer.py
│   ├── test_predictor.py
│   └── test_pipeline.py
├── data/
│   └── raw/
│       └── PHMRC/
│           └── IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv
├── demo.py                       # Demo script
├── setup.py                      # Setup and environment check
├── tabula_model.py              # Core Tabula-8B wrapper
├── pyproject.toml               # Poetry dependencies
├── .env.example                 # Environment template
└── TABULA_IMPLEMENTATION_PLAN.md # This document
```

---

## 5. Execution Timeline

| Phase | Task | Duration | Output |
|-------|------|----------|--------|
| Setup | Install dependencies, download model | 20 min | Working environment |
| Phase 1 | Data preparation modules | 30 min | preprocessor.py, serializer.py |
| Phase 2 | Model integration | 45 min | tabula_predictor.py, predict.py |
| Phase 3 | Evaluation and demo | 30 min | evaluate.py, demo.py |
| Phase 4 | Testing | 15 min | Unit tests, integration test |
| **Total** | **Complete pipeline** | **~2.5 hours** | **Working COD predictor** |

---

## 6. Success Criteria

### Minimum Viable Product (MVP)
- ✅ Pipeline runs without errors on full dataset
- ✅ Produces predictions for all 34 cause categories
- ✅ Achieves >60% accuracy on common causes
- ✅ Processes 100 records in <5 minutes
- ✅ All unit tests pass

### Performance Targets
- Accuracy: >65% on test set
- Top-5 accuracy: >85%
- Processing speed: 1000 records/minute
- Memory usage: <16GB for inference

---

## 7. Key Design Decisions

### Data Processing
- **Preserve semantic meaning**: Keep "Yes/No/Don't Know" as strings
- **Descriptive naming**: Use medical terminology, not codes
- **No imputation**: Treat missing as "not_assessed"

### Model Configuration
- **Zero-shot first**: No fine-tuning initially
- **Quantization**: Use 4-bit for memory efficiency
- **Batch processing**: 50 records per batch

### Architecture
- **Modular design**: Separate concerns (data, model, pipeline)
- **KISS principle**: Simple, clear implementation
- **File size limit**: Keep all files <350 lines

---

## 8. Next Steps After MVP

1. **Optimization Phase**
   - Fine-tune column name mappings with medical experts
   - Experiment with few-shot examples
   - Optimize batch sizes and memory usage

2. **Comparison Study**
   - Benchmark against InterVA, InSilicoVA, openVA
   - Create ensemble methods
   - Analyze failure cases

3. **Production Readiness**
   - Add confidence thresholding
   - Implement uncertainty quantification
   - Create API endpoints
   - Add monitoring and logging

4. **Extended Features**
   - Multi-language support for international sites
   - Child and neonate datasets
   - Real-time prediction service

---

## 9. Troubleshooting Guide

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| Out of memory | Enable 4-bit quantization: `use_4bit=True` |
| Slow inference | Reduce batch size, use GPU if available |
| Model download fails | Check internet connection, use manual download |
| Import errors | Run `poetry install` to ensure all dependencies |
| Low accuracy | Verify column mapping, check data preprocessing |

### Getting Help
- Check `setup.py --check-only` for environment issues
- Run tests with `pytest -v` for detailed error messages
- Review logs in `.cache/huggingface/` for model issues

---

## 10. References

- [Tabula-8B Paper](https://arxiv.org/abs/2406.19308): "RTFM: Tabula-8B Zero-Shot Tabular Prediction"
- [PHMRC Dataset](https://ghdx.healthdata.org/record/population-health-metrics-research-consortium-gold-standard-verbal-autopsy-data-2005-2011)
- [Verbal Autopsy Methods](https://www.who.int/standards/classifications/other-classifications/verbal-autopsy-standards)

---

*Document Version: 1.0*  
*Last Updated: 2025-08-08*  
*Status: Ready for Implementation*