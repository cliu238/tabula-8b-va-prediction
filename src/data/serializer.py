"""
VA Data Serializer for Tabula-8B

Converts PHMRC tabular data to natural language descriptions
for optimal Tabula-8B model performance.
"""

import pandas as pd
from typing import List, Dict, Optional


class VADataSerializer:
    """Serialize VA data rows to natural language for Tabula-8B."""
    
    def __init__(self, verbose: bool = False):
        """
        Initialize serializer.
        
        Args:
            verbose: Whether to include all features or just key ones
        """
        self.verbose = verbose
        
    def serialize_row(self, row: pd.Series) -> str:
        """
        Convert a patient record to natural language description.
        
        Args:
            row: Single row from preprocessed PHMRC dataframe
            
        Returns:
            Natural language description of the patient
        """
        parts = []
        
        # Demographics section
        demo_text = self._serialize_demographics(row)
        if demo_text:
            parts.append(demo_text)
        
        # Symptoms section
        symptoms_text = self._serialize_symptoms(row)
        if symptoms_text:
            parts.append(symptoms_text)
        
        # Duration and onset
        duration_text = self._serialize_duration(row)
        if duration_text:
            parts.append(duration_text)
        
        # Word count features (narrative-derived)
        if self.verbose:
            narrative_text = self._serialize_narrative_features(row)
            if narrative_text:
                parts.append(narrative_text)
        
        return " | ".join(parts)
    
    def _serialize_demographics(self, row: pd.Series) -> str:
        """Serialize demographic information."""
        demo_parts = []
        
        # Age
        if 'age' in row and pd.notna(row['age']):
            age = int(row['age']) if row['age'] >= 1 else f"{row['age']*12:.1f} month"
            demo_parts.append(f"{age} year old" if isinstance(age, int) else age)
        
        # Gender
        if 'gender' in row and pd.notna(row['gender']) and row['gender'] != 'not_assessed':
            demo_parts.append(str(row['gender']))
        
        # Education
        if 'education_level' in row and pd.notna(row['education_level']) and self.verbose:
            demo_parts.append(f"education: {row['education_level']}")
        
        # Marital status
        if 'marital_status' in row and pd.notna(row['marital_status']) and self.verbose:
            demo_parts.append(f"marital status: {row['marital_status']}")
        
        if demo_parts:
            return "Patient: " + " ".join(demo_parts)
        return ""
    
    def _serialize_symptoms(self, row: pd.Series) -> str:
        """Serialize symptom information."""
        # Key symptoms to always include
        key_symptoms = [
            'fever', 'cough', 'difficulty_breathing', 'diarrhea', 'vomiting',
            'abdominal_pain', 'headache', 'chest_pain', 'loss_of_consciousness',
            'skin_rash', 'weight_loss', 'night_sweats'
        ]
        
        present_symptoms = []
        absent_symptoms = []
        
        # Collect symptoms
        for col in row.index:
            if col in key_symptoms or (self.verbose and col.startswith('a')):
                value = row[col]
                if value == 'yes':
                    symptom_name = col.replace('_', ' ')
                    present_symptoms.append(symptom_name)
                elif value == 'no' and self.verbose:
                    symptom_name = col.replace('_', ' ')
                    absent_symptoms.append(symptom_name)
        
        # Build symptoms text
        symptoms_parts = []
        
        if present_symptoms:
            symptoms_parts.append(f"Symptoms present: {', '.join(present_symptoms)}")
        
        if absent_symptoms and self.verbose:
            symptoms_parts.append(f"Symptoms absent: {', '.join(absent_symptoms[:5])}")  # Limit for brevity
        
        if not present_symptoms and not self.verbose:
            symptoms_parts.append("No major symptoms reported")
        
        return " | ".join(symptoms_parts) if symptoms_parts else ""
    
    def _serialize_duration(self, row: pd.Series) -> str:
        """Serialize illness duration and onset information."""
        duration_parts = []
        
        # Illness duration
        if 'illness_duration_days' in row and pd.notna(row['illness_duration_days']):
            try:
                days = int(row['illness_duration_days'])
                if days == 0:
                    duration_parts.append("Sudden death")
                elif days == 1:
                    duration_parts.append("Illness duration: 1 day")
                elif days < 7:
                    duration_parts.append(f"Illness duration: {days} days (acute)")
                elif days < 30:
                    duration_parts.append(f"Illness duration: {days} days (subacute)")
                else:
                    duration_parts.append(f"Illness duration: {days} days (chronic)")
            except:
                pass
        
        # Sudden onset
        if 'sudden_onset' in row:
            if row['sudden_onset'] == 'yes':
                duration_parts.append("Sudden onset")
            elif row['sudden_onset'] == 'no' and self.verbose:
                duration_parts.append("Gradual onset")
        
        return " | ".join(duration_parts) if duration_parts else ""
    
    def _serialize_narrative_features(self, row: pd.Series) -> str:
        """Serialize word count features from narrative text."""
        word_features = []
        
        # Look for word_* columns with high counts
        word_cols = [col for col in row.index if col.startswith('word_')]
        
        if word_cols:
            significant_words = []
            for col in word_cols:
                if pd.notna(row[col]) and row[col] > 0:
                    word = col.replace('word_', '').replace('_', ' ')
                    count = int(row[col])
                    if count >= 2:  # Only include words mentioned multiple times
                        significant_words.append(word)
            
            if significant_words:
                # Limit to top 10 most relevant medical terms
                medical_terms = ['fever', 'cough', 'pain', 'blood', 'breathing',
                               'heart', 'accident', 'injury', 'pregnancy', 'delivery']
                relevant_words = [w for w in significant_words if any(term in w for term in medical_terms)]
                
                if relevant_words:
                    word_features.append(f"Key narrative terms: {', '.join(relevant_words[:10])}")
        
        return " | ".join(word_features) if word_features else ""
    
    def serialize_batch(self, df: pd.DataFrame, show_progress: bool = True) -> List[str]:
        """
        Serialize multiple rows to natural language.
        
        Args:
            df: Preprocessed PHMRC dataframe
            show_progress: Whether to show progress bar
            
        Returns:
            List of serialized text descriptions
        """
        serialized = []
        
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(df.iterrows(), total=len(df), desc="Serializing")
        else:
            iterator = df.iterrows()
        
        for idx, row in iterator:
            text = self.serialize_row(row)
            serialized.append(text)
        
        return serialized
    
    def create_prompt(self, serialized_text: str, task: str = "classification") -> str:
        """
        Create a prompt for Tabula-8B model.
        
        Args:
            serialized_text: Serialized patient description
            task: Type of task ('classification' or 'explanation')
            
        Returns:
            Formatted prompt for the model
        """
        if task == "classification":
            prompt = (
                "Based on the following patient information and symptoms, "
                "predict the most likely cause of death.\n\n"
                f"{serialized_text}\n\n"
                "Cause of death:"
            )
        elif task == "explanation":
            prompt = (
                "Analyze the following patient case and explain the likely "
                "cause of death with reasoning.\n\n"
                f"{serialized_text}\n\n"
                "Analysis:"
            )
        else:
            prompt = serialized_text
        
        return prompt
    
    def format_for_training(self, row: pd.Series, target_col: str = 'cause_of_death') -> Dict[str, str]:
        """
        Format a row for fine-tuning Tabula-8B.
        
        Args:
            row: Single row from preprocessed dataframe
            target_col: Name of target column
            
        Returns:
            Dictionary with 'input' and 'output' keys
        """
        # Remove target from row for serialization
        row_features = row.drop(target_col) if target_col in row else row
        
        serialized = self.serialize_row(row_features)
        prompt = self.create_prompt(serialized, task="classification")
        
        return {
            'input': prompt,
            'output': row[target_col] if target_col in row else "Unknown"
        }