"""
Tabula-8B Data Formatter for VA Prediction

Formats PHMRC tabular data in the CSV format expected by Tabula-8B model.
Replaces the natural language serializer with proper tabular formatting.
"""

import pandas as pd
from typing import List, Dict, Optional


class TabulaVAFormatter:
    """Format VA data for Tabula-8B model input."""
    
    def __init__(self):
        """Initialize formatter with VA-specific configurations."""
        self.task_description = (
            "Task: Predict cause of death based on verbal autopsy interview data.\n"
            "The data contains demographic information, symptoms, and medical history "
            "collected through structured interviews with family members."
        )
        
    def format_for_tabula(
        self,
        df: pd.DataFrame,
        target_column: str = 'cause_of_death',
        include_metadata: bool = True
    ) -> str:
        """
        Format preprocessed VA data for Tabula-8B input.
        
        Args:
            df: Preprocessed PHMRC DataFrame
            target_column: Name of target column (will be excluded from features)
            include_metadata: Whether to include column descriptions
            
        Returns:
            CSV-formatted string ready for Tabula-8B
        """
        # Remove target column if present
        feature_df = df.drop(columns=[target_column], errors='ignore')
        
        # Build the prompt
        prompt_parts = []
        
        # Add task description
        prompt_parts.append(self.task_description)
        
        # Add column metadata if requested
        if include_metadata:
            prompt_parts.append(self._generate_column_metadata(feature_df))
        
        # Add the actual data in CSV format
        prompt_parts.append("\nData (CSV format):")
        prompt_parts.append(feature_df.to_csv(index=False))
        
        # Add prediction instruction
        prompt_parts.append(f"\nPredict the '{target_column}' value for each row.")
        prompt_parts.append("Possible causes include common conditions like TB, AIDS, malaria, "
                          "cardiovascular disease, respiratory infections, and injuries.")
        
        return "\n".join(prompt_parts)
    
    def _generate_column_metadata(self, df: pd.DataFrame) -> str:
        """Generate column descriptions for VA data."""
        metadata_parts = ["\nColumn Information:"]
        
        # Group columns by category
        demographics = ['age', 'gender', 'marital_status', 'education_level']
        symptoms = [col for col in df.columns if col.startswith('a') or 
                   col in ['fever', 'cough', 'difficulty_breathing', 'diarrhea', 
                          'vomiting', 'abdominal_pain', 'headache', 'chest_pain']]
        word_features = [col for col in df.columns if col.startswith('word_')]
        
        # Add demographic columns
        for col in demographics:
            if col in df.columns:
                dtype = str(df[col].dtype)
                unique = df[col].nunique()
                metadata_parts.append(f"- {col}: demographic ({dtype}, unique: {unique})")
        
        # Add symptom columns (summarized)
        symptom_count = len([col for col in symptoms if col in df.columns])
        if symptom_count > 0:
            metadata_parts.append(f"- {symptom_count} symptom indicators: Yes/No/not_assessed values")
        
        # Add word features (summarized)
        word_count = len([col for col in word_features if col in df.columns])
        if word_count > 0:
            metadata_parts.append(f"- {word_count} word frequency features from narrative text")
        
        # Add other columns
        categorized = set(demographics + symptoms + word_features)
        other_cols = [col for col in df.columns if col not in categorized]
        if other_cols:
            metadata_parts.append(f"- {len(other_cols)} additional clinical and contextual features")
        
        return "\n".join(metadata_parts)
    
    def format_batch(
        self,
        df: pd.DataFrame,
        batch_size: int = 10,
        target_column: str = 'cause_of_death'
    ) -> List[str]:
        """
        Format multiple rows for batch prediction.
        
        Args:
            df: Preprocessed DataFrame
            batch_size: Number of rows per batch
            target_column: Target column name
            
        Returns:
            List of formatted prompts for each batch
        """
        prompts = []
        
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            prompt = self.format_for_tabula(batch_df, target_column)
            prompts.append(prompt)
        
        return prompts
    
    def prepare_single_record(
        self,
        row: pd.Series,
        column_names: List[str],
        target_column: str = 'cause_of_death'
    ) -> str:
        """
        Format a single patient record for prediction.
        
        Args:
            row: Single row from DataFrame
            column_names: List of column names
            target_column: Target column to exclude
            
        Returns:
            CSV-formatted string for single prediction
        """
        # Convert Series to DataFrame for consistent formatting
        df = pd.DataFrame([row])
        
        # Remove target if present
        if target_column in df.columns:
            df = df.drop(columns=[target_column])
        
        # Format as CSV
        prompt = (
            "Predict cause of death for the following patient:\n\n"
            f"{df.to_csv(index=False)}\n"
            "Cause of death:"
        )
        
        return prompt