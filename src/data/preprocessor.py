"""
PHMRC Data Preprocessor for Tabula-8B

Handles cleaning and preprocessing of PHMRC verbal autopsy data
for use with Tabula-8B model.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class PHMRCPreprocessor:
    """Preprocessor for PHMRC verbal autopsy datasets."""
    
    def __init__(self):
        """Initialize preprocessor with column mappings."""
        self.columns_to_drop = self._get_columns_to_drop()
        self.column_mapping = self._create_column_mapping()
        
    def _get_columns_to_drop(self) -> List[str]:
        """Get list of columns to drop from dataset."""
        return [
            # Administrative columns
            'site', 'module', 'newid',
            
            # Duplicate target variables (keep only gs_text34 or gs_code34)
            'gs_code46', 'gs_text46', 'va46',
            'gs_code55', 'gs_text55', 'va55',
            'va34',  # Alternative encoding of target
            
            # Gold standard metadata
            'gs_comorbid1', 'gs_comorbid2', 'gs_level',
            
            # Birth and death dates (keep age but not raw dates)
            'g1_01d', 'g1_01m', 'g1_01y',  # Birth dates
            'g1_06d', 'g1_06m', 'g1_06y',  # Death dates
            
            # Interview administrative data (dates and process info)
            'g2_01', 'g2_02', 'g2_03ad', 'g2_03am', 'g2_03ay',
            'g2_03bd', 'g2_03bm', 'g2_03by', 'g2_03cd', 'g2_03cm', 'g2_03cy',
            'g2_03dd', 'g2_03dm', 'g2_03dy', 'g2_03ed', 'g2_03em', 'g2_03ey',
            'g2_03fd', 'g2_03fm', 'g2_03fy',
            
            # Additional g2 interview metadata
            'g2_04', 'g2_05', 'g2_06', 'g2_07', 'g2_08', 'g2_09',
            
            # Respondent information (not about deceased)
            'g3_01', 'g3_02', 'g3_03', 'g3_04', 'g3_05',
            
            # Interview location and process
            'g4_01', 'g4_02', 'g4_03a', 'g4_03b', 'g4_04', 'g4_05', 
            'g4_06', 'g4_07', 'g4_08', 'g4_09', 'g4_10',
            
            # Interview completion metadata
            'g5_01d', 'g5_01m', 'g5_01y', 'g5_02', 'g5_03d', 'g5_03m', 'g5_03y',
            'g5_04a', 'g5_04b', 'g5_04c', 'g5_05', 'g5_06a', 'g5_06b', 'g5_07', 'g5_08',
            'g5_09', 'g5_10', 'g5_11', 'g5_12'
        ]
    
    def _create_column_mapping(self) -> Dict[str, str]:
        """Create mapping from cryptic column names to descriptive ones."""
        return {
            # Demographics
            'g1_05': 'gender',
            'g1_07a': 'age_years',
            'g1_07b': 'age_months',
            'g1_07c': 'age_days',
            'g1_08': 'marital_status',
            'g1_09': 'education_level',
            'g1_10': 'language',
            
            # Common symptoms - keeping original column names for consistency
            # Note: We'll keep most symptom columns with their original names
            # to preserve the data structure Tabula-8B expects
            'a1_01_1': 'fever',
            'a1_01_2': 'cough',
            'a1_01_3': 'difficulty_breathing',
            'a1_01_4': 'diarrhea',
            'a1_01_5': 'vomiting',
            'a1_01_6': 'abdominal_pain',
            'a1_01_7': 'headache',
            'a1_01_8': 'chest_pain',
            'a1_01_9': 'muscle_pain',
            'a1_01_10': 'joint_pain',
            'a1_01_11': 'skin_rash',
            'a1_01_12': 'weight_loss',
            'a1_01_13': 'night_sweats',
            'a1_01_14': 'loss_of_consciousness',
            
            # Additional important symptoms
            'a2_03': 'fever_continuous',
            'a2_04': 'fever_night_sweats',
            'a2_05': 'cough_productive',
            'a2_06': 'cough_blood',
            'a2_07': 'breathlessness',
            'a2_08': 'chest_pain_sudden',
            'a2_09': 'diarrhea_blood',
            'a2_10': 'diarrhea_persistent',
            
            # Duration and onset
            'a2_01': 'illness_duration_days',
            'a2_02': 'sudden_onset',
            
            # Target variable
            'gs_text34': 'cause_of_death',
            'gs_code34': 'cause_of_death_code'
        }
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load PHMRC CSV file."""
        df = pd.read_csv(filepath, low_memory=False)
        print(f"Loaded data with shape: {df.shape}")
        return df
    
    def preprocess(self, df: pd.DataFrame, keep_target: bool = True) -> pd.DataFrame:
        """
        Preprocess PHMRC dataframe.
        
        Args:
            df: Raw PHMRC dataframe
            keep_target: Whether to keep the target variable
            
        Returns:
            Preprocessed dataframe
        """
        df = df.copy()
        
        # Drop administrative columns
        cols_to_drop = [col for col in self.columns_to_drop if col in df.columns]
        df = df.drop(columns=cols_to_drop, errors='ignore')
        print(f"Dropped {len(cols_to_drop)} administrative columns")
        
        # Rename columns to be descriptive
        df = df.rename(columns=self.column_mapping)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Convert symptom columns to descriptive values
        df = self._convert_symptoms(df)
        
        # Process demographics
        df = self._process_demographics(df)
        
        if not keep_target and 'cause_of_death' in df.columns:
            df = df.drop(columns=['cause_of_death', 'cause_of_death_code'], errors='ignore')
        
        print(f"Preprocessed data shape: {df.shape}")
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataframe."""
        # For symptoms: Convert various missing representations to 'not_assessed'
        missing_values = ['', 'NA', 'N/A', 'Unknown', "Don't Know", 'DK', -99, -999]
        
        # Suppress FutureWarning for replace
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning)
            for col in df.columns:
                if col.startswith('a') or col in ['fever', 'cough', 'difficulty_breathing']:
                    df[col] = df[col].replace(missing_values, 'not_assessed')
                    df[col] = df[col].fillna('not_assessed')
        
        return df
    
    def _convert_symptoms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert symptom columns to descriptive values."""
        # Keep original format for Tabula-8B compatibility
        symptom_mapping = {
            1: 'Yes',
            0: 'No',
            '1': 'Yes',
            '0': 'No',
            'yes': 'Yes',
            'no': 'No',
            'Y': 'Yes',
            'N': 'No',
            'y': 'Yes',
            'n': 'No'
        }
        
        symptom_cols = [col for col in df.columns if 
                       col.startswith('a') or 
                       col in self.column_mapping.values()]
        
        for col in symptom_cols:
            if col in df.columns:
                df[col] = df[col].replace(symptom_mapping)
        
        return df
    
    def _process_demographics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process demographic variables."""
        # Make a copy to avoid fragmentation warning
        df = df.copy()
        
        # Process age into single field
        if 'age_years' in df.columns:
            # Convert to numeric, handling any string values
            df['age'] = pd.to_numeric(df['age_years'], errors='coerce').fillna(0)
            if 'age_months' in df.columns:
                age_months = pd.to_numeric(df['age_months'], errors='coerce').fillna(0)
                df['age'] += age_months / 12
            if 'age_days' in df.columns:
                age_days = pd.to_numeric(df['age_days'], errors='coerce').fillna(0)
                df['age'] += age_days / 365
            
            # Drop individual age columns
            df = df.drop(columns=['age_years', 'age_months', 'age_days'], errors='ignore')
        
        # Process gender
        if 'gender' in df.columns:
            gender_mapping = {
                1: 'male',
                2: 'female',
                'M': 'male',
                'F': 'female',
                'Male': 'male',
                'Female': 'female'
            }
            df['gender'] = df['gender'].replace(gender_mapping)
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns (excluding target)."""
        exclude = ['cause_of_death', 'cause_of_death_code']
        return [col for col in df.columns if col not in exclude]
    
    def get_symptom_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of symptom columns."""
        return [col for col in df.columns if 
                col.startswith('a') or 
                col in ['fever', 'cough', 'difficulty_breathing', 'diarrhea',
                       'vomiting', 'abdominal_pain', 'headache', 'chest_pain']]
    
    def create_summary_stats(self, df: pd.DataFrame) -> Dict:
        """Create summary statistics for the dataset."""
        stats = {
            'n_records': len(df),
            'n_features': len(self.get_feature_columns(df)),
            'n_symptom_features': len(self.get_symptom_columns(df))
        }
        
        if 'cause_of_death' in df.columns:
            stats['n_causes'] = df['cause_of_death'].nunique()
            stats['top_causes'] = df['cause_of_death'].value_counts().head(10).to_dict()
        
        if 'gender' in df.columns:
            stats['gender_distribution'] = df['gender'].value_counts().to_dict()
        
        if 'age' in df.columns:
            stats['age_stats'] = {
                'mean': df['age'].mean(),
                'median': df['age'].median(),
                'min': df['age'].min(),
                'max': df['age'].max()
            }
        
        return stats