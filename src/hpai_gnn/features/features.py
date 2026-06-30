import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_feature_dataset(feature_path):
    """Loads the compiled feature dataset."""
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature dataset not found at: {feature_path}")
    return pd.read_csv(feature_path)

def get_year_from_month(month_str):
    """Helper to extract year integer from YYYY-MM string."""
    return int(month_str.split('-')[0])

def train_val_test_split_temporal(df, train_years, val_years, test_years):
    """
    Splits the dataframe into train, validation, and test sets based on years.
    Prevents temporal data leakage by doing a rigid chronological split.
    
    Parameters:
    - df: DataFrame containing a 'month' column in format 'YYYY-MM'
    - train_years: list of integers (e.g. [2021, 2022, 2023])
    - val_years: list of integers (e.g. [2024])
    - test_years: list of integers (e.g. [2025])
    
    Returns:
    - train_df, val_df, test_df
    """
    # Extract year
    df_copy = df.copy()
    df_copy['year'] = df_copy['month'].apply(get_year_from_month)
    
    train_df = df_copy[df_copy['year'].isin(train_years)].copy()
    val_df = df_copy[df_copy['year'].isin(val_years)].copy()
    test_df = df_copy[df_copy['year'].isin(test_years)].copy()
    
    # Drop the temporary year column
    train_df = train_df.drop(columns=['year'])
    val_df = val_df.drop(columns=['year'])
    test_df = test_df.drop(columns=['year'])
    
    return train_df, val_df, test_df

def scale_features(train_df, val_df, test_df, feature_cols, method='standard'):
    """
    Fits a scaler on the training set and transforms train, validation, and test sets.
    This strictly avoids feature scaling leakage from validation/test sets back into training.
    
    Parameters:
    - train_df, val_df, test_df: DataFrames for each partition
    - feature_cols: list of column names to scale
    - method: 'standard' for StandardScaler, 'minmax' for MinMaxScaler
    
    Returns:
    - scaled_train_df, scaled_val_df, scaled_test_df, scaler
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
        
    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()
    
    # Fit on train features
    scaler.fit(train_df[feature_cols])
    
    # Transform all splits
    train_scaled[feature_cols] = scaler.transform(train_df[feature_cols])
    
    if len(val_df) > 0:
        val_scaled[feature_cols] = scaler.transform(val_df[feature_cols])
    if len(test_df) > 0:
        test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])
        
    return train_scaled, val_scaled, test_scaled, scaler

def prepare_tabular_data(feature_path, train_years, val_years, test_years, feature_cols, target_col='outbreak_next_month', scaling_method='standard'):
    """
    Helper function to load, split, scale, and extract feature matrices/vectors.
    Handles dropping NaNs in the target column (which occur at the temporal boundary).
    """
    # 1. Load data
    df = load_feature_dataset(feature_path)
    
    # 2. Split temporally
    train_df, val_df, test_df = train_val_test_split_temporal(df, train_years, val_years, test_years)
    
    # 3. Handle target NaNs (e.g. the last month in the dataset has no next month label)
    train_df = train_df.dropna(subset=[target_col])
    val_df = val_df.dropna(subset=[target_col]) if len(val_df) > 0 else val_df
    test_df = test_df.dropna(subset=[target_col]) if len(test_df) > 0 else test_df
    
    # 4. Scale features
    train_scaled, val_scaled, test_scaled, scaler = scale_features(
        train_df, val_df, test_df, feature_cols, method=scaling_method
    )
    
    # 5. Extract X and y
    X_train = train_scaled[feature_cols].values
    y_train = train_scaled[target_col].values.astype(int)
    
    X_val = val_scaled[feature_cols].values if len(val_scaled) > 0 else np.array([])
    y_val = val_scaled[target_col].values.astype(int) if len(val_scaled) > 0 else np.array([])
    
    X_test = test_scaled[feature_cols].values if len(test_scaled) > 0 else np.array([])
    y_test = test_scaled[target_col].values.astype(int) if len(test_scaled) > 0 else np.array([])
    
    # Keep metadata for alignment/tracking
    metadata = {
        'train_meta': train_scaled[['cd_uid', 'month']].reset_index(drop=True),
        'val_meta': val_scaled[['cd_uid', 'month']].reset_index(drop=True) if len(val_scaled) > 0 else pd.DataFrame(),
        'test_meta': test_scaled[['cd_uid', 'month']].reset_index(drop=True) if len(test_scaled) > 0 else pd.DataFrame()
    }
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, metadata
