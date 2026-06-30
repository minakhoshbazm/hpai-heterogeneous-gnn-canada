import os
import yaml
import argparse
import pickle
import numpy as np
import pandas as pd

from hpai_gnn.features.features import prepare_tabular_data
from hpai_gnn.models.baselines import get_baseline_model
from hpai_gnn.evaluation.metrics import calculate_metrics, print_metrics_summary, save_metrics_report

def train_and_evaluate_baselines(config_path):
    """Loads configuration, trains tabular baseline models, and evaluates them."""
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Extract config parameters
    paths = config.get('paths', {})
    processed_dir = paths.get('processed_data_dir', 'data/processed')
    output_dir = paths.get('output_dir', 'outputs/baselines')
    os.makedirs(output_dir, exist_ok=True)
    
    # Feature file path - check config or default to mock_features.csv
    feature_path = os.path.join(processed_dir, 'features', 'mock_features.csv')
    
    # Features schema definition
    feature_cols = [
        'poultry_density_proxy', 'road_density_proxy',
        'ndvi_anomaly', 'ndwi_anomaly', 'temp_anomaly', 'precip_anomaly', 'water_fraction',
        'mallard_abundance_proxy', 'canada_goose_abundance_proxy',
        'outbreak_lag_1', 'outbreak_lag_2'
    ]
    target_col = 'outbreak_next_month'
    
    # Temporal splitting definition
    # Based on labels range 2021-12 to 2025-12
    train_years = [2021, 2022, 2023]
    val_years = [2024]
    test_years = [2025]
    
    print(f"Loading and preparing feature splits from: {feature_path}")
    print(f"  - Train years: {train_years}")
    print(f"  - Val years:   {val_years}")
    print(f"  - Test years:  {test_years}")
    
    # Run the feature pipeline
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, metadata = prepare_tabular_data(
        feature_path=feature_path,
        train_years=train_years,
        val_years=val_years,
        test_years=test_years,
        feature_cols=feature_cols,
        target_col=target_col,
        scaling_method='standard'
    )
    
    print(f"Dataset summary:")
    print(f"  - Train features shape: {X_train.shape}, positive labels: {np.sum(y_train)} / {len(y_train)}")
    print(f"  - Val features shape:   {X_val.shape}, positive labels: {np.sum(y_val)} / {len(y_val)}")
    print(f"  - Test features shape:  {X_test.shape}, positive labels: {np.sum(y_test)} / {len(y_test)}")
    
    # Calculate scale_pos_weight for imbalanced learning in XGBoost
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"  - Imbalance ratio (Neg/Pos): {scale_pos_weight:.2f}")
    
    # List of baseline models to train
    model_names = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']
    results = {}
    
    for name in model_name_list_or_args(model_names):
        print(f"\nTraining baseline model: {name}...")
        
        # Instantiate model
        model = get_baseline_model(name, random_state=42, scale_pos_weight=scale_pos_weight)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Save trained model structure
        model_save_path = os.path.join(output_dir, f"{name}_model.pkl")
        with open(model_save_path, 'wb') as f_out:
            pickle.dump(model, f_out)
        print(f"Saved trained {name} model to {model_save_path}")
        
        # Evaluate on splits
        model_results = {}
        for split_name, (X, y) in [('train', (X_train, y_train)), ('val', (X_val, y_val)), ('test', (X_test, y_test))]:
            if len(X) == 0:
                continue
                
            # Get predictions and probabilities
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)[:, 1]
            
            # Compute evaluation metrics
            metrics = calculate_metrics(y, y_pred, y_prob)
            model_results[split_name] = metrics
            
            # Print console summary for validation/testing
            if split_name in ['val', 'test']:
                print_metrics_summary(metrics, title=f"{name.upper()} - {split_name.upper()} Split")
                
        # Save metrics report for the model
        metrics_save_path = os.path.join(output_dir, f"{name}_metrics.json")
        save_metrics_report(model_results, metrics_save_path)
        
        results[name] = model_results
        
    # Save the feature scaler to reuse in inference
    scaler_save_path = os.path.join(output_dir, "feature_scaler.pkl")
    with open(scaler_save_path, 'wb') as f_out:
        pickle.dump(scaler, f_out)
    print(f"\nSaved feature scaler to {scaler_save_path}")
    
    print("\nAll baseline models successfully trained and evaluated.")
    return results

def model_name_list_or_args(default_list):
    """Quick helper for iterating model names."""
    return default_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HPAI Stage 1 Tabular Baselines")
    parser.add_argument('--config', type=str, default='configs/example_local.yaml',
                        help='Path to the yaml configuration file')
    args = parser.parse_args()
    
    # Resolve relative paths from project root
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    config_file = os.path.join(base_dir, args.config)
    
    if not os.path.exists(config_file):
        # Fallback to local name check
        config_file = args.config
        
    train_and_evaluate_baselines(config_file)
