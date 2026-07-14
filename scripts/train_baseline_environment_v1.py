import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

def train_baselines(input_path, metrics_out, preds_out):
    print("Training Spatiotemporal Baseline Models (V1 Environment Only)...")
    print("================================================================")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    # Read the dataset
    df = pd.read_csv(input_path, dtype={'csd_uid': str})
    
    # Define splits
    train_df = df[df['year'].isin([2022, 2023, 2024])].copy()
    test_df = df[df['year'] == 2025].copy()
    
    # Define columns to exclude from features
    exclude_cols = [
        'csd_uid', 'csd_name', 'province_code', 'province_name', 'year', 'month',
        'outbreak_next_month', 'ccs_uid', 'dw_fill_method', 'climate_extraction_method'
    ]
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Keep only numeric features
    numeric_features = train_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = numeric_features
    
    # Separate features and target
    X_train = train_df[feature_cols].astype(float)
    y_train = train_df['outbreak_next_month'].astype(int)
    
    X_test = test_df[feature_cols].astype(float)
    y_test = test_df['outbreak_next_month'].astype(int)
    
    train_pos = int(y_train.sum())
    test_pos = int(y_test.sum())
    
    print(f"Dataset split summary:")
    print(f"  - Train split: {len(X_train)} rows, {train_pos} positives ({train_pos / len(X_train) * 100:.4f}%)")
    print(f"  - Test split: {len(X_test)} rows, {test_pos} positives ({test_pos / len(X_test) * 100:.4f}%)")
    print(f"  - Number of features: {len(feature_cols)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Models to train
    models = {
        'LogisticRegression': LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        )
    }
    
    metrics_list = []
    predictions_dict = {
        'csd_uid': test_df['csd_uid'].values,
        'csd_name': test_df['csd_name'].values,
        'province_code': test_df['province_code'].values,
        'year': test_df['year'].values,
        'month': test_df['month'].values,
        'actual': y_test.values
    }
    
    print("\nTraining models...")
    for model_name, clf in models.items():
        print(f"  - Fitting {model_name}...")
        clf.fit(X_train_scaled, y_train)
        
        # Get positive class probabilities
        probs = clf.predict_proba(X_test_scaled)[:, 1]
        predictions_dict[f'{model_name}_prob'] = probs
        
        # Calculate standard metrics
        ap = average_precision_score(y_test, probs)
        roc_auc = roc_auc_score(y_test, probs)
        
        # Calculate recall & precision at top K = 50, 100, 200
        # Sort predictions descending
        sorted_indices = np.argsort(probs)[::-1]
        sorted_actuals = y_test.values[sorted_indices]
        
        k_metrics = {}
        for k in [50, 100, 200]:
            top_k_actuals = sorted_actuals[:k]
            pos_in_top_k = int(top_k_actuals.sum())
            precision_at_k = pos_in_top_k / k
            recall_at_k = pos_in_top_k / test_pos if test_pos > 0 else 0.0
            
            k_metrics[f'precision_at_{k}'] = precision_at_k
            k_metrics[f'recall_at_{k}'] = recall_at_k
            
        metrics_dict = {
            'model': model_name,
            'ap': ap,
            'roc_auc': roc_auc,
            'precision_at_50': k_metrics['precision_at_50'],
            'recall_at_50': k_metrics['recall_at_50'],
            'precision_at_100': k_metrics['precision_at_100'],
            'recall_at_100': k_metrics['recall_at_100'],
            'precision_at_200': k_metrics['precision_at_200'],
            'recall_at_200': k_metrics['recall_at_200'],
            'feature_count': len(feature_cols)
        }
        metrics_list.append(metrics_dict)
        
    metrics_df = pd.DataFrame(metrics_list)
    preds_df = pd.DataFrame(predictions_dict)
    
    # Save outputs
    os.makedirs(os.path.dirname(metrics_out), exist_ok=True)
    os.makedirs(os.path.dirname(preds_out), exist_ok=True)
    
    metrics_df.to_csv(metrics_out, index=False)
    preds_df.to_csv(preds_out, index=False)
    
    # Print metrics by model
    print("\n================== Baseline Metrics Summary ==================")
    print(metrics_df.to_string(index=False))
    print("==============================================================")
    
    # Print top 20 highest risk predictions for 2025
    # We will show the highest predictions from the tree model (RandomForest)
    print("\nTop 20 Highest Risk Predictions (RandomForest model, 2025):")
    top_rf = preds_df.sort_values(by='RandomForest_prob', ascending=False).head(20)
    for idx, (_, row) in enumerate(top_rf.iterrows()):
        print(f"  {idx+1:2d}. CSD {row['csd_uid']} ({row['csd_name']}), M {int(row['month'])}: prob={row['RandomForest_prob']:.4f} (Actual Outbreak: {int(row['actual'])})")
        
    print("\n==============================================================")
    print("Baseline training completed successfully.")

if __name__ == '__main__':
    default_input = 'data/processed/modeling/model_table_v1_environment_train_ready.csv'
    default_metrics = 'results/baselines/environment_v1_baseline_metrics.csv'
    default_preds = 'results/baselines/environment_v1_test_predictions.csv'
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else default_input
    metrics_file = sys.argv[2] if len(sys.argv) > 2 else default_metrics
    preds_file = sys.argv[3] if len(sys.argv) > 3 else default_preds
    
    try:
        train_baselines(input_file, metrics_file, preds_file)
    except Exception as e:
        print(f"Execution failed: {e}", file=sys.stderr)
        sys.exit(1)
