import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Conditional imports for XGBoost and LightGBM
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

def get_baseline_model(model_name, random_state=42, scale_pos_weight=1.0):
    """
    Factory function to instantiate Stage 1 tabular baseline models.
    Supports: 'logistic_regression', 'random_forest', 'xgboost', 'lightgbm'.
    
    Imbalances are handled via class_weight='balanced' or scale_pos_weight.
    """
    model_name = model_name.lower().strip()
    
    if model_name == 'logistic_regression':
        return LogisticRegression(
            class_weight='balanced', 
            max_iter=2000, 
            random_state=random_state,
            solver='lbfgs'
        )
        
    elif model_name == 'random_forest':
        return RandomForestClassifier(
            n_estimators=150, 
            class_weight='balanced', 
            random_state=random_state,
            n_jobs=-1
        )
        
    elif model_name == 'xgboost':
        if XGBClassifier is None:
            raise ImportError("xgboost package is not installed. Run 'pip install xgboost'")
        return XGBClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=6,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric='logloss',
            use_label_encoder=False,
            n_jobs=-1
        )
        
    elif model_name == 'lightgbm':
        if LGBMClassifier is None:
            raise ImportError("lightgbm package is not installed. Run 'pip install lightgbm'")
        return LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=6,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1
        )
        
    else:
        raise ValueError(
            f"Unknown model name: {model_name}. "
            f"Available options: 'logistic_regression', 'random_forest', 'xgboost', 'lightgbm'"
        )
