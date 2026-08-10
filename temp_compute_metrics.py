import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

baseline_path = r'd:/gdrive/models/hpai-heterogeneous-gnn-canada/results/baselines/environment_with_lags_v1_strict_test_predictions.csv'
base = pd.read_csv(baseline_path)
# Identify probability column
prob_col = None
for c in base.columns:
    if 'prob' in c.lower():
        prob_col = c
        break
if prob_col is None:
    raise ValueError('Probability column not found')

y = base['actual']
prob = base[prob_col]
pr_auc = average_precision_score(y, prob)
roc_auc = roc_auc_score(y, prob)
print('Baseline PR-AUC:', pr_auc)
print('Baseline ROC-AUC:', roc_auc)
