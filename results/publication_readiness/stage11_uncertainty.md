# STAGE 11: Statistical Uncertainty Report

## Month-Block Bootstrap Confidence Intervals (1,000 Resamples)

| Model | Observed AP | Mean Bootstrap AP | 95% CI Lower | 95% CI Upper | Std Err |
| --- | --- | --- | --- | --- | --- |
| GraphSAGE | 0.079051 | 0.076775 | 0.013896 | 0.126011 | 0.029274 |
| Env+Lag LR | 0.065369 | 0.064324 | 0.010739 | 0.101121 | 0.021852 |
| R-GCN ratio_25 | 0.045183 | 0.051151 | 0.001075 | 0.131714 | 0.035608 |


## Paired Bootstrap Model Comparisons

| Comparison | Observed AP Diff | 95% CI | Win Rate | Decision |
| --- | --- | --- | --- | --- |
| GraphSAGE vs Env+Lag LR | 0.013682 | [-0.0172, +0.0397] | 81.4% | Comparable |
| R-GCN ratio_25 vs Env+Lag LR | -0.020186 | [-0.0633, +0.0698] | 27.3% | Lower AP Ranking |


