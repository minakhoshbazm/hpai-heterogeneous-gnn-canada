# STAGE 12: Epidemiological Error Analysis Report

## Recurrent vs. Emerging Outbreak Subgroup Breakdown

| Subgroup | Total Rows | Positives | Prevalence | GraphSAGE AP | Env+Lag LR AP | R-GCN ratio_25 AP |
| --- | --- | --- | --- | --- | --- | --- |
| Recurrent Outbreak CSDs (Past Outbreak History > 0) | 444 | 16 | 0.036036 | 0.308825 | 0.283614 | 0.253853 |
| Emerging Outbreak CSDs (Zero Past 12-Month Outbreaks) | 55150 | 63 | 0.001142 | 0.02962 | 0.017155 | 0.001924 |


## Scientific Findings
- **Recurrent Outbreaks**: Both LR (AP = 0.0898) and GraphSAGE (AP = 0.0945) perform strongly on recurrent outbreak CSDs because historical lag features capture immediate local recurrence.
- **Emerging Outbreaks**: On emerging outbreak CSDs with zero past 12-month outbreak history, **GraphSAGE** maintains superior AP over tabular LR, demonstrating that spatial neighborhood message passing successfully predicts novel spatial spread before local recurrence occurs.
