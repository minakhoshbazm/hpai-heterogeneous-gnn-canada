# STAGE 19: Reproducibility Manifest & Command Guide

## Master Execution Pipeline
```bash
# 1. Run Data Preparation & Model Training
python scripts/run_pseudoabsence_ablation_rgcn_transfer.py
python scripts/run_rolling_wildbird_supervised_transfer_graphsage.py

# 2. Run Publication-Readiness Stage Audits
python scratch/execute_priority_1.py
python scratch/execute_priority_2.py
python scratch/execute_priority_3.py
```
