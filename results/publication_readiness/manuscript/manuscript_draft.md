# Manuscript Draft: Spatiotemporal Graph Neural Networks for Retrospective Forecasting of Avian Influenza Outbreaks in Canada

## Title Candidates
1. *Spatiotemporal Graph Neural Networks for Retrospective Outbreak Risk Forecasting of Highly Pathogenic Avian Influenza in Poultry: A Canadian Nationwide Study*
2. *Evaluating Spatial Graph Neural Networks vs. Interpretable Lag Baselines under Severe Class Imbalance in Avian Influenza Epidemiology*

## Structured Abstract
**Background**: Highly Pathogenic Avian Influenza (HPAI) H5N1 poses severe threats to commercial poultry production and global public health. Predicting monthly regional outbreak risk under extreme class imbalance (prevalence ~0.14%) is critical for veterinary disease surveillance.
**Methods**: We developed a prospective-style spatiotemporal forecasting framework evaluating 237,538 region-month records across 5,054 Census Subdivisions in Canada (2022–2025). We evaluated Spatial GraphSAGE, Spatiotemporal Directed Relational GCNs (R-GCN), and Environment+Lag Logistic Regression across 3 frozen epidemiological cohorts.
**Results**: Ensembled Spatial GraphSAGE achieved the highest broad risk ranking (**Average Precision = 0.0791, ROC-AUC = 0.9263**, 55.6x lift over random), significantly outperforming tabular Environment+Lag Logistic Regression (AP = 0.0654, p < 0.05). Two-phase transfer learning from wild-bird surveillance increased R-GCN AP by 14.6x (0.0031 to 0.0452).
**Conclusion**: Spatial graph message passing provides genuine predictive value for regional outbreak forecasting, particularly for emerging non-recurrent outbreaks.
