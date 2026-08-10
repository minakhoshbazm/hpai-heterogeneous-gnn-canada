# Scientific Claim Retraction & Revision Log

## Decision: **NO-GO FOR SUBMISSION UNTIL ALIGNMENT AND STATISTICAL ERRORS ARE CORRECTED**

### Formal Retractions & Revised Claims:

1. **Retraction: 'GraphSAGE significantly outperforms Logistic Regression'**
   - *Correction*: Standalone GraphSAGE achieves AP = **0.0903**, but non-graph PyTorch MLP achieves AP = **0.0723**. Paired bootstrap CIs confirm comparable ranking performance without universal GNN dominance.

2. **Retraction: 'Real edges significantly outperform shuffled edges'**
   - *Correction*: Across 99 multi-realization shuffled edge graph controls, mean shuffled AP is **0.0312 ± 0.0004** vs real-edge Graph-Augmented LightGBM AP **0.0318** (randomization $p = 0.0200$). While real edges produce a slight statistical edge over shuffled topology in tree models, spatial graph embeddings do **not** add substantial predictive value beyond original tabular features alone.

3. **Retraction: 'BC evaluation demonstrates spatial OOD transfer'**
   - *Correction*: Evaluating regional subgroups on prospective predictions is not OOD validation. True spatial OOD province-holdout cross-validation yields spatial APs ranging from **0.0012 to 0.0154** when entire provinces are held out during training.

4. **Retraction: '2025 is an untouched confirmatory test'**
   - *Correction*: The 2025 test set is formally categorized as a **post-hoc held-out benchmark** because earlier baseline patterns motivated model development.

