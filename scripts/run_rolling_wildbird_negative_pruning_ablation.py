#!/usr/bin/env python3
"""
scripts/run_rolling_wildbird_negative_pruning_ablation.py

Negative-Cohort Pruning Ablation under Strict Rolling-Origin Temporal Validation.

MOTIVATION
----------
The wild-bird CSD-month task has a pooled prevalence of ~0.00108 (191 positives in
176,890 test rows across 2023-2025). Absolute PR-AUC values (~0.05-0.06) are therefore
hard to interpret: a random ranker scores 0.00108, so the models already achieve ~50-58x
lift. This ablation asks whether a meaningful share of that low absolute number is caused
by *denominator dilution* — negatives drawn from CSDs that carry no poultry at all and
that no surveillance program would ever visit — rather than by a genuine ceiling on signal.

DESIGN
------
Cohorts are defined ONLY from static, label-independent covariates that are constant per
CSD across all months (verified by assertion below), so restricting to a cohort cannot
leak outbreak information or future covariates:

  full          : all CSDs (status quo reference)
  poultry_any   : chicken density > 0 OR broiler production > 0 OR turkey production > 0
  chicken_gt0   : glw2020_chicken_density_mean > 0
  chicken_gt1   : glw2020_chicken_density_mean > 1
  chicken_gt10  : glw2020_chicken_density_mean > 10

Two evaluation arms are reported for every cohort:

  ARM A "restricted"  — train on cohort rows, evaluate on cohort TEST rows only.
      Answers: how much of the low PR-AUC is mechanical denominator dilution?
      NOTE: PR-AUC is NOT comparable across arms/cohorts because the prevalence of the
      evaluation set changes. Lift over prevalence is the comparable quantity.

  ARM B "full_eval"   — train on cohort rows, evaluate on ALL test rows, with
      out-of-cohort rows forced to the bottom of the ranking (score = -inf surrogate).
      Answers: does pruning actually improve deployment screening performance?
      This arm shares one fixed evaluation set with the `full` baseline, so PR-AUC IS
      directly comparable and is the arm the bootstrap test uses.
      Out-of-cohort positives are unrecoverable in this arm — that is the real cost of
      pruning, and it is charged honestly here.

Models: Logistic Regression (replicating the project's reference LR pipeline exactly)
and XGBoost (dev/val grid selection, identical to the tree-baseline script).

Folds are identical to every other rolling evaluation in this project:
  Fold 1: DevTrain 2022-01..09, DevVal 2022-10..12, FinalTrain 2022,      Test 2023
  Fold 2: DevTrain 2022-01..2023-09, DevVal 2023-10..12, FinalTrain <=2023, Test 2024
  Fold 3: DevTrain 2022-01..2024-09, DevVal 2024-10..12, FinalTrain <=2024, Test 2025

Outputs (all new files; nothing existing is overwritten):
- results/baselines/rolling_wildbird_negative_pruning_fold_metrics.csv
- results/baselines/rolling_wildbird_negative_pruning_pooled_metrics.csv
- results/baselines/rolling_wildbird_negative_pruning_bootstrap.csv
- results/baselines/rolling_wildbird_negative_pruning_predictions.csv
- results/baselines/rolling_wildbird_negative_pruning_summary.md
- results/data_qa/rolling_wildbird_negative_pruning_cohort_audit.md
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import xgboost as xgb
except ImportError:
    print("ERROR: xgboost is not installed. Install it with: pip install xgboost")
    sys.exit(1)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

INPUT_MODEL_TABLE = "data/processed/modeling/model_table_v1_environment_with_lags_wildbird.csv"
REFERENCE_PREDS = "results/baselines/rolling_wildbird_monthly_snapshot_graphsage_vs_lr_predictions.csv"

OUT_FOLD_METRICS = "results/baselines/rolling_wildbird_negative_pruning_fold_metrics.csv"
OUT_POOLED_METRICS = "results/baselines/rolling_wildbird_negative_pruning_pooled_metrics.csv"
OUT_BOOTSTRAP = "results/baselines/rolling_wildbird_negative_pruning_bootstrap.csv"
OUT_PREDS = "results/baselines/rolling_wildbird_negative_pruning_predictions.csv"
OUT_MD = "results/baselines/rolling_wildbird_negative_pruning_summary.md"
OUT_QA_MD = "results/data_qa/rolling_wildbird_negative_pruning_cohort_audit.md"

# Identical exclusion set to the tree-baseline / GraphSAGE rolling scripts (yields 62 features)
EXCLUDE_EXPLICIT = {
    "csd_uid", "csd_name", "province_code", "province_name", "ccs_uid",
    "year", "month", "outbreak_next_month",
    "dw_fill_method", "climate_extraction_method",
    "glw2020_chicken_extraction_method", "glw2015_duck_extraction_method",
    "road_data_coverage", "outbreak_current_month",
}

# Static, label-independent columns used to define cohorts. Asserted constant per CSD.
COHORT_SOURCE_COLS = [
    "glw2020_chicken_density_mean",
    "ccs_broilers_roasters_cornish_production_kg",
    "ccs_turkey_production_kg",
]

XGB_GRID = [
    {"max_depth": 4, "learning_rate": 0.05, "min_child_weight": 10, "subsample": 0.8},
    {"max_depth": 5, "learning_rate": 0.05, "min_child_weight": 5,  "subsample": 0.8},
    {"max_depth": 4, "learning_rate": 0.10, "min_child_weight": 10, "subsample": 0.7},
    {"max_depth": 6, "learning_rate": 0.05, "min_child_weight": 10, "subsample": 0.9},
]

N_BOOT = 1000
BOOT_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Metric utilities (identical definitions to the other rolling scripts)
# ─────────────────────────────────────────────────────────────────────────────
def precision_recall_at_k(y_true, y_prob, k):
    order = np.argsort(-y_prob, kind="stable")[:k]
    top = y_true[order]
    total_pos = y_true.sum()
    prec = top.mean() if len(top) else 0.0
    rec = top.sum() / total_pos if total_pos > 0 else 0.0
    return prec, rec


def evaluate_metrics(y_true, y_prob):
    m = dict(
        pr_auc=average_precision_score(y_true, y_prob),
        roc_auc=roc_auc_score(y_true, y_prob),
    )
    for k in (50, 100, 200):
        p, r = precision_recall_at_k(y_true, y_prob, k)
        m[f"precision_at_{k}"] = p
        m[f"recall_at_{k}"] = r
    return m


def define_cohorts(df):
    """Return {cohort_name: set(csd_uid)} from static covariates only."""
    static = df.groupby("csd_uid")[COHORT_SOURCE_COLS].first()
    chicken = static["glw2020_chicken_density_mean"].fillna(0.0)
    broiler = static["ccs_broilers_roasters_cornish_production_kg"].fillna(0.0)
    turkey = static["ccs_turkey_production_kg"].fillna(0.0)

    return {
        "full":         set(static.index),
        "poultry_any":  set(static.index[(chicken > 0) | (broiler > 0) | (turkey > 0)]),
        "chicken_gt0":  set(static.index[chicken > 0]),
        "chicken_gt1":  set(static.index[chicken > 1]),
        "chicken_gt10": set(static.index[chicken > 10]),
    }


def fit_lr(df_train, df_test, feature_cols):
    """Replicates the project's reference LR pipeline exactly:
    fillna(0.0) -> StandardScaler fit on train -> class-weighted liblinear LR."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(df_train[feature_cols].fillna(0.0).values)
    X_te = scaler.transform(df_test[feature_cols].fillna(0.0).values)
    y_tr = df_train["outbreak_next_month"].values
    pos = y_tr.sum()
    pos_weight = (len(y_tr) - pos) / pos if pos > 0 else 1.0
    clf = LogisticRegression(
        max_iter=1000, class_weight={0: 1.0, 1: pos_weight},
        solver="liblinear", random_state=42,
    )
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]


def fit_xgb(df_dev_train, df_dev_val, df_final_train, df_test, feature_cols):
    """Dev/val grid + early-stopping selection, then refit on final train.
    Imputation uses training-set medians only (no test information)."""
    med_dev = df_dev_train[feature_cols].median()
    X_tr_dev = df_dev_train[feature_cols].fillna(med_dev).values
    X_val = df_dev_val[feature_cols].fillna(med_dev).values
    y_tr_dev = df_dev_train["outbreak_next_month"].values
    y_val = df_dev_val["outbreak_next_month"].values

    med_final = df_final_train[feature_cols].median()
    X_tr_final = df_final_train[feature_cols].fillna(med_final).values
    X_te = df_test[feature_cols].fillna(med_final).values
    y_tr_final = df_final_train["outbreak_next_month"].values

    pos_w_dev = (len(y_tr_dev) - y_tr_dev.sum()) / max(y_tr_dev.sum(), 1)
    pos_w_final = (len(y_tr_final) - y_tr_final.sum()) / max(y_tr_final.sum(), 1)

    dtrain_dev = xgb.DMatrix(X_tr_dev, label=y_tr_dev)
    dval = xgb.DMatrix(X_val, label=y_val)

    best_ap, best_params, best_rounds = -1.0, XGB_GRID[0], 50
    for params in XGB_GRID:
        p = {
            "objective": "binary:logistic", "eval_metric": "aucpr",
            "max_depth": params["max_depth"], "learning_rate": params["learning_rate"],
            "min_child_weight": params["min_child_weight"], "subsample": params["subsample"],
            "scale_pos_weight": pos_w_dev, "nthread": 1, "seed": 42, "verbosity": 0,
        }
        booster = xgb.train(p, dtrain_dev, num_boost_round=400, evals=[(dval, "val")],
                            early_stopping_rounds=20, verbose_eval=False)
        ap = average_precision_score(y_val, booster.predict(dval))
        if ap > best_ap:
            best_ap, best_params, best_rounds = ap, params, booster.best_iteration + 1

    p_final = {
        "objective": "binary:logistic", "eval_metric": "aucpr",
        "max_depth": best_params["max_depth"], "learning_rate": best_params["learning_rate"],
        "min_child_weight": best_params["min_child_weight"], "subsample": best_params["subsample"],
        "scale_pos_weight": pos_w_final, "nthread": 1, "seed": 42, "verbosity": 0,
    }
    booster_final = xgb.train(p_final, xgb.DMatrix(X_tr_final, label=y_tr_final),
                              num_boost_round=best_rounds, verbose_eval=False)
    probs = booster_final.predict(xgb.DMatrix(X_te))
    return probs, best_ap, {**best_params, "n_rounds": best_rounds}


def main():
    print("=" * 78)
    print("NEGATIVE-COHORT PRUNING ABLATION (Rolling-Origin, Wild-Bird CSD-Month)")
    print("=" * 78)

    df = pd.read_csv(INPUT_MODEL_TABLE)
    print(f"Loaded modeling table: {df.shape[0]:,} rows x {df.shape[1]} cols")

    feature_cols = sorted([
        c for c in df.columns
        if c not in EXCLUDE_EXPLICIT
        and not any(sub in c.lower() for sub in ["ebird", "gbif", "neighbor", "pressure"])
        and pd.api.types.is_numeric_dtype(df[c])
    ])
    assert len(feature_cols) == 62, f"Expected 62 features, got {len(feature_cols)}"
    print(f"Feature count assertion PASSED: {len(feature_cols)} features\n")

    # ── Leakage guard: cohort covariates must be static per CSD ──────────────
    print("Cohort covariate staticness audit (must be 1 distinct value per CSD):")
    for col in COHORT_SOURCE_COLS:
        n_distinct = df.groupby("csd_uid")[col].nunique(dropna=False).max()
        assert n_distinct == 1, f"LEAKAGE RISK: {col} varies within CSD ({n_distinct} values)"
        print(f"  {col:<48} max distinct/CSD = {n_distinct}  PASS")
    print()

    cohorts = define_cohorts(df)

    # ── Cohort composition audit ────────────────────────────────────────────
    csd_ever_pos = df.groupby("csd_uid")["outbreak_next_month"].max()
    total_ever_pos = int((csd_ever_pos == 1).sum())
    cohort_audit = []
    print(f"Cohort composition (total CSDs={df['csd_uid'].nunique():,}, "
          f"ever-positive CSDs={total_ever_pos}):")
    for name, members in cohorts.items():
        kept_pos_csds = int(csd_ever_pos.loc[sorted(members)].sum())
        sub = df[df["csd_uid"].isin(members)]
        cohort_audit.append(dict(
            cohort=name, n_csds=len(members),
            pct_csds_retained=100.0 * len(members) / df["csd_uid"].nunique(),
            n_rows=len(sub), n_positives=int(sub["outbreak_next_month"].sum()),
            prevalence=sub["outbreak_next_month"].mean(),
            ever_positive_csds_retained=kept_pos_csds,
            ever_positive_csds_lost=total_ever_pos - kept_pos_csds,
        ))
        print(f"  {name:<14} CSDs={len(members):>5,} ({100.0*len(members)/df['csd_uid'].nunique():>5.1f}%)  "
              f"rows={len(sub):>7,}  pos={int(sub['outbreak_next_month'].sum()):>4}  "
              f"prev={sub['outbreak_next_month'].mean():.6f}  "
              f"ever-pos CSDs kept={kept_pos_csds}/{total_ever_pos}")
    df_cohort_audit = pd.DataFrame(cohort_audit)
    print()

    folds_def = [
        {"fold": 1, "test_year": 2023,
         "dev_train": (df["year"] == 2022) & (df["month"] <= 9),
         "dev_val":   (df["year"] == 2022) & (df["month"] >= 10),
         "final_train": (df["year"] == 2022),
         "test": (df["year"] == 2023)},
        {"fold": 2, "test_year": 2024,
         "dev_train": (df["year"] <= 2022) | ((df["year"] == 2023) & (df["month"] <= 9)),
         "dev_val":   (df["year"] == 2023) & (df["month"] >= 10),
         "final_train": (df["year"] <= 2023),
         "test": (df["year"] == 2024)},
        {"fold": 3, "test_year": 2025,
         "dev_train": (df["year"] <= 2023) | ((df["year"] == 2024) & (df["month"] <= 9)),
         "dev_val":   (df["year"] == 2024) & (df["month"] >= 10),
         "final_train": (df["year"] <= 2024),
         "test": (df["year"] == 2025)},
    ]

    fold_metrics = []
    pred_frames = []

    for f_info in folds_def:
        fold, test_yr = f_info["fold"], f_info["test_year"]
        print("=" * 78)
        print(f"FOLD {fold}  (Test Year {test_yr})")
        print("=" * 78)

        df_test_full = df[f_info["test"]].copy().reset_index(drop=True)
        y_test_full = df_test_full["outbreak_next_month"].values
        n_full, pos_full = len(df_test_full), int(y_test_full.sum())
        print(f"  Full test set: {n_full:,} rows | {pos_full} pos | prev={y_test_full.mean():.6f}")

        # Per-fold prediction frame, one column per (cohort, model)
        fold_preds = df_test_full[["csd_uid", "year", "month"]].copy()
        fold_preds["actual"] = y_test_full
        fold_preds["fold"] = fold

        for cohort_name, members in cohorts.items():
            in_cohort_test = df_test_full["csd_uid"].isin(members).values
            df_dev_train = df[f_info["dev_train"] & df["csd_uid"].isin(members)].copy()
            df_dev_val = df[f_info["dev_val"] & df["csd_uid"].isin(members)].copy()
            df_final_train = df[f_info["final_train"] & df["csd_uid"].isin(members)].copy()
            df_test_c = df_test_full[in_cohort_test].copy()

            y_test_c = df_test_c["outbreak_next_month"].values
            pos_c = int(y_test_c.sum())
            print(f"\n  [{cohort_name}] train={len(df_final_train):,} rows "
                  f"({int(df_final_train['outbreak_next_month'].sum())} pos) | "
                  f"test={len(df_test_c):,} rows ({pos_c} pos, "
                  f"{pos_full - pos_c} positives excluded) | prev={y_test_c.mean():.6f}")

            if pos_c < 10:
                print(f"    SKIPPED: fewer than 10 test positives -> metrics unstable")
                continue

            # ── Logistic Regression ──────────────────────────────────────────
            lr_probs_c = fit_lr(df_final_train, df_test_c, feature_cols)

            # ── XGBoost ──────────────────────────────────────────────────────
            xgb_probs_c, xgb_val_ap, xgb_params = fit_xgb(
                df_dev_train, df_dev_val, df_final_train, df_test_c, feature_cols)

            for model_name, probs_c, val_ap, sel_params in [
                ("Logistic Regression", lr_probs_c, np.nan, ""),
                ("XGBoost", xgb_probs_c, xgb_val_ap, str(xgb_params)),
            ]:
                # ARM A: restricted evaluation (cohort test rows only)
                m_a = evaluate_metrics(y_test_c, probs_c)
                prev_c = y_test_c.mean()
                m_a.update(dict(
                    fold=fold, test_year=test_yr, cohort=cohort_name, model=model_name,
                    arm="restricted",
                    train_rows=len(df_final_train),
                    train_positives=int(df_final_train["outbreak_next_month"].sum()),
                    eval_rows=len(df_test_c), eval_positives=pos_c, prevalence=prev_c,
                    pr_auc_lift=m_a["pr_auc"] / prev_c,
                    positives_excluded=pos_full - pos_c,
                    best_val_pr_auc=val_ap, selected_params=sel_params,
                ))
                fold_metrics.append(m_a)

                # ARM B: full evaluation set, out-of-cohort rows ranked last
                probs_full = np.zeros(n_full, dtype=float)
                probs_full[in_cohort_test] = probs_c
                # Out-of-cohort rows get a score strictly below every in-cohort score
                floor = probs_c.min() if len(probs_c) else 0.0
                probs_full[~in_cohort_test] = floor - 1.0

                m_b = evaluate_metrics(y_test_full, probs_full)
                prev_full = y_test_full.mean()
                m_b.update(dict(
                    fold=fold, test_year=test_yr, cohort=cohort_name, model=model_name,
                    arm="full_eval",
                    train_rows=len(df_final_train),
                    train_positives=int(df_final_train["outbreak_next_month"].sum()),
                    eval_rows=n_full, eval_positives=pos_full, prevalence=prev_full,
                    pr_auc_lift=m_b["pr_auc"] / prev_full,
                    positives_excluded=pos_full - pos_c,
                    best_val_pr_auc=val_ap, selected_params=sel_params,
                ))
                fold_metrics.append(m_b)

                col = f"{cohort_name}__{'lr' if model_name.startswith('Logistic') else 'xgb'}"
                fold_preds[col] = probs_full

                print(f"    {model_name:<20} restricted PR-AUC={m_a['pr_auc']:.4f} "
                      f"(lift {m_a['pr_auc_lift']:.1f}x) | "
                      f"full-eval PR-AUC={m_b['pr_auc']:.4f} (lift {m_b['pr_auc_lift']:.1f}x)")

        pred_frames.append(fold_preds)
        print()

    df_fold_metrics = pd.DataFrame(fold_metrics)
    os.makedirs(os.path.dirname(OUT_FOLD_METRICS), exist_ok=True)
    df_fold_metrics.to_csv(OUT_FOLD_METRICS, index=False)
    print(f"Saved fold metrics -> {OUT_FOLD_METRICS}")

    df_preds = pd.concat(pred_frames, ignore_index=True)
    df_preds.to_csv(OUT_PREDS, index=False)
    print(f"Saved predictions -> {OUT_PREDS} ({len(df_preds):,} rows)")

    # ─────────────────────────────────────────────────────────────────────────
    # POOLED EVALUATION
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("POOLED OUT-OF-SAMPLE EVALUATION (2023-2025 combined)")
    print("=" * 78)

    y_pool = df_preds["actual"].values
    prev_pool = y_pool.mean()
    pooled_rows = []
    score_cols = [c for c in df_preds.columns
                  if c not in ("csd_uid", "year", "month", "actual", "fold")]

    for col in score_cols:
        cohort_name, model_key = col.split("__")
        model_name = "Logistic Regression" if model_key == "lr" else "XGBoost"
        p = df_preds[col].values

        # ARM B pooled (fixed full evaluation set)
        m_b = evaluate_metrics(y_pool, p)
        m_b.update(dict(cohort=cohort_name, model=model_name, arm="full_eval",
                        eval_rows=len(y_pool), eval_positives=int(y_pool.sum()),
                        prevalence=prev_pool, pr_auc_lift=m_b["pr_auc"] / prev_pool))
        pooled_rows.append(m_b)

        # ARM A pooled (restricted to cohort rows: those not floored)
        in_c = df_preds["csd_uid"].isin(cohorts[cohort_name]).values
        y_c, p_c = y_pool[in_c], p[in_c]
        prev_c = y_c.mean()
        m_a = evaluate_metrics(y_c, p_c)
        m_a.update(dict(cohort=cohort_name, model=model_name, arm="restricted",
                        eval_rows=int(in_c.sum()), eval_positives=int(y_c.sum()),
                        prevalence=prev_c, pr_auc_lift=m_a["pr_auc"] / prev_c))
        pooled_rows.append(m_a)

    df_pooled = pd.DataFrame(pooled_rows)
    df_pooled.to_csv(OUT_POOLED_METRICS, index=False)

    for model_name in ("Logistic Regression", "XGBoost"):
        print(f"\n  {model_name}")
        print(f"  {'cohort':<14} {'arm':<11} {'rows':>8} {'pos':>5} {'prev':>9} "
              f"{'PR-AUC':>8} {'lift':>8} {'P@100':>7} {'R@100':>7}")
        sub = df_pooled[df_pooled["model"] == model_name]
        for arm in ("restricted", "full_eval"):
            for _, r in sub[sub["arm"] == arm].iterrows():
                print(f"  {r['cohort']:<14} {r['arm']:<11} {r['eval_rows']:>8,} "
                      f"{r['eval_positives']:>5} {r['prevalence']:>9.6f} "
                      f"{r['pr_auc']:>8.4f} {r['pr_auc_lift']:>7.1f}x "
                      f"{r['precision_at_100']:>7.4f} {r['recall_at_100']:>7.4f}")
    print(f"\nSaved pooled metrics -> {OUT_POOLED_METRICS}")

    # ─────────────────────────────────────────────────────────────────────────
    # BASELINE REPRODUCTION CHECK vs published rolling LR
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("BASELINE REPRODUCTION CHECK")
    print("=" * 78)
    repro_note = "reference prediction file not found; check skipped"
    if os.path.exists(REFERENCE_PREDS):
        df_ref = pd.read_csv(REFERENCE_PREDS)
        ref_lr_pr = average_precision_score(df_ref["actual"].values,
                                            df_ref["wildbird_lr_prob"].values)
        mine = df_pooled[(df_pooled["cohort"] == "full") &
                         (df_pooled["model"] == "Logistic Regression") &
                         (df_pooled["arm"] == "full_eval")]["pr_auc"].values[0]
        delta = mine - ref_lr_pr
        repro_note = (f"published rolling LR pooled PR-AUC = {ref_lr_pr:.4f}; "
                      f"this script's `full` cohort LR = {mine:.4f} (delta {delta:+.5f})")
        print(f"  {repro_note}")
        if abs(delta) > 0.005:
            print("  WARNING: reproduction delta exceeds 0.005 — pipelines may have drifted.")
        else:
            print("  Reproduction within tolerance (|delta| <= 0.005).")

    # ─────────────────────────────────────────────────────────────────────────
    # CSD-CLUSTER BOOTSTRAP: each pruned cohort vs `full`, on the SAME full eval set
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"{N_BOOT} CSD-CLUSTER BOOTSTRAP (pruned cohort vs full, ARM B fixed eval set)")
    print("=" * 78)

    unique_csds = df_preds["csd_uid"].unique()
    csd_idx = df_preds.groupby("csd_uid").indices

    def boot_indices(i):
        """Regenerate resample i deterministically so every comparison sees the same
        resamples without holding 1,000 index arrays in memory at once."""
        rng_i = np.random.default_rng([BOOT_SEED, i])
        drawn = rng_i.choice(unique_csds, size=len(unique_csds), replace=True)
        return np.concatenate([csd_idx[c] for c in drawn])

    boot_rows = []
    for model_key, model_name in (("lr", "Logistic Regression"), ("xgb", "XGBoost")):
        base_col = f"full__{model_key}"
        if base_col not in df_preds.columns:
            continue
        p_base = df_preds[base_col].values
        for cohort_name in cohorts:
            if cohort_name == "full":
                continue
            col = f"{cohort_name}__{model_key}"
            if col not in df_preds.columns:
                continue
            p_alt = df_preds[col].values
            diffs = np.empty(N_BOOT)
            for i in range(N_BOOT):
                idx = boot_indices(i)
                yb = y_pool[idx]
                if yb.sum() == 0:
                    diffs[i] = 0.0
                    continue
                diffs[i] = (average_precision_score(yb, p_alt[idx]) -
                            average_precision_score(yb, p_base[idx]))
            ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
            obs = (average_precision_score(y_pool, p_alt) -
                   average_precision_score(y_pool, p_base))
            if ci_lo > 0:
                decision = "pruning statistically better"
            elif ci_hi < 0:
                decision = "pruning statistically worse"
            else:
                decision = "comparable, not proven better"
            boot_rows.append(dict(
                model=model_name, comparison=f"{cohort_name} vs full",
                observed_diff=obs, bootstrap_mean=diffs.mean(),
                ci_lower_95=ci_lo, ci_upper_95=ci_hi,
                win_rate_pct=100.0 * np.mean(diffs > 0), decision=decision,
            ))
            print(f"  {model_name:<20} {cohort_name:<14} obs={obs:+.4f}  "
                  f"CI=[{ci_lo:+.4f},{ci_hi:+.4f}]  win={100.0*np.mean(diffs>0):5.1f}%  -> {decision}")

    df_boot = pd.DataFrame(boot_rows)
    df_boot.to_csv(OUT_BOOTSTRAP, index=False)
    print(f"\nSaved bootstrap -> {OUT_BOOTSTRAP}")

    # ─────────────────────────────────────────────────────────────────────────
    # MARKDOWN SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    md = f"""# Negative-Cohort Pruning Ablation (Rolling-Origin, Wild-Bird CSD-Month)

**Question**: How much of the low absolute PR-AUC (~0.05-0.06) on this task is caused by
*denominator dilution* from CSDs that carry no poultry, versus a genuine signal ceiling?

**Protocol**: Strict 3-fold rolling-origin temporal validation (test years 2023, 2024, 2025),
62-feature set, identical splits to every other rolling evaluation in this project.

**Leakage guard**: cohorts are defined only from covariates verified constant per CSD across
all months (`{'`, `'.join(COHORT_SOURCE_COLS)}`). No outbreak label and no
time-varying covariate participates in cohort definition.

---

## 1. Cohort Composition

| Cohort | CSDs | % Retained | Rows | Positives | Prevalence | Ever-Positive CSDs Kept | Lost |
|---|---|---|---|---|---|---|---|
"""
    for _, r in df_cohort_audit.iterrows():
        md += (f"| `{r['cohort']}` | {r['n_csds']:,} | {r['pct_csds_retained']:.1f}% | "
               f"{r['n_rows']:,} | {r['n_positives']} | {r['prevalence']:.6f} | "
               f"{r['ever_positive_csds_retained']}/{total_ever_pos} | "
               f"{r['ever_positive_csds_lost']} |\n")

    md += """
---

## 2. Two Evaluation Arms

- **`restricted`** — train on cohort, evaluate on cohort test rows only. Shows how much of the
  low PR-AUC is mechanical. **PR-AUC is not comparable across cohorts here** (the evaluation
  prevalence changes); compare **lift over prevalence** instead.
- **`full_eval`** — train on cohort, evaluate on the *entire* test set with out-of-cohort rows
  forced to the bottom of the ranking. Positives living outside the cohort are unrecoverable,
  so the cost of pruning is charged honestly. This arm shares one fixed evaluation set with
  the `full` baseline, so PR-AUC **is** directly comparable — this is the arm the bootstrap uses.

---

## 3. Pooled Results (2023-2025)

"""
    for model_name in ("Logistic Regression", "XGBoost"):
        md += f"### {model_name}\n\n"
        md += ("| Cohort | Arm | Eval Rows | Pos | Prevalence | PR-AUC | Lift | ROC-AUC "
               "| P@50 | R@50 | P@100 | R@100 | P@200 | R@200 |\n")
        md += "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
        sub = df_pooled[df_pooled["model"] == model_name]
        for arm in ("restricted", "full_eval"):
            for _, r in sub[sub["arm"] == arm].iterrows():
                md += (f"| `{r['cohort']}` | {r['arm']} | {int(r['eval_rows']):,} | "
                       f"{int(r['eval_positives'])} | {r['prevalence']:.6f} | "
                       f"{r['pr_auc']:.4f} | {r['pr_auc_lift']:.1f}x | {r['roc_auc']:.4f} | "
                       f"{r['precision_at_50']:.4f} | {r['recall_at_50']:.4f} | "
                       f"{r['precision_at_100']:.4f} | {r['recall_at_100']:.4f} | "
                       f"{r['precision_at_200']:.4f} | {r['recall_at_200']:.4f} |\n")
        md += "\n"

    md += f"""---

## 4. CSD-Cluster Bootstrap ({N_BOOT} resamples, `full_eval` arm)

Each pruned cohort is compared against the `full` cohort on the identical pooled evaluation set.

| Model | Comparison | Observed PR-AUC Diff | Bootstrap Mean | 95% CI | Win Rate | Decision |
|---|---|---|---|---|---|---|
"""
    for _, r in df_boot.iterrows():
        md += (f"| {r['model']} | {r['comparison']} | {r['observed_diff']:+.4f} | "
               f"{r['bootstrap_mean']:+.4f} | [{r['ci_lower_95']:+.4f}, {r['ci_upper_95']:+.4f}] | "
               f"{r['win_rate_pct']:.1f}% | **{r['decision'].upper()}** |\n")

    md += f"""
---

## 5. Reproduction Check

{repro_note}

---

## 6. Per-Fold Detail

See `{OUT_FOLD_METRICS}`. Cohort/fold combinations with fewer than 10 test positives were
skipped rather than reported, consistent with the project's existing reliability threshold.

> [!IMPORTANT]
> **METHODOLOGICAL ASSURANCE**
> 1. Cohort membership derives only from static, label-independent covariates asserted
>    constant within each CSD; no outbreak label or future covariate is involved.
> 2. Training, hyperparameter selection, and imputation statistics are computed inside each
>    cohort from training rows only; no test-year information influences any fitted quantity.
> 3. The `full_eval` arm charges pruning for every positive it discards, so improvements
>    there are not an artifact of a shrinking evaluation set.
> 4. PR-AUC across `restricted` arms is deliberately NOT compared directly, since evaluation
>    prevalence differs by construction.
"""
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Saved summary -> {OUT_MD}")

    qa_md = f"""# Negative-Cohort Pruning: Cohort Definition Audit

**Input table**: `{INPUT_MODEL_TABLE}`
**Total CSDs**: {df['csd_uid'].nunique():,}
**Total rows**: {len(df):,}
**Ever-positive CSDs**: {total_ever_pos}

## 1. Leakage Guard: Cohort Covariate Staticness

| Column | Max Distinct Values per CSD | Result |
|---|---|---|
"""
    for col in COHORT_SOURCE_COLS:
        n_distinct = df.groupby("csd_uid")[col].nunique(dropna=False).max()
        qa_md += f"| `{col}` | {n_distinct} | {'PASS' if n_distinct == 1 else 'FAIL'} |\n"

    qa_md += """
A value of 1 confirms the covariate is constant across all months for every CSD, so cohort
membership is fixed before any temporal split and cannot encode label or future information.

## 2. Cohort Composition

| Cohort | CSDs | % Retained | Rows | Positives | Prevalence | Ever-Positive CSDs Kept | Lost |
|---|---|---|---|---|---|---|---|
"""
    for _, r in df_cohort_audit.iterrows():
        qa_md += (f"| `{r['cohort']}` | {r['n_csds']:,} | {r['pct_csds_retained']:.1f}% | "
                  f"{r['n_rows']:,} | {r['n_positives']} | {r['prevalence']:.6f} | "
                  f"{r['ever_positive_csds_retained']}/{total_ever_pos} | "
                  f"{r['ever_positive_csds_lost']} |\n")

    qa_md += f"""
## 3. Feature Set

62 features asserted on every cohort and fold (identical exclusion set to
`run_rolling_wildbird_tree_baselines_vs_graphsage.py`).

## 4. Reproduction Check

{repro_note}
"""
    os.makedirs(os.path.dirname(OUT_QA_MD), exist_ok=True)
    with open(OUT_QA_MD, "w", encoding="utf-8") as f:
        f.write(qa_md)
    print(f"Saved QA audit -> {OUT_QA_MD}")

    print("\n" + "=" * 78)
    print("COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
