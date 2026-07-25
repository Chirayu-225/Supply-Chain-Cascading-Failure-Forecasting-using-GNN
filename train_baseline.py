"""
train_baseline.py

Trains Baseline A: a non-graph classifier predicting "will this node be
disrupted given this shock", using only hand-engineered structural features
(shortest-path distance, centrality) -- no message passing.

IMPORTANT: the label is ~0.9% positive (most cascades stay small), so:
  - We split by sim_id (not by row) so the same simulation never leaks
    across train/test.
  - We report ROC-AUC and PR-AUC (average precision), NOT accuracy --
    accuracy is meaningless here (predicting "never disrupted" scores ~99%).
  - We use class_weight="balanced" so the model doesn't just learn to
    always predict the majority class.

This uses scikit-learn (RandomForest / HistGradientBoosting) so it runs
anywhere -- swap in xgboost.XGBClassifier for a likely small accuracy bump if
you have it installed, the code is left as a note below.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.model_selection import GroupShuffleSplit


def train_and_evaluate(csv_path="output/baseline_dataset.csv", seed=42):
    df = pd.read_csv(csv_path)

    feature_cols = [c for c in df.columns if c not in ("sim_id", "node_id", "disrupted")]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["disrupted"].to_numpy()
    groups = df["sim_id"].to_numpy()

    # split by sim_id so no simulation appears in both train and test
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"Train rows: {len(X_train)} ({y_train.mean():.3%} positive)")
    print(f"Test rows:  {len(X_test)} ({y_test.mean():.3%} positive)")

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced",
            n_jobs=-1, random_state=seed,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_depth=8, class_weight="balanced", random_state=seed,
        ),
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        print(f"  ROC-AUC: {roc_auc:.4f}   PR-AUC: {pr_auc:.4f}  (baseline PR-AUC if random = {y_test.mean():.4f})")
        print(classification_report(y_test, y_pred, digits=3))

        results[name] = {"model": model, "roc_auc": roc_auc, "pr_auc": pr_auc}

    # feature importance from the RF, since it's directly interpretable --
    # useful for your report to show WHICH structural signals matter
    rf = results["RandomForest"]["model"]
    importance = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop feature importances (RandomForest):")
    print(importance.head(8))

    return results, feature_cols


# --- Note: to swap in XGBoost instead of HistGradientBoosting ---
# import xgboost as xgb
# scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
# model = xgb.XGBClassifier(n_estimators=300, max_depth=6, scale_pos_weight=scale_pos_weight,
#                            eval_metric="aucpr", n_jobs=-1, random_state=seed)
# model.fit(X_train, y_train)
# Typically gives a small (1-3 point) PR-AUC improvement over HistGradientBoosting on
# tabular data like this -- worth trying once you have it installed, but not required.


if __name__ == "__main__":
    train_and_evaluate()
