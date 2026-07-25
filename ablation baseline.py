"""
ablation_baseline.py

Ablation study: retrain Baseline A WITHOUT `shortest_path_len_from_seed`
(the feature that took ~88% of RandomForest's importance in the full model).

Why this matters for your report: if PR-AUC collapses without it, that tells
you the baseline's strong score was mostly "I know how many hops away this
node is from the shock" -- a fairly trivial, almost rule-based signal -- rather
than genuine learning from the richer structural/financial features. It also
sets a fairer bar for the GNN: a GNN that also basically just learns "distance
from seed" isn't offering much beyond what a simple BFS would give you, so
this number matters for judging whether the GNN's advantage is real structure
learning or the same shortcut in a fancier wrapper.

Run this AFTER baseline_features.py has produced output/baseline_dataset.csv
(same file train_baseline.py uses -- no need to regenerate anything).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupShuffleSplit


def evaluate_feature_set(df, feature_cols, label, seed=42):
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["disrupted"].to_numpy()
    groups = df["sim_id"].to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = HistGradientBoostingClassifier(max_depth=8, class_weight="balanced", random_state=seed)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    print(f"{label:35s}  ROC-AUC: {roc_auc:.4f}   PR-AUC: {pr_auc:.4f}")
    return roc_auc, pr_auc


if __name__ == "__main__":
    df = pd.read_csv("output/baseline_dataset.csv")
    all_features = [c for c in df.columns if c not in ("sim_id", "node_id", "disrupted")]
    no_distance_features = [c for c in all_features if c != "shortest_path_len_from_seed"]

    print("Ablation: does the baseline's strength come mainly from graph-distance-to-seed?\n")
    evaluate_feature_set(df, all_features, "Full feature set (with distance)")
    evaluate_feature_set(df, no_distance_features, "Without shortest_path_len_from_seed")

    print("\nInterpretation guide:")
    print("- Large PR-AUC drop without distance  -> baseline mostly learned 'count hops from shock'")
    print("- Small/no drop                        -> other features (centrality, resilience, etc.)")
    print("                                          carry real signal on their own")
    print("Compare the 'without distance' number to the GNN's PR-AUC for the fairest comparison,")
    print("since the GNN doesn't get an explicit hop-count feature handed to it either.")
