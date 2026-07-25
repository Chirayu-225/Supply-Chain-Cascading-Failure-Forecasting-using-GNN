"""
pyg_loader_example.py

Reference code showing how to convert the generated CSVs into a format ready
for PyTorch Geometric / PyTorch Geometric Temporal training.

This file is NOT run as part of the generation pipeline (torch / torch-geometric
are heavy, optional dependencies) -- it's here so you have a clear, working
starting point when you get to the modeling stage. Install with:

    pip install torch torch_geometric torch_geometric_temporal

Two framings are shown:
  1. Static graph + per-timestep node state -> use DynamicGraphTemporalSignal
     for a Temporal GNN (e.g. A3TGCN, DCRNN) predicting next-state per node.
  2. Per-simulation graph-level summary -> plain PyG `Data` objects for a
     graph-level regression/classification task (e.g. predict total cascade size
     from the seed node + graph state), good for a simpler baseline before you
     invest in the temporal model.
"""

import numpy as np
import pandas as pd


def load_raw(output_dir="output"):
    nodes = pd.read_csv(f"{output_dir}/nodes.csv")
    edges = pd.read_csv(f"{output_dir}/edges.csv")
    log = pd.read_csv(f"{output_dir}/cascade_log.csv.gz")
    meta = pd.read_csv(f"{output_dir}/cascade_meta.csv")
    return nodes, edges, log, meta


def build_static_edge_index(edges, node_id_to_idx):
    """PyG expects edge_index as a [2, num_edges] tensor of contiguous integer indices."""
    src = edges["source"].map(node_id_to_idx).to_numpy()
    dst = edges["target"].map(node_id_to_idx).to_numpy()
    edge_index = np.stack([src, dst])  # shape [2, num_edges]
    edge_attr = edges[["volume_share", "criticality", "lead_time_days"]].to_numpy(dtype=np.float32)
    return edge_index, edge_attr


def build_static_node_features(nodes, node_id_to_idx):
    """Static features (don't change across a single simulation): capacity, health, etc.
    One-hot encode tier and region since GNN layers expect numeric input."""
    nodes = nodes.set_index("node_id").loc[[k for k in node_id_to_idx.keys()]]
    tier_dummies = pd.get_dummies(nodes["tier"], prefix="tier")
    region_dummies = pd.get_dummies(nodes["region"], prefix="region")
    numeric = nodes[["capacity_score", "financial_health", "resilience", "baseline_risk"]]
    static_features = pd.concat([numeric, tier_dummies, region_dummies], axis=1)
    return static_features.to_numpy(dtype=np.float32), list(static_features.columns)


def build_temporal_signal_for_one_sim(sim_id, log, static_node_features, node_id_to_idx, max_steps=26):
    """
    Builds the per-timestep feature matrix for ONE simulation:
    combines the static node features with the time-varying state (one-hot: susceptible/disrupted/recovered).

    Returns a list of feature matrices, one per timestep -- this is the format
    torch_geometric_temporal.signal.DynamicGraphTemporalSignal expects (features list + a
    shared or per-step edge_index).
    """
    sim_log = log[log.sim_id == sim_id]
    n_nodes = len(node_id_to_idx)
    feature_snapshots = []

    for step in sorted(sim_log.step.unique()):
        step_df = sim_log[sim_log.step == step].set_index("node_id")
        state_onehot = np.zeros((n_nodes, 3), dtype=np.float32)  # susceptible/disrupted/recovered
        for node_id, idx in node_id_to_idx.items():
            state = step_df.loc[node_id, "state"] if node_id in step_df.index else 0
            state_onehot[idx, int(state)] = 1.0
        combined = np.concatenate([static_node_features, state_onehot], axis=1)
        feature_snapshots.append(combined)

    return feature_snapshots  # list of [n_nodes, n_features] arrays, one per timestep


if __name__ == "__main__":
    nodes, edges, log, meta = load_raw()
    node_id_to_idx = {nid: i for i, nid in enumerate(sorted(nodes.node_id.unique()))}

    edge_index, edge_attr = build_static_edge_index(edges, node_id_to_idx)
    static_features, feature_names = build_static_node_features(nodes, node_id_to_idx)

    print(f"edge_index shape: {edge_index.shape}")
    print(f"static_features shape: {static_features.shape}")
    print(f"feature columns: {feature_names}")

    example_sim_id = meta.iloc[meta.n_nodes_affected.idxmax()].sim_id  # pick the biggest cascade
    snapshots = build_temporal_signal_for_one_sim(int(example_sim_id), log, static_features, node_id_to_idx)
    print(f"\nExample simulation {example_sim_id}: {len(snapshots)} timesteps, "
          f"each snapshot shape {snapshots[0].shape}")
    print("\nNext step in modeling: feed `snapshots` (as a sequence) + `edge_index` into "
          "torch_geometric_temporal.signal.DynamicGraphTemporalSignal, then train an "
          "A3TGCN or DCRNN to predict snapshot[t+1] state channels from snapshot[t].")
