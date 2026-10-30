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


def build_model_ready_edges(edge_index_np, edge_attr_np, num_nodes, device):
    """Builds the final edge_index/edge_weight tensors used by both train_gnn.py and
    predict_cascade.py -- centralized here so the two stay in sync automatically.

    Includes an out-degree dilution correction. DCRNN's diffusion convolution
    normalizes each source node's outgoing signal to sum to (approximately) a fixed
    budget, regardless of how many downstream neighbors it has -- so a node with 8
    real neighbors effectively transmits a much weaker signal to any ONE of them than
    a node with only 1 neighbor does, even if the individual edge criticality is far
    higher. Diagnostic testing found exactly this: a node with a single low-criticality
    (0.005) edge produced a correctly detected downstream signal, while a different
    node with 8 edges -- including one at criticality 0.72, the highest in the
    comparison -- produced no detectable signal at all for any of its real neighbors.

    Fix: pre-multiply each edge's weight by its source node's out-degree (count of
    real outgoing edges), so that after DCRNN's internal per-source normalization,
    the effective signal reaching each individual neighbor is restored closer to that
    edge's true criticality, independent of how many neighbors the source has.
    """
    import torch
    from torch_geometric.utils import add_self_loops

    edge_index_raw = torch.tensor(edge_index_np, dtype=torch.long, device=device)
    edge_weight_raw = torch.tensor(edge_attr_np[:, 1], dtype=torch.float32, device=device).clamp(min=0.1)

    # out-degree (real edges only, not self-loops) per source node
    sources = edge_index_raw[0]
    out_degree_count = torch.zeros(num_nodes, device=device)
    out_degree_count.scatter_add_(0, sources, torch.ones_like(sources, dtype=torch.float32))
    out_degree_count = out_degree_count.clamp(min=1.0)  # nodes with 0 real out-edges: no-op

    edge_weight_corrected = edge_weight_raw * out_degree_count[sources]

    # tiny self-loop, still needed to avoid divide-by-zero for true zero-out-degree
    # (retail-tier) nodes -- scaled down further since real edges are now larger in
    # magnitude than before, to keep the self-loop from becoming relatively dominant
    edge_index, edge_weight = add_self_loops(
        edge_index_raw, edge_weight_corrected, fill_value=0.01, num_nodes=num_nodes
    )
    return edge_index, edge_weight


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
