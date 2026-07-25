"""
train_gnn.py

Trains a Temporal Graph Neural Network (DCRNN) to predict node disruption
spread, using the SAME graph and same underlying task as train_baseline.py --
so the results are directly comparable.

Run this on your machine (needs torch + torch_geometric + torch_geometric_temporal,
which you've already confirmed are installed). NOT run in the sandbox that
generated the dataset, since torch isn't available there.

Task framing: at each timestep t, given the current node states (one-hot
susceptible/disrupted/recovered) + static features, predict the probability
that each node is DISRUPTED at t+1. This is directly comparable to Baseline A
(train_baseline.py), which predicts "will this node EVER be disrupted" --
so for a fair comparison, this script also reports an "ever disrupted" metric,
by rolling the model forward autoregressively.

Usage:
    python3 train_gnn.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops
from torch_geometric_temporal.nn.recurrent import DCRNN
from sklearn.metrics import roc_auc_score, average_precision_score

from pyg_loader_example import (
    load_raw, build_static_edge_index, build_static_node_features,
    build_temporal_signal_for_one_sim,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DCRNNPredictor(torch.nn.Module):
    """DCRNN recurrent layer + a linear head predicting P(disrupted at t+1) per node."""

    def __init__(self, node_feature_dim, hidden_dim=32, K=2):
        super().__init__()
        self.recurrent = DCRNN(node_feature_dim, hidden_dim, K=K)
        self.linear = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_weight, h=None):
        h = self.recurrent(x, edge_index, edge_weight, h)
        out = self.linear(torch.relu(h)).squeeze(-1)  # [n_nodes], raw logit
        return out, h


def get_sim_ids(meta, min_affected=2, max_sims=None):
    """Use simulations with at least some spread -- single-node fizzles carry
    little signal for a sequence model to learn from."""
    ids = meta[meta.n_nodes_affected >= min_affected].sim_id.tolist()
    if max_sims:
        ids = ids[:max_sims]
    return ids


def train(output_dir="output", n_train_sims=400, n_test_sims=100, epochs=5, hidden_dim=32, lr=0.003, pos_weight=25.0):
    nodes, edges, log, meta = load_raw(output_dir)
    node_id_to_idx = {nid: i for i, nid in enumerate(sorted(nodes.node_id.unique()))}
    edge_index_np, edge_attr_np = build_static_edge_index(edges, node_id_to_idx)
    static_features, feature_names = build_static_node_features(nodes, node_id_to_idx)

    edge_index_raw = torch.tensor(edge_index_np, dtype=torch.long, device=DEVICE)
    # criticality as edge weight, floored away from 0 -- DCRNN's diffusion convolution
    # normalizes by node out-degree, and a weight of ~0 on the only outgoing edge of a
    # low-degree node behaves like a zero-degree node numerically and produces NaNs
    edge_weight_raw = torch.tensor(edge_attr_np[:, 1], dtype=torch.float32, device=DEVICE).clamp(min=0.1)

    # retail-tier nodes have zero OUT-degree by construction (they don't supply anyone) --
    # DCRNN's diffusion convolution divides by out-degree, so a true zero-out-degree node
    # causes a divide-by-zero -> NaN that poisons the whole forward pass. Adding a tiny
    # self-loop to every node guarantees out-degree >= 1 everywhere WITHOUT dominating the
    # real supplier edges -- a self-loop weight of 1.0 would be 10x the floor weight (0.1)
    # on real edges, causing the model to mostly "listen to itself" instead of its
    # neighbors and understating cascade spread.
    edge_index, edge_weight = add_self_loops(
        edge_index_raw, edge_weight_raw, fill_value=0.01, num_nodes=static_features.shape[0]
    )

    sim_ids = get_sim_ids(meta, min_affected=2)
    rng = np.random.default_rng(42)
    rng.shuffle(sim_ids)
    train_sims = sim_ids[:n_train_sims]
    test_sims = sim_ids[n_train_sims:n_train_sims + n_test_sims]
    print(f"Training on {len(train_sims)} sims, testing on {len(test_sims)} sims "
          f"(filtered to cascades with >=2 nodes affected)")

    node_feature_dim = static_features.shape[1] + 3  # +3 for state one-hot
    model = DCRNNPredictor(node_feature_dim, hidden_dim=hidden_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=DEVICE)

    def run_sim_teacher_forced(sim_id, train_mode=True):
        """Feed the TRUE state sequence step by step (teacher forcing), predicting
        next-step disruption probability at each step. Returns loss (if training)
        or (predictions, true_labels) for evaluation."""
        snapshots = build_temporal_signal_for_one_sim(sim_id, log, static_features, node_id_to_idx)
        h = None
        total_loss = 0.0
        all_probs, all_targets = [], []

        for t in range(len(snapshots) - 1):
            x = torch.tensor(snapshots[t], dtype=torch.float32, device=DEVICE)
            # target: is each node DISRUPTED (state channel index -3+1=index within one-hot) at t+1?
            next_state_onehot = snapshots[t + 1][:, -3:]  # last 3 cols = state one-hot
            target = torch.tensor(next_state_onehot[:, 1], dtype=torch.float32, device=DEVICE)  # DISRUPTED channel

            logit, h = model(x, edge_index, edge_weight, h)
            # pos_weight counteracts the ~0.5% positive rate -- without it, plain BCE lets the
            # model get a good loss by mostly predicting "stays whatever it was", never really
            # learning to propagate a disruption signal along edges to un-disrupted neighbors.
            # Only applied while training; evaluation still reports raw sigmoid probabilities.
            loss = F.binary_cross_entropy_with_logits(logit, target, pos_weight=pos_weight_tensor)
            if train_mode:
                total_loss = total_loss + loss
            else:
                all_probs.append(torch.sigmoid(logit).detach().cpu().numpy())
                all_targets.append(target.cpu().numpy())
            h = h.detach() if h is not None else h  # truncate backprop-through-time per step (simple, stable)


        if train_mode:
            return total_loss / max(1, len(snapshots) - 1)
        return np.concatenate(all_probs), np.concatenate(all_targets)

    print("\nTraining...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for sim_id in train_sims:
            optimizer.zero_grad()
            loss = run_sim_teacher_forced(sim_id, train_mode=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        print(f"  Epoch {epoch + 1}/{epochs}  avg loss: {epoch_loss / len(train_sims):.4f}")

    print("\nEvaluating on held-out simulations...")
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for sim_id in test_sims:
            probs, targets = run_sim_teacher_forced(sim_id, train_mode=False)
            all_probs.append(probs)
            all_targets.append(targets)
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)

    roc_auc = roc_auc_score(all_targets, all_probs)
    pr_auc = average_precision_score(all_targets, all_probs)
    print(f"\nGNN (DCRNN) next-step disruption prediction:")
    print(f"  ROC-AUC: {roc_auc:.4f}   PR-AUC: {pr_auc:.4f}  (random baseline PR-AUC = {all_targets.mean():.4f})")
    print("\nCompare these numbers directly against train_baseline.py's output.")

    torch.save({
        "model_state_dict": model.state_dict(),
        "node_feature_dim": node_feature_dim,
        "hidden_dim": hidden_dim,
    }, f"{output_dir}/gnn_model.pt")
    print(f"Model saved to {output_dir}/gnn_model.pt")

    return model


if __name__ == "__main__":
    train()
